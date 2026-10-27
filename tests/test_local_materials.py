import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.display import value
from common.geometry import bounds
from rgb.materials import prepare, rank, MaterialClassifier
from rgb.camera import make_frame
from rgb.store import now
from test_local import local, openrouter_local


def manifest():
    return {'input_size':2,'embedding_size':2,'labels':['plastic','metal'],
            'minimum_similarity':.2,'minimum_margin':.02,
            'processor':{'do_resize':True,'do_rescale':True,'do_normalize':True,'resample':2,
                         'rescale_factor':1/255,'image_mean':[.5]*3,'image_std':[.5]*3}}


def result():
    return {'status':'identified','label':'plastic','best_candidate':'plastic',
            'similarity':.38,'margin':.04,'candidates':[{'label':'plastic','similarity':.38}],
            'score_type':'uncalibrated_image_text_similarity','model':'google/siglip2-base-patch16-224',
            'revision':'b'*40,'manifest_sha256':'c'*64,'artifact_sha256':'d'*64,'inference_ms':12}


def material_detection(local):
    store,camera,frame,detected,runtime,app=local
    classifier=Mock()
    classifier.classify.side_effect=lambda crop:result()
    detection=dict(detected['detections'][0])
    runtime._material_pass(classifier,frame,[detection])
    runtime._attach_material(frame,detection)
    return detection,classifier


def test_siglip_preprocessing_preserves_rgb_and_exact_normalization():
    red=np.full((3,4,3),[255,0,0],np.uint8)
    tensor=prepare(red,manifest())
    assert tensor.shape==(1,3,2,2) and tensor.dtype==np.float32
    assert np.array_equal(tensor[0,:,0,0],[1,-1,-1])
    with pytest.raises(ValueError):
        prepare(red,{**manifest(),'processor':{**manifest()['processor'],'resample':0}})


def test_material_acceptance_requires_both_similarity_and_separation():
    m=manifest()
    accepted=rank([1,0],np.eye(2,dtype=np.float32),m)
    assert accepted['status']=='identified' and accepted['label']=='plastic'
    text=np.array([[1,0],[.999,.04471018]],np.float32)
    ambiguous=rank([1,0],text,m)
    assert ambiguous['status']=='uncertain' and ambiguous['label'] is None
    assert ambiguous['best_candidate']=='plastic'
    low=rank([1,0],np.array([[.1,.9949874],[-.3,.9539392]],np.float32),m)
    assert low['status']=='uncertain'
    with pytest.raises(ValueError):rank([np.nan,0],np.eye(2),m)
    with pytest.raises(ValueError):rank([0,0],np.eye(2),m)
    assert 'confidence' not in accepted
    rejected=rank([1,0],np.eye(2),{**m,'rejection_labels':['plastic']})
    assert rejected['status']=='uncertain' and rejected['label'] is None


def test_material_worker_uses_crops_without_creating_records(local):
    store,camera,frame,detected,runtime,app=local
    detection,classifier=material_detection(local)
    x1,y1,x2,y2=bounds(detection['box'],40,30)
    assert np.array_equal(classifier.classify.call_args.args[0],frame.color_rgb[y1:y2,x1:x2])
    assert detection['local_material']['source_frame_id']==frame.id
    assert store.storage_status()['observations']==0 and store.storage_status()['asset_bytes']==0
    assert value(detection,None,{'display_mode':'materials'})=={
        'label':'plastic','confidence':None,'provider':'siglip2'}
    assert value(detection,None,{'display_mode':'objects','vlm_threshold':.65})['label']=='bottle'


def test_material_worker_caps_work_and_skips_tiny_or_stale_crops(local):
    store,camera,frame,detected,runtime,app=local
    detections=[{**detected['detections'][0],'track_id':str(uuid4())} for _ in range(12)]
    classifier=Mock();classifier.classify.side_effect=lambda crop:result()
    runtime._material_pass(classifier,frame,detections)
    assert classifier.classify.call_count==4
    classifier.reset_mock();camera.freshness_s=-1
    runtime._material_pass(classifier,frame,detections)
    classifier.classify.assert_not_called()
    camera.freshness_s=2
    tiny={**detected['detections'][0],'track_id':str(uuid4()),'box':[.1,.1,.2,.2]}
    runtime._material_pass(classifier,frame,[tiny])
    classifier.classify.assert_not_called()


@pytest.mark.parametrize('tiny_box',[[0,0,.2,1],[0,0,1,.2],[0,0,.2,.2]])
def test_undersized_crops_cannot_starve_eligible_tracks_across_passes(local,tiny_box):
    store,camera,frame,detected,runtime,app=local
    base=detected['detections'][0]
    tiny=[{**base,'track_id':str(uuid4()),'box':tiny_box} for _ in range(4)]
    eligible=[{**base,'track_id':str(uuid4())} for _ in range(6)]
    classifier=Mock();classifier.classify.side_effect=lambda crop:result()
    for expected in [4,2,0]:
        classifier.reset_mock()
        runtime._material_pass(classifier,frame,tiny+eligible)
        assert classifier.classify.call_count==expected
    assert set(runtime.material_cache)=={d['track_id'] for d in eligible}
    for detection in eligible:
        runtime._attach_material(frame,detection)
        assert detection['local_material']['source_frame_id']==frame.id
    assert store.storage_status()['observations']==0 and store.storage_status()['asset_bytes']==0


@pytest.mark.parametrize('change',['age','session','calibration','box'])
def test_stale_or_mismatched_material_results_are_not_attached(local,change):
    store,camera,frame,detected,runtime,app=local
    detection,_=material_detection(local)
    detection.pop('local_material')
    cached=runtime.material_cache[detection['track_id']]
    if change=='age':cached['received_monotonic_s']=time.monotonic()-6
    if change=='session':cached['session_id']=str(uuid4())
    if change=='calibration':cached['calibration_sha256']='f'*64
    if change=='box':cached['source_box']=[0,0,.1,.1]
    runtime._attach_material(frame,detection)
    assert 'local_material' not in detection


def test_local_material_evidence_survives_archive_queries_and_typed_exports(local):
    store,camera,frame,detected,runtime,app=local
    frame.metadata['device']={'name':'Femto Bolt'}
    detection,_=material_detection(local)
    event=runtime.archive(frame,{**detected,'detections':[detection]})
    reading=event['detections'][0]['local_material']
    assert reading['crop_asset']['sha256']==reading['crop_sha256']
    response=app.test_client().get('/api/events/'+event['event_id']+'/assets/material_crop_0')
    assert response.status_code==200 and hashlib.sha256(response.data).hexdigest()==reading['crop_sha256']
    crop=cv2.imdecode(np.frombuffer(response.data,np.uint8),cv2.IMREAD_COLOR)
    x1,y1,x2,y2=bounds(detection['box'],40,30)
    assert np.array_equal(crop,cv2.cvtColor(frame.color_rgb[y1:y2,x1:x2],cv2.COLOR_RGB2BGR))
    records=app.test_client().get('/api/review/records?material=plastic').json['records']
    assert records[0]['local_material']['source_frame_id']==frame.id
    assert records[0]['camera_name']=='Femto Bolt'
    page=app.test_client().get('/api/review/records?format=parquet')
    row=pq.read_table(pa.BufferReader(page.data)).to_pylist()[0]
    assert json.loads(row['local_material_json'])==reading and row['camera_name']=='Femto Bolt'
    store.update_settings({'display_mode':'materials'})
    assert store.get(event['event_id'])['detections'][0]['display_label']=='plastic'
    response=app.test_client().get('/api/review/export/parquet')
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert json.loads(archive.read('manifest.json'))['schema_version']==9
        rows=pq.read_table(pa.BufferReader(archive.read('material_readings.parquet'))).to_pylist()
        assert rows[0]['provider']=='siglip2' and rows[0]['confidence'] is None
        assert rows[0]['similarity']==.38 and rows[0]['accepted'] is True
        detections=pq.read_table(pa.BufferReader(archive.read('detections.parquet'))).to_pylist()
        assert json.loads(detections[0]['local_material_json'])==reading
    response.close()
    evidence=app.test_client().get('/api/review/evidence?event_ids='+event['event_id'])
    with zipfile.ZipFile(io.BytesIO(evidence.data)) as archive:
        assert 'assets/'+reading['crop_asset']['file'] in archive.namelist()
    evidence.close()


@pytest.mark.parametrize('provider',['bedrock','openrouter'])
def test_local_material_evidence_and_provider_audit_survive_shared_review(openrouter_local,provider):
    (store,camera,frame,detected,runtime,app),session=openrouter_local
    model='us.amazon.nova-2-lite-v1:0' if provider=='bedrock' else 'amazon/nova-2-lite-v1'
    store.update_settings({'selected_provider':provider,'selected_model':model})
    runtime.update_capture_settings({'capture_paused':True})
    detection,_=material_detection((store,camera,frame,detected,runtime,app))
    event=runtime.archive(frame,{**detected,'detections':[detection]})
    original=event['detections'][0]['local_material']
    assert store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    calls=runtime.client.calls if provider=='bedrock' else session.calls
    other=session.calls if provider=='bedrock' else runtime.client.calls
    assert len(calls)==1 and not other
    projected=store.get(event['event_id'])['detections'][0]
    assert projected['local_material']==original
    assert projected['enhancement']['endpoint']==provider and projected['enhancement']['model']==model
    assert store.settings()['capture_paused'] is True and store.storage_status()['observations']==1
    store.update_settings({'display_mode':'materials'})
    assert store.get(event['event_id'])['detections'][0]['label_provider']=='siglip2'
    with app.test_client().get('/api/review/export/arrow') as response:
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert json.loads(archive.read('manifest.json'))['schema_version']==9
            attempts=pa.ipc.open_file(pa.BufferReader(archive.read('model_attempts.arrow'))).read_all().to_pylist()
            assert attempts[-1]['endpoint']==provider and attempts[-1]['outcome']=='success'
            assert attempts[-1]['ordered_image_sha256']==attempts[0]['ordered_image_sha256']
            if provider=='openrouter':
                assert attempts[-1]['resolved_model']==model and attempts[-1]['cost']==.0003
            materials=pa.ipc.open_file(pa.BufferReader(archive.read('material_readings.arrow'))).read_all().to_pylist()
            assert {r['provider'] for r in materials}=={'siglip2','vlm'}
            local_row=next(r for r in materials if r['provider']=='siglip2')
            assert json.loads(local_row['value_json'])==original and local_row['confidence'] is None
    html=app.test_client().get('/').data
    assert b'id="vlm-provider"' in html and b'id="local-material-state"' in html
    assert b'id="detection-indicator"' in html and b'id="bedrock-model"' not in html


def test_missing_crop_evidence_cannot_be_archived_as_identified(local):
    store,camera,frame,detected,runtime,app=local
    detection,_=material_detection(local)
    runtime.material_inputs.clear()
    event=runtime.archive(frame,{**detected,'detections':[detection]})
    material=event['detections'][0]['local_material']
    assert material['status']=='evidence_unavailable' and material['label'] is None
    assert not app.test_client().get('/api/review/records?material=plastic').json['records']
    assert value(event['detections'][0],None,{'display_mode':'materials'})['label']=='Material uncertain'


def test_manual_capture_defaults_to_material_annotated_frame_and_requires_pinned_detection(local):
    store,camera,frame,detected,runtime,app=local
    detection,_=material_detection(local)
    camera.latest_detection=(frame,{**detected,'detections':[detection]})
    newer=make_frame(frame.color_rgb,frame.depth_native,frame.ir_native,frame.depth_registered,
                     {**frame.metadata,'frame_id':str(uuid4()),'sequence':2,'received_at':now(),
                      'received_monotonic_s':time.monotonic()})
    camera.publish(newer)
    response=app.test_client().post('/api/camera/observations',json={})
    assert response.status_code==201 and response.json['frame_id']==frame.id
    assert store.raw(response.json['event_id'])['detections'][0]['local_material']['crop_asset']
    pinned=app.test_client().post('/api/camera/observations',json={'frame_id':newer.id})
    assert pinned.status_code==503 and store.event_for_frame(newer.id) is None
    camera.latest_detection=(newer,{**detected,'detections':[detection]})
    pinned=app.test_client().post('/api/camera/observations',json={'frame_id':newer.id})
    assert pinned.status_code==201
    assert pinned.json['frame_id']==newer.id
    saved=store.raw(pinned.json['event_id'])['detections'][0]['local_material']
    assert saved['source_frame_id']==frame.id and saved['crop_asset']


def test_uncertain_material_does_not_become_a_label_or_probability(local):
    store,camera,frame,detected,runtime,app=local
    detection,_=material_detection(local)
    detection['local_material'].update(status='uncertain',label=None,margin=.001)
    event=runtime.archive(frame,{**detected,'detections':[detection]})
    store.update_settings({'display_mode':'materials'})
    projected=store.get(event['event_id'])['detections'][0]
    assert projected['display_label']=='Material uncertain' and projected['display_confidence'] is None
    assert not app.test_client().get('/api/review/records?material=plastic').json['records']


@pytest.mark.skipif(not (Path.home()/'.cache/bolt-rgb/models/siglip2-base-224/manifest.json').exists(),
                    reason='Optional privately installed local model')
def test_installed_siglip_artifacts_run_offline_and_check_integrity(tmp_path):
    pytest.importorskip('openvino')
    directory=Path.home()/'.cache/bolt-rgb/models/siglip2-base-224'
    classifier=MaterialClassifier(directory)
    reading=classifier.classify(np.full((48,64,3),128,np.uint8))
    assert np.isfinite(reading['similarity']) and reading['score_type']=='uncalibrated_image_text_similarity'
    m=json.loads((directory/'manifest.json').read_text())
    m['files']['vision.xml']='0'*64
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'vision.xml').write_bytes((directory/'vision.xml').read_bytes())
    with pytest.raises(ValueError,match='checksum'):
        MaterialClassifier(tmp_path)
