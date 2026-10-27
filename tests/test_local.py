import hashlib
import io
import json
import threading
import time
import zipfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rgb.camera import Camera, CameraBusy, CameraLock, CameraUnavailable, FrameExpired, depth_summary, make_frame
from rgb.enrichment import validate_readings
from rgb.export import build_export
from rgb.local_runtime import LocalRuntime
from rgb.local_store import ArchiveConflict, FALLBACK_MODEL, PRIMARY_MODEL, LocalStore, QueueFull, encode
from rgb.store import StaleEvaluation, detection_id, now
from rgb.web import create_app
from rgb.vlm_providers import OpenRouterProvider


def reading(index=0):
    return {'box_index':index,'label':'plastic bottle','confidence':.9,'comparison':'refines',
            'reason':'Visible bottle shape','materials':[{'label':'plastic','confidence':.8,'reason':'Translucent walls'}]}


@pytest.fixture
def local(tmp_path):
    store=LocalStore(tmp_path)
    camera=Camera(max_frames=2,max_bytes=1000000)
    calibration={'schema_version':1,'test':True}
    metadata={'frame_id':str(uuid4()),'session_id':str(uuid4()),'source_id':'femto-bolt',
              'sequence':1,'received_at':now(),'received_monotonic_s':time.monotonic(),
              'timestamps_us':{'color':1000,'depth':1000,'ir':1000},'timestamp_domain':'test',
              'profiles':{'color':{'width':40,'height':30},'depth':{'width':10,'height':8},'ir':{'width':10,'height':8}},
              'depth_scale_mm':1,'calibration':calibration,'calibration_sha256':hashlib.sha256(encode(calibration).encode()).hexdigest()}
    frame=make_frame(np.full((30,40,3),80,np.uint8),np.full((8,10),1234,np.uint16),
                     np.arange(80,dtype=np.uint16).reshape(8,10),np.full((30,40),1234,np.uint16),metadata)
    camera.publish(frame);camera.session_id=metadata['session_id']
    result={'frame_id':frame.id,'session_id':camera.session_id,'observed_at':now(),'detector_ms':10,
            'detections':[{'class_id':1,'label':'bottle','confidence':.8,'box':[.1,.2,.9,.8],'track_id':str(uuid4()),
                           'depth':depth_summary(frame.depth_registered,[.1,.2,.9,.8],1)}],
            'detector':{'model':'test','sha256':'a'*64,'config':{}}}
    camera.latest_detection=(frame,result)
    runtime=LocalRuntime(store,camera)
    app=create_app(store,sources={'femto-bolt':{'label':'Bolt','local':True}},local_runtime=runtime)
    app.testing=True
    return store,camera,frame,result,runtime,app


def test_frame_bundle_raw_planes_calibration_and_headers(local):
    store,camera,frame,result,runtime,app=local
    client=app.test_client()
    response=client.get('/api/camera/frame?frame_id='+frame.id)
    assert response.headers['X-RGB-Frame-ID']==frame.id
    assert response.headers['Cache-Control']=='no-store'
    with np.load(io.BytesIO(response.data),allow_pickle=False) as bundle:
        assert bundle['color_rgb'].dtype==np.uint8
        assert bundle['depth_registered'].shape==(30,40)
        assert np.array_equal(bundle['ir_native'],frame.ir_native)
        metadata=json.loads(bundle['metadata_json_utf8'].tobytes())
        assert metadata['calibration_sha256']==hashlib.sha256(encode(metadata['calibration']).encode()).hexdigest()
    for channel in ['depth','ir']:
        response=client.get('/api/camera/'+channel+'/raw.png?frame_id='+frame.id)
        array=cv2.imdecode(np.frombuffer(response.data,np.uint8),cv2.IMREAD_UNCHANGED)
        assert array.dtype==np.uint16
        assert np.array_equal(array,getattr(frame,channel+'_native'))
    assert client.get('/api/camera/frame?frame_id=invalid').status_code==400
    assert client.get('/api/camera/frame?frame_id='+str(uuid4())).status_code==410
    assert client.get('/api/camera/thermal/snapshot.jpg').status_code==404


def test_local_source_reports_actual_camera_name_without_renaming_archive_identity(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    camera.backend='v4l2'
    monkeypatch.setattr('rgb.camera.v4l2_name',lambda device:'C922 Pro Stream Webcam')
    client=app.test_client()
    assert client.get('/api/sources').json[0]['label']=='C922 Pro Stream Webcam'
    assert client.get('/api/sources').json[0]['id']=='femto-bolt'
    status=client.get('/api/camera/status').json
    assert status['name']==status['device']['name']=='C922 Pro Stream Webcam'
    assert status['backend']=='v4l2'
    frame.metadata['device']={'name':'C922 Pro Stream Webcam'}
    runtime.archive(frame,result)
    camera.backend='orbbec';camera.info={'name':'Orbbec Femto Bolt','serial':'test'}
    assert client.get('/api/sources').json[0]['label']=='Orbbec Femto Bolt'
    assert client.get('/api/review/records').json['records'][0]['camera_name']=='C922 Pro Stream Webcam'


def test_stale_camera_leaves_status_health_archive_available(local):
    store,camera,frame,result,runtime,app=local
    event=runtime.archive(frame,result)
    camera.freshness_s=-1
    client=app.test_client()
    assert client.get('/api/camera/color/snapshot.jpg').status_code==503
    assert client.get('/api/camera/status').status_code==200
    assert client.get('/healthz').status_code==200
    assert client.get('/api/events/'+event['event_id']).status_code==200


def test_local_dashboard_and_manual_analysis_keep_sqlite_backend(local):
    store,camera,frame,result,runtime,app=local
    client=app.test_client()
    page=client.get('/').data
    assert b'/static/live-video.js' in page and b'/static/pointcloud.js' in page
    assert b'id="pause-capture"' in page
    assert b'id="classification-toggle"' not in page
    assert b'id="catch-up"' not in page
    event=runtime.archive(frame,result)
    endpoint='/api/events/'+event['event_id']+'/analyze'
    assert client.post(endpoint).status_code==400
    assert client.post(endpoint,headers={'X-RGB-Request':'analysis'}).status_code==202
    assert store.queued_manual()['event_id']==event['event_id']
    assert client.post(endpoint,headers={'X-RGB-Request':'analysis'}).json['created'] is False
    assert client.post('/api/settings',json={'classification_running':False,'record_observations':False},
                       headers={'X-RGB-Request':'settings'}).status_code==400
    key=detection_id(event['event_id'],0)
    assert client.post('/api/detections/'+key+'/analyze').status_code==404
    assert client.post('/api/analysis/catch-up').status_code==404
    assert client.get('/api/camera/color/snapshot.jpg?frame_id='+frame.id).status_code==200
    assert client.get('/api/live/femto-bolt/status').status_code==200


def test_frame_eviction_and_pinned_detections_pending(local):
    store,camera,frame,result,runtime,app=local
    new=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                   {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
    camera.publish(new)
    client=app.test_client()
    assert client.get('/api/camera/detections?frame_id='+new.id).status_code==202
    response=client.get('/api/camera/detections').json
    assert response['frame_id']==frame.id and response['camera_frame_id']==new.id
    camera.max_frames=1;camera.publish(new)
    with pytest.raises(FrameExpired):camera.get(frame.id)


def test_manual_capture_uses_completed_frame_and_remains_analyzable(local):
    store,camera,frame,result,runtime,app=local
    newer=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                     {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
    camera.publish(newer)
    runtime.update_capture_settings({'capture_paused':True})
    response=app.test_client().post('/api/camera/observations',json={})
    assert response.status_code==201
    assert response.json['frame_id']==frame.id
    event=store.raw(response.json['event_id'])
    assert event['detections']==result['detections']
    assert event['detector']==result['detector']
    assert event['sensor_assets']['color_preview']['sha256']==hashlib.sha256(frame.previews['color']).hexdigest()
    assert store.event_for_frame(newer.id) is None
    runtime.client=FakeBedrock()
    store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    assert len(runtime.client.calls)==1
    assert store.get(event['event_id'])['detections'][0]['enhancement']['label']=='plastic bottle'


@pytest.mark.parametrize('unavailable',['missing','stale','session','calibration','dimensions'])
def test_manual_capture_rejects_unusable_detection_without_archiving(local,unavailable):
    store,camera,frame,result,runtime,app=local
    if unavailable=='missing':camera.latest_detection=None
    elif unavailable=='stale':frame.metadata['received_monotonic_s']=time.monotonic()-camera.freshness_s-1
    metadata={**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()}
    if unavailable=='session':metadata['session_id']=str(uuid4())
    if unavailable=='calibration':metadata['calibration_sha256']='b'*64
    shape=(20,20,3) if unavailable=='dimensions' else (30,40,3)
    camera.publish(make_frame(np.zeros(shape,np.uint8),None,None,None,metadata))
    response=app.test_client().post('/api/camera/observations',json={})
    assert response.status_code==503
    assert store.storage_status()['observations']==0


def test_pinned_capture_waits_for_matching_detection_and_preserves_identity(local):
    store,camera,frame,result,runtime,app=local
    newer=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                     {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
    camera.publish(newer)
    client=app.test_client()
    assert client.post('/api/camera/observations',json={'frame_id':newer.id}).status_code==503
    with pytest.raises(CameraUnavailable):runtime.archive(newer)
    assert store.storage_status()['observations']==0
    camera.latest_detection=(newer,{**result,'frame_id':newer.id})
    saved=client.post('/api/camera/observations',json={'frame_id':newer.id})
    assert saved.status_code==201 and saved.json['frame_id']==newer.id
    camera.latest_detection=None
    repeated=client.post('/api/camera/observations',json={'frame_id':newer.id})
    assert repeated.status_code==201 and repeated.json==saved.json
    assert store.storage_status()['observations']==1


def test_manual_capture_preserves_genuinely_empty_detection_result(local):
    store,camera,frame,result,runtime,app=local
    camera.latest_detection=(frame,{**result,'detections':[]})
    response=app.test_client().post('/api/camera/observations',json={})
    assert response.status_code==201
    event=store.raw(response.json['event_id'])
    assert event['detections']==[] and event['detector']==result['detector']


def test_depth_statistics_units_invalids_and_minimum_samples():
    depth=np.array([[0,1000,2000],[3000,4000,5000]],np.uint16)
    result=depth_summary(depth,[0,0,1,1],.5,minimum=3)
    assert result['median_m']==1.5
    assert result['valid_samples']==5 and result['roi_pixels']==6
    assert result['valid_fraction']==pytest.approx(5/6)
    insufficient=depth_summary(depth,[0,0,1,1],1,minimum=30)
    assert insufficient['median_m'] is None and insufficient['status']=='insufficient'


def test_camera_lock_is_shared_across_state_directories(tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    first=CameraLock('test-physical-serial')
    try:
        with pytest.raises(CameraBusy):CameraLock('test-physical-serial')
        second=CameraLock('different-physical-camera');second.close()
    finally:first.close()
    next_owner=CameraLock('test-physical-serial');next_owner.close()


def test_archive_idempotence_conflicts_and_exact_assets(local):
    store,camera,frame,result,runtime,app=local
    event=runtime.archive(frame,result)
    assert runtime.archive(frame,result)['event_id']==event['event_id']
    assert store.storage_status()['observations']==1
    assert store.get(event['event_id'])['detections'][0]['detection_id']==detection_id(event['event_id'],0)
    for asset in [*event['sensor_assets'].values(),event['calibration_asset'],event['metadata_asset']]:
        assert hashlib.sha256((store.root/asset['file']).read_bytes()).hexdigest()==asset['sha256']
    changed={**event,'location':'different'}
    with pytest.raises(ArchiveConflict):store.save(changed,frame.previews['color'])
    assert store.raw(event['event_id'])['location'] is None


def test_settings_concurrency_restart_and_cli_defaults(local):
    store,*_=local
    with ThreadPoolExecutor(max_workers=4) as pool:
        versions=list(pool.map(lambda n:store.update_settings({'vlm_threshold':.5+n*.01})['version'],range(12)))
    assert sorted(versions)==list(range(1,13))
    store.update_settings({'selected_model':FALLBACK_MODEL,'model_selection_reason':'latency_fallback','vlm_paused':True})
    restarted=LocalStore(store.directory)
    assert restarted.settings()['selected_model']==FALLBACK_MODEL
    assert restarted.settings()['vlm_paused'] is True
    from rgb.local import parser
    assert parser().parse_args([]).model is None
    assert parser().parse_args([]).paused is None


def test_jobs_queue_limit_dedup_policy_pin_and_recovery(local):
    store,camera,frame,result,runtime,app=local
    event=runtime.archive(frame,result)
    assert store.request_analysis(event['event_id'])
    assert not store.request_analysis(event['event_id'])
    job=store.queued_manual()
    store.update_settings({'selected_model':FALLBACK_MODEL})
    assert job['policy']['selected_model']==PRIMARY_MODEL
    # Never-started requests survive restart.
    restarted=LocalStore(store.directory)
    assert restarted.queued_manual()['id']==job['id']
    store.set_job(job,'running')
    attempt={'attempt_id':str(uuid4()),'job_id':job['id'],'outcome':'started','box_indices':[0],'started_at':now()}
    store.record_attempt(event,[0],attempt)
    restarted=LocalStore(store.directory)
    assert restarted.queued_manual() is None
    assert restarted.analysis(event['event_id'])['status']=='failed'
    attempts=[json.loads(fields['value']) for _,fields in restarted.stream('rgb:model-attempts')]
    assert [a['outcome'] for a in attempts]==['started','interrupted']
    assert restarted.active_attempt() is None
    assert restarted.request_analysis(event['event_id'])


class FakeBedrock:
    def __init__(self,text=None,fail=False):
        self.calls=[]
        self.text=text or json.dumps({'observations':[reading()]})
        self.fail=fail

    def converse(self,**body):
        self.calls.append(body)
        if self.fail:raise TimeoutError('injected network timeout')
        return {'output':{'message':{'content':[{'text':self.text}]}},'stopReason':'end_turn',
                'usage':{'inputTokens':100,'outputTokens':100},'metrics':{'latencyMs':5},
                'ResponseMetadata':{'HTTPStatusCode':200,'RequestId':'test-request'}}


def start_job(local,client):
    store,camera,frame,result,runtime,app=local
    runtime.client=client
    event=runtime.archive(frame,result)
    assert store.request_analysis(event['event_id'])
    return event,store.job(event['event_id'])


def test_bedrock_uses_images_and_commits_validated_batch_and_audit(local):
    store,camera,frame,result,runtime,app=local
    client=FakeBedrock()
    event,job=start_job(local,client)
    assert runtime._batch(job)
    body=client.calls[0]
    assert body['modelId']==PRIMARY_MODEL
    assert body['additionalModelRequestFields']['reasoningConfig']['type']=='disabled'
    images=[block['image'] for block in body['messages'][0]['content'] if 'image' in block]
    assert len(images)==4
    assert all(image['source']['bytes'].startswith(b'\xff\xd8') for image in images)
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    assert [a['outcome'] for a in attempts]==['started','success']
    assert attempts[-1]['request_id']=='test-request'
    for image,asset in zip(images,attempts[0]['inputs']):
        assert image['source']['bytes']==(store.root/asset['asset']['file']).read_bytes()
    enhanced=store.get(event['event_id'])['detections'][0]['enhancement']
    assert enhanced['label']=='plastic bottle'
    assert enhanced['attempt_id']==attempts[-1]['attempt_id']
    assert store.analysis(event['event_id'])['status']=='complete'


@pytest.mark.parametrize('text',[
    '{"observations":[]}',json.dumps({'observations':[{**reading(),'extra':True}]}),
    json.dumps({'observations':[{**reading(),'confidence':True}]}),
    json.dumps({'observations':[{**reading(),'confidence':float('nan')}]}),
    '{"observations":[],"extra":true}','{"observations":[],"observations":[]}',
    json.dumps({'observations':[{**reading(),'box_index':True}]}),
    json.dumps({'observations':[{**reading(),'reason':'  '}]}),
])
def test_invalid_bedrock_response_never_becomes_a_label(local,text):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock(text=text))
    assert runtime._batch(job)
    assert store.get(event['event_id'])['detections'][0]['enhancement'] is None
    assert store.analysis(event['event_id'])['status']=='failed'
    assert runtime.retry_at>time.monotonic()
    assert store.settings()['selected_model']==PRIMARY_MODEL


def test_outer_json_fence_is_allowed():
    assert validate_readings({'message':{'content':'```json\n'+json.dumps({'observations':[reading()]})+'\n```'}},[{'index':0}])==[reading()]


def test_pause_and_manual_fallback_pin(local):
    store,camera,frame,result,runtime,app=local
    client=FakeBedrock()
    event,job=start_job(local,client)
    store.update_settings({'vlm_paused':True})
    assert runtime._batch(job) is False and not client.calls
    store.update_settings({'vlm_paused':False,'selected_model':FALLBACK_MODEL})
    runtime._batch(job)
    assert client.calls[0]['modelId']==PRIMARY_MODEL
    assert store.settings()['selected_model']==FALLBACK_MODEL
    assert store.request_analysis(event['event_id'])
    runtime._batch(store.job(event['event_id']))
    assert client.calls[-1]['modelId']==FALLBACK_MODEL
    assert 'additionalModelRequestFields' not in client.calls[-1]


def test_sqlite_terminal_and_enhancement_roll_back_together(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    original=store._update
    def fail(db,key,kind,value):
        if kind=='enhanced':raise OSError('injected disk failure')
        original(db,key,kind,value)
    monkeypatch.setattr(store,'_update',fail)
    runtime._batch(job)
    outcomes=[json.loads(fields['value'])['outcome'] for _,fields in store.stream('rgb:model-attempts')]
    assert outcomes==['started','failed']
    assert store.get(event['event_id'])['detections'][0]['enhancement'] is None


def test_stale_evaluation_and_local_review_export_snapshot(local):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    old=store.get(event['event_id'])['detections'][0]
    runtime._batch(job)
    values={'reference_label':'bottle','detector_result':'correct','vlm_result':'correct','notes':''}
    with pytest.raises(StaleEvaluation):store.evaluate(old['detection_id'],values,old['review_context'])
    d=store.get(event['event_id'])['detections'][0]
    store.evaluate(d['detection_id'],values,d['review_context'])
    with store.export_snapshot(['rgb:events']) as (snapshot,stops,settings):
        store.update_settings({'display_mode':'materials'})
        assert snapshot.settings()['display_mode']=='objects'
        assert settings['display_mode']=='objects'
    directory,path=build_export(store,'parquet')
    try:
        with zipfile.ZipFile(path) as archive:
            detections=pq.read_table(pa.BufferReader(archive.read('detections.parquet'))).to_pylist()
            assert detections[0]['frame_id']==frame.id
            assert detections[0]['depth']['median_m']==1.234
            assert detections[0]['display_mode']=='materials'
            attempts=pq.read_table(pa.BufferReader(archive.read('model_attempts.parquet'))).to_pylist()
            assert attempts[-1]['request_id']=='test-request'
    finally:directory.cleanup()


def test_evidence_export_and_unauthenticated_camera_write(local):
    store,camera,frame,result,runtime,app=local
    client=app.test_client()
    response=client.post('/api/camera/observations',json={'frame_id':frame.id})
    assert response.status_code==201
    event_id=response.json['event_id']
    assert client.post('/api/camera/observations',json={'frame_id':frame.id}).json['event_id']==event_id
    response=client.get('/api/review/evidence?event_ids='+event_id)
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['observations'][0]['frame_id']==frame.id
        for name,asset in manifest['assets'].items():
            assert hashlib.sha256(archive.read('assets/'+name)).hexdigest()==asset['sha256']
    response.close()
    assert client.post('/api/ingest',json={}).status_code==404


def test_all_eight_stream_slots_released_without_generator_iteration(local):
    *_,app=local
    responses=[]
    with app.test_request_context('/api/camera/color/stream'):
        for _ in range(8):responses.append(app.view_functions['camera_stream']('color'))
        assert app.view_functions['camera_stream']('color')[1]==429
        for response in responses:response.close()
        available=app.view_functions['camera_stream']('color')
        assert available.status_code==200
        available.close()


def test_model_selection_validates_reason_and_audits_revision(local):
    *_,app=local
    client=app.test_client()
    headers={'X-RGB-Request':'settings'}
    bad=client.post('/api/settings',json={'selected_model':FALLBACK_MODEL,'model_selection_reason':'restore_primary'},headers=headers)
    assert bad.status_code==400
    good=client.post('/api/settings',json={'selected_model':FALLBACK_MODEL,'model_selection_reason':'latency_fallback'},headers=headers)
    assert good.status_code==200
    assert good.json['version']==1
    assert client.get('/api/vlm/status').json['selected_model']==FALLBACK_MODEL


@pytest.fixture
def openrouter_local(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    monkeypatch.setenv('OPENROUTER_API_KEY','private-test-key')
    class Response:
        status_code=200
        headers={}
        def __init__(self,payload):self.payload=payload
        def json(self):return self.payload
        def close(self):pass
    class Session:
        def __init__(self):
            self.calls=[]
            self.text=json.dumps({'observations':[reading()]})
            self.finish_reason='stop'
        def post(self,url,**kwargs):
            self.calls.append(kwargs)
            return Response({'id':'gen-local','provider':'Amazon Bedrock','model':kwargs['json']['model'],
                'choices':[{'finish_reason':self.finish_reason,'message':{'content':self.text}}],
                'usage':{'prompt_tokens':120,'completion_tokens':90,'cost':.0003,
                         'completion_tokens_details':{'reasoning_tokens':0}}})
    session=Session()
    runtime.providers['openrouter']=OpenRouterProvider(session=session)
    runtime.client=FakeBedrock()
    response=app.test_client().post('/api/settings',json={'selected_provider':'openrouter',
        'selected_model':'amazon/nova-2-lite-v1','model_selection_reason':'restore_primary'},
        headers={'X-RGB-Request':'settings'})
    assert response.status_code==200
    return local,session


def test_openrouter_shared_validation_audit_projection_status_and_export(openrouter_local):
    (store,camera,frame,result,runtime,app),session=openrouter_local
    event=runtime.archive(frame,result)
    store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    projected=store.get(event['event_id'])['detections'][0]['enhancement']
    assert projected['label']=='plastic bottle' and projected['endpoint']=='openrouter'
    assert projected['model']=='amazon/nova-2-lite-v1'
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    assert [a['outcome'] for a in attempts]==['started','success']
    assert {a['endpoint'] for a in attempts}=={'openrouter'}
    terminal=attempts[-1]
    assert terminal['resolved_model']=='amazon/nova-2-lite-v1' and terminal['upstream_provider']=='Amazon Bedrock'
    assert terminal['request_id']=='gen-local' and terminal['prompt_tokens']==120
    assert terminal['ordered_image_sha256']==attempts[0]['ordered_image_sha256']
    assert len(terminal['inputs'])==4
    assert not runtime.client.calls and len(session.calls)==1
    assert 'private-test-key' not in encode(attempts)
    status=app.test_client().get('/api/vlm/status').json
    assert status['selected_provider']=='openrouter'
    assert status['latencies'][0]['provider']=='openrouter'
    assert status['model_outcomes']['amazon/nova-2-lite-v1']['success']==1
    assert status['providers'][1]['models'][0]['id']=='amazon/nova-2-lite-v1'
    assert LocalStore(store.directory).settings()['selected_provider']=='openrouter'
    with app.test_client().get('/api/review/export/parquet') as response:
        assert response.status_code==200
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert json.loads(archive.read('manifest.json'))['schema_version']==9
            rows=pq.read_table(pa.BufferReader(archive.read('model_attempts.parquet'))).to_pylist()
            assert rows[-1]['endpoint']=='openrouter' and rows[-1]['cost']==.0003
            assert rows[-1]['reasoning_tokens']==0
            settings=pq.read_table(pa.BufferReader(archive.read('settings.parquet'))).to_pylist()
            assert settings[-1]['selected_provider']=='openrouter'


def test_queued_openrouter_job_keeps_provider_when_selection_changes(openrouter_local):
    (store,camera,frame,result,runtime,app),session=openrouter_local
    event=runtime.archive(frame,result)
    store.request_analysis(event['event_id'])
    job=store.job(event['event_id'])
    store.update_settings({'selected_provider':'bedrock','selected_model':PRIMARY_MODEL})
    assert runtime._batch(job)
    assert len(session.calls)==1 and not runtime.client.calls
    assert job['policy']['selected_provider']=='openrouter'
    store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    assert len(runtime.client.calls)==1


@pytest.mark.parametrize('failure',['schema','truncated','missing_key'])
def test_failed_openrouter_never_publishes_or_switches_provider(openrouter_local,failure,monkeypatch):
    (store,camera,frame,result,runtime,app),session=openrouter_local
    if failure=='schema':session.text='{"observations":[]}'
    elif failure=='truncated':session.finish_reason='length'
    else:monkeypatch.delenv('OPENROUTER_API_KEY')
    event=runtime.archive(frame,result);store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    assert store.analysis(event['event_id'])['status']=='failed'
    assert store.get(event['event_id'])['detections'][0]['enhancement'] is None
    assert not runtime.client.calls
    assert store.settings()['selected_provider']=='openrouter'
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    assert [a['outcome'] for a in attempts]==['started','failed']
    if failure!='missing_key':assert attempts[-1]['request_id']=='gen-local'


@pytest.mark.parametrize('provider,model',[
    ('openrouter',PRIMARY_MODEL),('bedrock','amazon/nova-2-lite-v1'),
    ('unknown','amazon/nova-2-lite-v1'),([],PRIMARY_MODEL),('bedrock',[]),
])
def test_provider_model_pair_validation_is_atomic(local,provider,model):
    store,*_,app=local
    response=app.test_client().post('/api/settings',json={'selected_provider':provider,'selected_model':model,
        'model_selection_reason':'restore_primary'},headers={'X-RGB-Request':'settings'})
    assert response.status_code==400
    assert store.settings()['selected_provider']=='bedrock' and store.settings()['version']==0


def test_legacy_queued_job_without_provider_still_uses_bedrock(local):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    del job['policy']['selected_provider']
    assert runtime._batch(job)
    assert len(runtime.client.calls)==1
    assert store.get(event['event_id'])['detections'][0]['enhancement']['endpoint']=='bedrock'


def test_provider_cli_preserves_saved_choices_and_maps_explicit_provider_defaults(local,monkeypatch):
    from rgb.local import parser, provider_settings
    store,*_=local
    monkeypatch.delenv('RGB_VLM_PROVIDER',raising=False)
    monkeypatch.delenv('RGB_MODEL',raising=False)
    default=provider_settings(parser().parse_args(['--provider','openrouter']),store.settings())
    assert default['selected_model']=='amazon/nova-2-lite-v1'
    store.update_settings({**default,'selected_model':'google/gemini-2.5-flash-lite'})
    assert provider_settings(parser().parse_args([]),store.settings())=={}
    assert provider_settings(parser().parse_args(['--provider','bedrock']),store.settings())['selected_model']==PRIMARY_MODEL
    monkeypatch.setenv('RGB_VLM_PROVIDER','openrouter')
    monkeypatch.setenv('RGB_MODEL','amazon/nova-lite-v1')
    assert provider_settings(parser().parse_args([]),store.settings())['selected_model']=='amazon/nova-lite-v1'
    with pytest.raises(ValueError):provider_settings(parser().parse_args(['--provider','bedrock']),store.settings())
    monkeypatch.setenv('RGB_VLM_PROVIDER','unknown')
    with pytest.raises(ValueError):provider_settings(parser().parse_args([]),store.settings())


def test_preparation_error_marks_manual_job_failed_instead_of_stranding_it(local):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    (store.root/(event['image_sha256']+'.jpg')).unlink()
    assert runtime._safe_batch(job)
    assert store.analysis(event['event_id'])['status']=='failed'
    assert store.queued_manual() is None


def test_missing_channels_and_excessive_skew_are_rejected():
    camera=Camera(skew_ms=10)
    assert not camera.accept_timestamps({'color':1000,'depth':1000})
    assert not camera.accept_timestamps({'color':1000,'depth':20000,'ir':1000})
    assert camera.accept_timestamps({'color':1000,'depth':11000,'ir':1000})
    assert camera.counters['incomplete']==1 and camera.counters['skew_rejected']==1


def test_registered_preview_uses_color_coordinates_and_keeps_raw_planes(local):
    store,camera,frame,result,runtime,app=local
    preview=cv2.imdecode(np.frombuffer(frame.previews['depth'],np.uint8),cv2.IMREAD_UNCHANGED)
    assert preview.shape[:2]==frame.color_rgb.shape[:2]
    assert frame.metadata['depth_preview']['plane']=='depth_registered'
    assert np.all(frame.depth_native==1234)
    with pytest.raises(ValueError):frame.depth_native[0,0]=0


def test_commit_failure_rolls_back_success_and_does_not_advance_job_progress(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    original=store.transaction
    injected=False
    @contextmanager
    def transaction():
        nonlocal injected
        with original() as db:
            yield db
            row=db.execute("SELECT fields FROM journal WHERE topic='rgb:model-attempts' ORDER BY seq DESC LIMIT 1").fetchone()
            if row and not injected and json.loads(json.loads(row[0])['value'])['outcome']=='success':
                injected=True
                raise OSError('injected failure before commit')
    monkeypatch.setattr(store,'transaction',transaction)
    runtime._batch(job)
    assert injected and job['progress']==0
    assert store.analysis(event['event_id'])['status']=='failed'
    assert store.get(event['event_id'])['detections'][0]['enhancement'] is None
    assert [json.loads(fields['value'])['outcome'] for _,fields in store.stream('rgb:model-attempts')]==['started','failed']


def test_unsaved_request_images_withhold_cloud_dispatch(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    client=FakeBedrock()
    event,job=start_job(local,client)
    def full(*args):raise OSError('disk full')
    monkeypatch.setattr(store,'asset',full)
    runtime._safe_batch(job)
    assert not client.calls
    assert store.analysis(event['event_id'])['status']=='failed'
    assert store.ping()


def test_storage_failure_remains_visible_even_if_failure_audit_cannot_be_written(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    client=FakeBedrock()
    event,job=start_job(local,client)
    def full(*args,**kwargs):raise OSError('disk full')
    monkeypatch.setattr(store,'set_job',full)
    monkeypatch.setattr(store,'processing_failure',full)
    assert runtime._safe_batch(job)
    assert not client.calls
    assert 'Archive storage unavailable' in runtime.last_error
    assert runtime.retry_at>time.monotonic()
    assert app.test_client().get('/api/camera/color/snapshot.jpg').status_code==200


def test_empty_manual_observation_never_invokes_bedrock(local):
    store,camera,frame,result,runtime,app=local
    runtime.client=FakeBedrock()
    event=runtime.archive(frame,{**result,'detections':[]})
    store.request_analysis(event['event_id'])
    assert runtime._batch(store.job(event['event_id']))
    assert not runtime.client.calls
    assert store.analysis(event['event_id'])['status']=='complete'


def test_explicit_recovery_has_a_new_job_and_links_the_interrupted_attempt(local):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock(fail=True))
    runtime._batch(job)
    previous=store.active_attempt()
    assert previous is None
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    failed=attempts[-1]
    store.request_analysis(event['event_id'])
    resumed=store.job(event['event_id'])
    assert resumed['id']!=job['id']
    runtime.client=FakeBedrock()
    runtime._batch(resumed)
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    assert attempts[-1]['attempt_id']!=failed['attempt_id']
    assert attempts[-1]['recovery_of']=={'job_id':job['id'],'attempt_id':failed['attempt_id']}


def test_live_readings_ignore_manual_old_policy_lost_track_and_reconnect(local):
    store,camera,frame,result,runtime,app=local
    event=runtime.archive(frame,result)
    attempt={'model':PRIMARY_MODEL,'attempt_id':'test'}
    job={'kind':'manual','policy':store.settings()}
    runtime._publish_readings(event,[reading()],attempt,job)
    assert not runtime.live_readings
    job['kind']='live'
    runtime._publish_readings(event,[reading()],attempt,job)
    assert len(runtime.live_readings)==1
    d={**result['detections'][0]}
    runtime._live_value(frame,d,store.settings())
    assert d['live_value']['display_label']=='plastic bottle'
    runtime.live_readings.clear()
    store.update_settings({'selected_model':FALLBACK_MODEL})
    runtime._publish_readings(event,[reading()],attempt,job)
    assert not runtime.live_readings
    job['policy']=store.settings()
    camera.session_id=str(uuid4())
    runtime._publish_readings(event,[reading()],attempt,job)
    assert not runtime.live_readings
    camera.session_id=event['session_id']
    camera.latest_detection=(frame,{**result,'detections':[]})
    runtime._publish_readings(event,[reading()],attempt,job)
    assert not runtime.live_readings


def test_manual_queue_capacity_and_unsupported_v4l2_channels(local):
    store,camera,frame,result,runtime,app=local
    for n in range(101):
        event={'event_id':str(uuid4()),'frame_id':str(uuid4()),'source_id':'femto-bolt','observed_at':now(),
               'received_at':now(),'location':None,'detector':result['detector'],'detections':[]}
        store.save(event,frame.previews['color'])
        if n<100:assert store.request_analysis(event['event_id'])
        else:
            with pytest.raises(QueueFull):store.request_analysis(event['event_id'])
    camera.backend='v4l2'
    client=app.test_client()
    assert client.get('/api/camera/depth/raw.png').status_code==404
    assert client.get('/api/camera/ir/snapshot.jpg').status_code==404
    assert client.get('/api/camera/ir/stream').status_code==404


def test_unknown_caller_headers_are_not_trusted_as_local_identity(local):
    store,camera,frame,result,runtime,app=local
    response=app.test_client().post('/api/settings',json={'display_mode':'materials'},
                                   headers={'X-RGB-Request':'settings','X-Laminarauth-Identity':'invented-user'})
    assert response.status_code==200 and response.json['actor'] is None


def test_publication_failure_cannot_rewrite_a_committed_success(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    def fail(*args):raise OSError('injected live publication failure')
    monkeypatch.setattr(runtime,'_publish_readings',fail)
    assert runtime._batch(job)
    assert store.analysis(event['event_id'])['status']=='complete'
    assert store.get(event['event_id'])['detections'][0]['enhancement']['label']=='plastic bottle'
    assert [json.loads(fields['value'])['outcome'] for _,fields in store.stream('rgb:model-attempts')]==['started','success']


def test_backend_alias_reservation_uses_same_physical_lock(tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    first=Camera(serial='fixture-serial')
    first.reserve()
    monkeypatch.setattr('rgb.camera.v4l2_serial',lambda device:'fixture-serial')
    try:
        other=Camera(backend='v4l2',device='/dev/by-alias')
        with pytest.raises(CameraBusy):other.reserve()
    finally:first.lock.close()


def test_unavailable_v4l2_node_needs_explicit_identity_before_opening(tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_RUNTIME_DIR',str(tmp_path))
    def absent(device):raise CameraUnavailable('node missing')
    monkeypatch.setattr('rgb.camera.v4l2_serial',absent)
    with pytest.raises(CameraUnavailable):Camera(backend='v4l2').reserve()
    explicit=Camera(backend='v4l2',serial='fixture-serial')
    explicit.reserve()
    assert explicit.lock is not None
    explicit.lock.close()


def test_local_continuous_tracks_do_not_reuse_identity_after_an_empty_detection():
    from board.tracking import Tracker
    tracker=Tracker(continuous=True)
    original=tracker.update([{'class_id':1,'box':[.1,.1,.9,.9]}],0)[0]['track_id']
    tracker.update([],1)
    replacement=tracker.update([{'class_id':1,'box':[.1,.1,.9,.9]}],2)[0]['track_id']
    assert original!=replacement


@pytest.mark.parametrize('stop_reason',['max_tokens','content_filtered','guardrail_intervened'])
def test_truncated_or_refused_bedrock_output_never_updates_labels(local,stop_reason):
    store,camera,frame,result,runtime,app=local
    class Stopped(FakeBedrock):
        def converse(self,**body):
            response=super().converse(**body)
            response['stopReason']=stop_reason
            return response
    event,job=start_job(local,Stopped())
    runtime._batch(job)
    assert store.get(event['event_id'])['detections'][0]['enhancement'] is None
    attempts=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    assert attempts[-1]['outcome']=='failed' and attempts[-1]['finish_reason']==stop_reason


def test_manual_backlog_cannot_starve_fresh_live_work(local):
    store,camera,frame,result,runtime,app=local
    order=[]
    class Logged(FakeBedrock):
        def converse(self,**body):
            active=store.active_attempt()
            order.append(store.connection().execute('SELECT kind FROM jobs WHERE id=?',(active['job_id'],)).fetchone()[0])
            return super().converse(**body)
    runtime.client=Logged()
    for _ in range(3):
        f=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                     {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
        event=runtime.archive(f,result)
        store.request_analysis(event['event_id'])
    with runtime.condition:runtime.pending=(frame,result)
    thread=threading.Thread(target=runtime._vlm,daemon=True);thread.start()
    deadline=time.monotonic()+3
    while len(order)<4 and time.monotonic()<deadline:time.sleep(.01)
    runtime.stop_event.set()
    with runtime.condition:runtime.condition.notify_all()
    thread.join(1)
    assert order==['manual','live','manual','manual']


def test_live_camera_keeps_advancing_when_the_detector_result_is_stale(local):
    store,camera,frame,result,runtime,app=local
    frame.metadata['received_monotonic_s']=time.monotonic()-10
    newest=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                      {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
    camera.publish(newest)
    with app.test_request_context('/api/live/femto-bolt/stream'):
        response=app.view_functions['stream']('femto-bolt')
        part=next(response.response)
        assert newest.id.encode() in part and frame.id.encode() not in part
        response.close()


def test_automatic_capture_and_fast_vlm_share_one_record_interval(local):
    store,camera,frame,result,runtime,app=local
    runtime.client=FakeBedrock()
    assert store.settings()['capture_interval_s']==60
    first=runtime._capture_tick()
    assert first['frame_id']==frame.id
    thread=threading.Thread(target=runtime._vlm,daemon=True);thread.start()
    deadline=time.monotonic()+3
    while not runtime.client.calls and time.monotonic()<deadline:time.sleep(.01)
    try:
        for _ in range(20):
            f=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                         {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
            camera.publish(f);camera.latest_detection=(f,{**result,'frame_id':f.id})
            assert runtime._capture_tick() is None
        assert store.storage_status()['observations']==1
        assert len(runtime.client.calls)==1
        runtime.last_auto_capture=time.monotonic()-61
        assert runtime._capture_tick()['frame_id']==f.id
        assert store.storage_status()['observations']==2
    finally:
        runtime.stop_event.set()
        with runtime.condition:runtime.condition.notify_all()
        thread.join(2)


def test_capture_pause_keeps_camera_readable_and_allows_intentional_snapshot(local):
    store,camera,frame,result,runtime,app=local
    first=runtime._capture_tick()
    client=app.test_client()
    saved=client.post('/api/settings',json={'capture_paused':True},
                      headers={'X-RGB-Request':'settings'})
    assert saved.status_code==200 and saved.json['capture_paused']
    assert runtime.pending is None
    f=make_frame(np.zeros((30,40,3),np.uint8),None,None,None,
                 {**frame.metadata,'frame_id':str(uuid4()),'received_monotonic_s':time.monotonic()})
    camera.publish(f);camera.latest_detection=(f,{**result,'frame_id':f.id})
    assert runtime._capture_tick() is None
    assert store.storage_status()['observations']==1
    assert client.get('/api/camera/color/snapshot.jpg').status_code==200
    assert client.get('/api/vlm/status').json['capture']['paused']
    assert client.post('/api/camera/observations',json={}).status_code==201
    assert store.storage_status()['observations']==2
    assert LocalStore(store.directory).settings()['capture_paused']
    client.post('/api/settings',json={'capture_paused':False},headers={'X-RGB-Request':'settings'})
    assert runtime._capture_tick()['frame_id']==f.id


@pytest.mark.parametrize('data',[
    {'capture_interval_s':0},{'capture_interval_s':86401},{'capture_interval_s':True},
    {'capture_interval_s':'60'},{'capture_interval_s':float('nan')},{'capture_paused':'true'},
])
def test_invalid_capture_preferences_are_rejected(local,data):
    *_,app=local
    assert app.test_client().post('/api/settings',json=data,headers={'X-RGB-Request':'settings'}).status_code==400


def test_capture_interval_saved_and_implicit_cli_does_not_reset_it(local):
    store,camera,frame,result,runtime,app=local
    client=app.test_client()
    response=client.post('/api/settings',json={'capture_interval_s':300},headers={'X-RGB-Request':'settings'})
    assert response.status_code==200
    assert LocalStore(store.directory).settings()['capture_interval_s']==300
    assert client.get('/api/camera/status').json['configuration']['archive_interval_s']==300
    from rgb.local import parser
    assert parser().parse_args([]).archive_interval is None


def test_capture_pause_blocks_live_but_allows_review_of_existing_observation(local):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock())
    runtime.update_capture_settings({'capture_paused':True})
    job['kind']='live'
    assert runtime._batch(job) is False and not runtime.client.calls
    job['kind']='manual'
    assert runtime._batch(job)
    assert len(runtime.client.calls)==1


def test_capture_pause_acknowledgement_serializes_with_live_dispatch_admission(local,monkeypatch):
    store,camera,frame,result,runtime,app=local
    event,job=start_job(local,FakeBedrock());job['kind']='live'
    entered,release=threading.Event(),threading.Event()
    original=store.record_attempt
    def admission(event,indices,attempt):
        if attempt['outcome']=='started':
            entered.set()
            assert release.wait(5)
        original(event,indices,attempt)
    monkeypatch.setattr(store,'record_attempt',admission)
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch=pool.submit(runtime._batch,job)
        assert entered.wait(5)
        pause=pool.submit(runtime.update_capture_settings,{'capture_paused':True})
        try:
            with pytest.raises(TimeoutError):pause.result(timeout=.05)
        finally:release.set()
        assert pause.result(timeout=5)['capture_paused']
        assert batch.result(timeout=5)
    assert len(runtime.client.calls)==1  # Already admitted; its terminal audit is retained.


def cloud_calibration(frame):
    calibration={'schema_version':1,'planes':{'color':{
        'intrinsics':{'width':40,'height':30,'fx':100.,'fy':100.,'cx':20.,'cy':15.},
        'distortion':{'model':'BROWN_CONRADY_K6','k1':0,'k2':0,'k3':0,'p1':0,'p2':0}}}}
    frame.metadata['calibration']=calibration
    frame.metadata['calibration_sha256']=hashlib.sha256(encode(calibration).encode()).hexdigest()
    return calibration


def decode_cloud(payload):
    import struct
    from rgb.pointcloud import POINT_DTYPE
    length=struct.unpack('<I',payload[:4])[0]
    header=json.loads(payload[4:4+length])
    return header,np.frombuffer(payload[4+length:],POINT_DTYPE)


def test_pointcloud_uses_calibrated_registered_depth_and_exact_rgb_without_archiving(local):
    store,camera,frame,result,runtime,app=local
    cloud_calibration(frame)
    frame.metadata['registered_depth_scale_mm']=2
    response=app.test_client().get('/api/camera/pointcloud?stride=5&frame_id='+frame.id)
    assert response.status_code==200
    assert response.headers['X-RGB-Depth-Units']=='metres'
    assert response.headers['X-RGB-Frame-ID']==frame.id
    header,points=decode_cloud(response.data)
    assert header['point_count']==48 and header['units']=='metres'
    assert header['coordinate_frame']=='color_camera'
    assert header['calibration_sha256']==frame.metadata['calibration_sha256']
    center=points[3*8+4]
    assert np.allclose(center['xyz'],[0,0,2.468])
    assert np.array_equal(center['rgb'],frame.color_rgb[15,20])
    assert np.all(points['padding']==0)
    assert np.allclose(points[0]['xyz'],[-.4936,-.3702,2.468])
    assert store.storage_status()['observations']==0


def test_pointcloud_undistorts_known_brown_conrady_pixels(local):
    store,camera,frame,result,runtime,app=local
    calibration=cloud_calibration(frame)
    calibration['planes']['color']['distortion']['k1']=.1
    frame.metadata['calibration_sha256']=hashlib.sha256(encode(calibration).encode()).hexdigest()
    response=app.test_client().get('/api/camera/pointcloud?stride=5')
    header,points=decode_cloud(response.data)
    x,y,z=points[0]['xyz']
    xn,yn=x/z,y/z
    radial=1+.1*(xn*xn+yn*yn)
    assert 100*xn*radial+20==pytest.approx(0,abs=.0001)
    assert 100*yn*radial+15==pytest.approx(0,abs=.0001)


def test_pointcloud_filters_zero_and_out_of_range_and_bounds_cache(local):
    from rgb.pointcloud import PointCloudBuilder
    store,camera,frame,result,runtime,app=local
    cloud_calibration(frame)
    depth=frame.depth_registered.copy();depth[0,0]=0;depth[0,5]=6000
    f=make_frame(np.full((30,40,3),80,np.uint8),frame.depth_native,frame.ir_native,depth,
                 {**frame.metadata,'frame_id':str(uuid4())})
    builder=PointCloudBuilder()
    payload=builder.build(f,5,5)
    header,points=decode_cloud(payload)
    assert header['point_count']==46 and np.isfinite(points['xyz']).all()
    assert builder.build(f,5,5) is payload
    builder.build(f,6,5);builder.build(f,7,5)
    assert len(builder.cache)==2
    assert len(payload)<=30000*16+65540


def test_pointcloud_http_errors_and_unsupported_backend(local):
    store,camera,frame,result,runtime,app=local
    client=app.test_client()
    assert client.get('/api/camera/pointcloud').status_code==503  # No invented calibration.
    cloud_calibration(frame)
    for suffix in ['stride=1','stride=no','max_range_m=nan','max_range_m=21','frame_id=bad']:
        assert client.get('/api/camera/pointcloud?'+suffix).status_code==400
    assert client.get('/api/camera/pointcloud?frame_id='+str(uuid4())).status_code==410
    camera.freshness_s=-1
    assert client.get('/api/camera/pointcloud').status_code==503
    assert client.get('/api/camera/pointcloud?frame_id='+frame.id).status_code==200
    camera.backend='v4l2'
    assert client.get('/api/camera/pointcloud').status_code==404
