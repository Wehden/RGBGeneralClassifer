import hashlib
import time
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import pytest

from rgb.camera import make_frame
from rgb.store import now
from test_local import local


def classifier():
    result=Mock()
    result.classify.side_effect=lambda crop:{
        'status':'identified','label':'plastic','best_candidate':'plastic',
        'score':.94,'similarity':.12,'margin':.8,'score_type':'supervised_material_head_score',
        'model':'google/siglip2-base-patch16-384','manifest_sha256':'b'*64,'inference_ms':12,
        'candidates':[{'label':'plastic','score':.94,'similarity':.12}],
        'policy':{'minimum_probability':.775,'minimum_margin':.05}}
    return result


def new_frame(frame,index):
    return make_frame(np.full_like(frame.color_rgb,40+index),frame.depth_native,frame.ir_native,
        frame.depth_registered,{**frame.metadata,'frame_id':str(uuid4()),'sequence':index,
        'received_at':now(),'received_monotonic_s':time.monotonic()})


def test_comparison_pairs_original_box_scores_even_after_live_frame_changes(local):
    store,camera,frame,detected,runtime,app=local
    model=classifier()
    runtime.client=Mock()
    detector={'model':'yoloe-26s','sha256':'a'*64,'threshold':.45,'settings_version':1}
    runtime._material_pass(model,frame,detected['detections'],detector)
    newer=new_frame(frame,2)
    camera.publish(newer)
    live={**detected['detections'][0],'label':'different object','confidence':.99}
    runtime._attach_material(newer,live)
    assert live['local_material']['paired_object']['label']=='bottle'
    response=app.test_client().get('/api/camera/comparison')
    snapshot=response.json
    row,=snapshot['rows']
    assert snapshot['pairing']=='exact_frame_and_box' and snapshot['same_task'] is False
    assert snapshot['frame_id']==frame.id
    assert row['object']['label']=='bottle' and row['object']['confidence']==.8
    assert row['material']['score']==.94
    assert row['object']['source_frame_id']==row['material']['source_frame_id']==frame.id
    assert row['object']['box']==row['material']['source_box']==row['box']
    image=app.test_client().get(snapshot['image_url'])
    assert image.status_code==200 and image.headers['X-RGB-Frame-ID']==frame.id
    assert hashlib.sha256(image.data).hexdigest()==snapshot['image_sha256']
    assert response.headers['Cache-Control']==image.headers['Cache-Control']=='no-store'
    assert store.storage_status()['observations']==0
    runtime.client.converse.assert_not_called()


def test_comparison_images_are_bounded_and_never_substitute_a_new_frame(local):
    store,camera,frame,detected,runtime,app=local
    frames=[frame,new_frame(frame,2),new_frame(frame,3)]
    for item in frames:
        camera.publish(item)
        detections=[{**d,'track_id':str(uuid4())} for d in detected['detections']]
        runtime._material_pass(classifier(),item,detections)
    assert len(runtime.material_comparison_images)==2
    client=app.test_client()
    assert client.get('/api/camera/comparison/image.jpg?frame_id='+frame.id).status_code==410
    response=client.get('/api/camera/comparison/image.jpg?frame_id='+frames[1].id)
    assert response.status_code==200 and response.headers['X-RGB-Frame-ID']==frames[1].id
    assert client.get('/api/camera/comparison/image.jpg?frame_id=bad').status_code==400
    assert client.get('/api/camera/comparison/image.jpg').status_code==400


def test_stale_comparison_remains_an_explicitly_dated_snapshot(local):
    store,camera,frame,detected,runtime,app=local
    runtime._material_pass(classifier(),frame,detected['detections'])
    runtime.material_comparison['received_monotonic_s']=time.monotonic()-8
    snapshot=app.test_client().get('/api/camera/comparison').json
    assert snapshot['status']=='stale' and snapshot['age_s']>=8
    assert snapshot['frame_id']==frame.id and snapshot['rows'][0]['material']['score']==.94


@pytest.mark.parametrize('reset',['session','calibration'])
def test_reset_hides_previously_published_snapshots_and_images(local,reset):
    store,camera,frame,detected,runtime,app=local
    runtime._material_pass(classifier(),frame,detected['detections'])
    if reset=='session':camera.session_id=str(uuid4())
    else:camera.calibration_hash='c'*64
    client=app.test_client()
    assert client.get('/api/camera/comparison').json['rows']==[]
    assert client.get('/api/camera/comparison/image.jpg?frame_id='+frame.id).status_code==410


@pytest.mark.parametrize('reset',['session','calibration'])
def test_old_pass_cannot_publish_after_a_camera_context_reset(local,reset):
    store,camera,frame,detected,runtime,app=local
    model=classifier()
    original=model.classify.side_effect
    def classify(crop):
        if reset=='session':camera.session_id=str(uuid4())
        else:runtime.material_context=(frame.metadata['session_id'],tuple(frame.color_rgb.shape),'c'*64)
        return original(crop)
    model.classify.side_effect=classify
    runtime._material_pass(model,frame,detected['detections'])
    assert runtime.comparison()['rows']==[]
    assert not runtime.material_comparison_images
