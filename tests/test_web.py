import json

import fakeredis
import pytest
from rgb.store import Store
from rgb.web import create_app
from test_events import event, sources


@pytest.fixture
def client(tmp_path,sources):
    store = Store(fakeredis.FakeRedis(decode_responses=True), tmp_path)
    app = create_app(store,sources,'x'*40)
    app.config['TESTING'] = True
    return app.test_client(), store


def test_ingest_requires_token(client,event):
    c,store = client
    assert c.post('/api/ingest',json=event).status_code == 401
    assert store.redis.xlen('rgb:events') == 0


def test_dedup_replay_history_and_image(client,event):
    c,store = client
    headers={'Authorization':'Bearer '+'x'*40}
    first=c.post('/api/ingest',json=event,headers=headers)
    assert first.status_code == 201
    assert c.post('/api/ingest',json=event,headers=headers).status_code == 200
    assert store.redis.xlen('rgb:events') == 1
    result=c.get('/api/events').json
    assert result[0]['event_id'] == event['event_id']
    assert result[0]['opinions'] == []
    assert result[0]['analysis']['status'] == 'queued'
    assert c.get('/api/events?before='+result[0]['cursor']).json == []
    assert c.get('/api/events?before=broken').status_code == 400
    assert c.get('/api/events/'+event['event_id']+'/image.jpg').data.startswith(b'\xff\xd8')
    assert c.get('/api/events/'+event['event_id']).json['location'] == 'Teaching bench'
    assert c.get('/api/events/not-a-uuid').status_code == 404
    assert c.get('/api/events/'+event['event_id']).headers['Cache-Control'] == 'no-store'


def test_separate_model_opinions(client,event):
    c,store=client
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    store.redis.hset('rgb:opinions',event['event_id'],json.dumps([{'model':'vlm','text':'Maybe glass','status':'complete'}]))
    item=c.get('/api/events/'+event['event_id']).json
    assert item['opinions'][0]['model'] == 'vlm'
    assert item['detections'][0]['label'] == 'bottle'


def test_source_url_private_and_unknown_source(client):
    c,_=client
    assert 'url' not in c.get('/api/sources').json[0]
    assert c.get('/api/live/unknown/stream').status_code == 404
    assert c.get('/api/live/unknown/status').status_code == 404
    assert c.get('/').status_code == 200


def test_ingest_role_cannot_expose_history_images_or_live(client,event,monkeypatch):
    c,_=client
    monkeypatch.setenv('ROLE','ingest')
    for path in ['/','/api/events','/api/sources','/api/live/portentauno/stream','/static/app.js','/api/events/'+event['event_id']+'/image.jpg']:
        assert c.get(path).status_code == 404
    assert c.get('/healthz').status_code == 200
    assert c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40}).status_code == 201


def test_duplicate_uuid_cannot_replace_original_evidence(client,event):
    c,store=client
    headers={'Authorization':'Bearer '+'x'*40}
    c.post('/api/ingest',json=event,headers=headers)
    original=store.get(event['event_id'])
    event['detections'][0]['label']='different claim'
    replay=c.post('/api/ingest',json=event,headers=headers)
    assert replay.status_code == 200
    assert replay.json['created'] is False
    assert store.get(event['event_id'])['detections'] == original['detections']


def test_unconsumed_stream_response_releases_admission(client,monkeypatch):
    from unittest.mock import Mock
    c,_=client
    upstream=Mock()
    upstream.iter_content.return_value=iter([b'frame'])
    monkeypatch.setattr('rgb.web.requests.get',lambda *a,**k:upstream)
    # WSGI may close a response before iterating it. Every admission must return.
    for _ in range(8):
        with c.application.test_request_context('/api/live/portentauno/stream'):
            response=c.application.view_functions['stream']('portentauno')
            assert response.status_code == 200
            response.close()
    assert upstream.close.call_count == 8


def test_threshold_controls_are_validated_and_audited(client):
    c,store=client
    assert c.get('/api/settings').json['detector_threshold']==.45
    data={'detector_threshold':.35,'vlm_threshold':.8}
    assert c.post('/api/settings',json=data).status_code==400
    headers={'X-RGB-Request':'settings','Origin':'https://localhost'}
    result=c.post('/api/settings',json=data,headers=headers)
    assert result.status_code==200
    assert result.json['version']==1
    assert c.get('/api/settings').json['vlm_threshold']==.8
    assert store.redis.xlen('rgb:settings-updates')==1
    assert c.post('/api/settings',json={**data,'vlm_threshold':2},headers=headers).status_code==400
    assert c.post('/api/settings',json=data,headers={**headers,'Origin':'https://elsewhere'}).status_code==403


def test_board_policy_requires_token_and_tracks_live_thresholds(client):
    c,store=client
    assert c.get('/api/board-policy/portentauno').status_code==401
    store.update_settings({'detector_threshold':.3,'vlm_threshold':.7})
    r=c.get('/api/board-policy/portentauno',headers={'Authorization':'Bearer '+'x'*40})
    assert r.json['settings']['detector_threshold']==.3
    assert r.json['refinements'] is None


def test_review_and_analysis_routes_stay_off_ingest_role(client,event,monkeypatch):
    c,_=client
    monkeypatch.setenv('ROLE','ingest')
    for path in ['/api/settings','/api/review/metrics','/api/review/export/parquet','/api/detections/'+event['event_id']+'/history']:
        assert c.get(path).status_code==404
    assert c.post('/api/events/'+event['event_id']+'/analyze',headers={'X-RGB-Request':'analysis'}).status_code==404


def test_evaluation_records_hypotheses_with_ground_truth(client,event):
    c,store=client
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    d=c.get('/api/events/'+event['event_id']).json['detections'][0]
    review={'reference_label':'can','detector_result':'incorrect','vlm_result':'unscored','notes':'Pull tab visible'}
    r=c.post('/api/detections/'+d['detection_id']+'/evaluate',json=review,headers={'X-RGB-Request':'evaluation'})
    assert r.status_code==201
    assert r.json['initial']['label']=='bottle'
    assert r.json['reference_label']=='can'
    assert store.redis.xlen('rgb:evaluations')==1
    assert store.history(d['detection_id'])[-1]['kind']=='evaluation'


def test_export_download_releases_slot_for_next_format(client,event):
    c,_=client
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    for format in ['parquet','arrow']:
        response=c.get('/api/review/export/'+format)
        assert response.status_code==200
        assert response.data.startswith(b'PK')
        assert 'attachment' in response.headers['Content-Disposition']
        response.close()
