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
    assert item['sorting_status'] == 'unassessed'


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
