import json
from datetime import datetime,timezone
from unittest.mock import Mock

import fakeredis
import pytest
import requests

from board.refinements import match
from rgb.enrichment import analyze, crops, read_boxes, validate_readings
from rgb.events import validate_event
from rgb.store import Store
from test_events import event, sources


@pytest.fixture(autouse=True)
def inventory(monkeypatch):
    result=Mock()
    result.json.return_value={'models':[{'name':'qwen','digest':'model-digest'}]}
    monkeypatch.setattr('rgb.enrichment.requests.get',Mock(return_value=result))


@pytest.fixture
def recorded(tmp_path,event,sources):
    store=Store(fakeredis.FakeRedis(decode_responses=True),tmp_path)
    initial,image=validate_event(event,sources)
    store.save(initial,image)
    return store,store.get(initial['event_id'])


def reading(index=0,label='beverage can',confidence=.9):
    return {'box_index':index,'label':label,'confidence':confidence,'comparison':'disagrees','reason':'Cylindrical body and pull-tab lid.','materials':[{'label':'metal','confidence':.93,'reason':'Metallic body and lid.'}]}


def response(readings):
    payload={'model':'qwen','done_reason':'stop','message':{'content':json.dumps({'observations':readings})},'eval_count':73,'prompt_eval_count':151,'total_duration':1000000000}
    result=Mock(status_code=200,text=json.dumps(payload))
    result.json.return_value=payload
    return result


def test_enhancement_preserves_initial_and_replays_without_duplicates(recorded,monkeypatch):
    store,event=recorded
    result=response([reading()])
    post=Mock(return_value=result)
    monkeypatch.setattr('rgb.enrichment.requests.post',post)
    state=analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job1')
    assert state['status']=='complete'
    projected=store.get(event['event_id'])['detections'][0]
    assert projected['label']=='bottle'
    assert projected['display_label']=='beverage can'
    assert store.raw(event['event_id'])['detections'][0]['label']=='bottle'
    history=store.history(projected['detection_id'])
    assert [x['kind'] for x in history]==['initial','model_attempt','model_attempt','enhanced']
    assert [x['outcome'] for x in history if x['kind']=='model_attempt']==['started','success']
    assert history[2]['raw_response']==result.text
    assert history[2]['output_tokens']==73
    assert history[3]['candidate']['confidence']==.9
    assert history[3]['decision_settings']['vlm_threshold']==.65
    store.enhance(event,0,reading(),'job1')
    assert len(store.history(projected['detection_id']))==4
    images=post.call_args.kwargs['json']['messages'][0]['images']
    assert len(images)==2


def test_confidence_threshold_changes_projection_without_destroying_scores(recorded,monkeypatch):
    store,event=recorded
    monkeypatch.setattr('rgb.enrichment.requests.post',Mock(return_value=response([reading(confidence=.6)])))
    analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job')
    d=store.get(event['event_id'])['detections'][0]
    assert d['display_label']=='bottle'
    assert d['enhancement']['confidence']==.6
    settings=store.update_settings({'detector_threshold':.4,'vlm_threshold':.55})
    assert settings['version']==1
    assert store.get(event['event_id'])['detections'][0]['display_label']=='beverage can'
    assert store.redis.xlen('rgb:settings-updates')==1


def test_failure_is_recorded_with_response_and_stable_detection_link(recorded,monkeypatch):
    store,event=recorded
    result=Mock(status_code=503,text='model unavailable')
    result.raise_for_status.side_effect=requests.HTTPError('503')
    monkeypatch.setattr('rgb.enrichment.requests.post',Mock(return_value=result))
    state=analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job')
    assert state['status']=='failed'
    d=store.get(event['event_id'])['detections'][0]
    assert d['display_label']=='bottle'
    history=store.history(d['detection_id'])
    assert history[-1]['outcome']=='failed'
    assert history[-1]['http_status']==503
    assert history[-1]['raw_response']=='model unavailable'
    assert history[-1]['error_type']=='HTTPError'
    assert store.redis.hget('rgb:metrics','request_failed')=='1'
    assert history[-1]['latency_ms']>=0


@pytest.mark.parametrize('readings',[[reading(index=1)],[],[reading(),reading()],[reading(confidence=float('nan'))],[{**reading(),'comparison':'whatever'}]])
def test_rejects_mismatched_or_invalid_box_answers(readings):
    payload={'message':{'content':json.dumps({'observations':readings})}}
    with pytest.raises(ValueError):validate_readings(payload,[{'index':0}])


def test_live_label_follows_track_when_screen_pixels_change():
    d={'class_id':44,'track_id':'track-a','box':[.1,.2,.8,.9]}
    ref={'event_id':'scene-id','observed_at':datetime.now(timezone.utc).isoformat(),'readings':[{'class_id':44,'track_id':'track-a','box':d['box'],'fingerprint':'1234000000000000','confidence':.9,'label':'can'}]}
    now=datetime.now(timezone.utc).timestamp()
    matched=match(d,'ffffffffffffffff',ref,now,.65)
    assert matched['label']=='can'
    assert matched['match_hash_distance']>4
    assert matched['match_method']=='continuous_track'
    assert match({**d,'track_id':'replacement'},'ffffffffffffffff',ref,now,.65) is None
    assert match(d,'ffffffffffffffff',ref,now+100,.65)['label']=='can'
    assert match(d,'ffffffffffffffff',ref,now,.95) is None


def test_initial_replay_and_legacy_analysis_request(recorded):
    store,event=recorded
    store.ensure_detections(store.raw(event['event_id']))
    assert len(store.history(event['detections'][0]['detection_id']))==1
    store.redis.hdel('rgb:analysis',event['event_id'])
    assert store.request_analysis(event['event_id']) is True
    assert store.request_analysis(event['event_id']) is False
    assert store.redis.xlen('rgb:analysis:requests')==1
