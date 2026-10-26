"""Regressions for reviewed failure paths and durable VLM pause."""
import json
import threading
import time
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flask import Flask
from redis.exceptions import ConnectionError

from board.streaming import response as stream_response
from rgb.enrichment import AnalysisPaused, analyze, run
from rgb.events import validate_event
from rgb.export import build_export
from test_enrichment import Finished
from test_events import event, sources
from test_identification import inventory, recorded, reading, response
from test_web import client


def test_board_head_and_unstarted_response_release_viewer_slots():
    app=Flask(__name__)
    viewers=threading.BoundedSemaphore(4)
    condition=threading.Condition()
    state={'sequence':1,'jpeg':b'jpeg','last_frame':time.time()}
    app.add_url_rule('/stream',view_func=lambda:stream_response(condition,state,viewers))
    with app.test_client() as c:
        for _ in range(10):
            result=c.head('/stream')
            assert result.status_code==200
            result.close()
    held=[stream_response(condition,state,viewers) for _ in range(4)]
    assert stream_response(condition,state,viewers)[1]==429
    for result in held:
        result.close();result.close()
    result=stream_response(condition,state,viewers)
    assert next(iter(result.response)).startswith(b'--frame')
    result.close()


def test_cursor_pair_retries_if_second_initial_read_fails(monkeypatch):
    redis=Mock()
    redis.get.side_effect=[None,ConnectionError('manual cursor'), '12-0','34-0']
    redis.xread.side_effect=Finished()
    store=SimpleNamespace(redis=redis,settings=lambda:{'vlm_paused':False})
    monkeypatch.setattr('rgb.enrichment.time.sleep',lambda _:None)
    with pytest.raises(Finished):run(store,[])
    assert redis.get.call_count==4
    assert redis.xread.call_args.args[0]=={'rgb:analysis:requests':'34-0','rgb:events':'12-0'}


@pytest.mark.parametrize('format',['parquet','arrow'])
def test_started_attempt_exports_running_instead_of_failure(recorded,format):
    store,event=recorded
    store.record_attempt(event,[0],{'attempt_id':'in-flight','started_at':event['observed_at'],'model':'qwen','outcome':'started'})
    directory,path=build_export(store,format)
    try:
        with zipfile.ZipFile(path) as bundle:
            raw=pa.BufferReader(bundle.read('detections.'+format))
            table=pq.read_table(raw) if format=='parquet' else pa.ipc.open_file(raw).read_all()
            assert table.to_pylist()[0]['analysis_status']=='running'
    finally:directory.cleanup()


@pytest.mark.parametrize('legacy',[False,True])
def test_job_replay_uses_persisted_candidate_in_archive_and_live(recorded,monkeypatch,legacy):
    store,event=recorded
    original={**reading(label='computer monitor'),'model':'qwen','endpoint':'http://vlm','attempt_id':'original'}
    saved=store.enhance(event,0,original,'interrupted')
    if legacy:store.redis.delete('rgb:enhancement-values')
    post=Mock(return_value=response([reading(label='television')]))
    monkeypatch.setattr('rgb.enrichment.requests.post',post)
    assert analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'interrupted')['status']=='complete'
    assert not post.called
    live=json.loads(store.redis.hget('rgb:refinements',event['source_id']))['readings'][0]
    archive=store.get(event['event_id'])['detections'][0]['enhancement']
    assert live['label']==archive['label']=='computer monitor'
    assert live['attempt_id']==archive['attempt_id']=='original'
    assert store.enhance(event,0,{**original,'label':'wrong'},'interrupted')==saved


def test_evaluation_rejects_changed_candidate_and_preserves_displayed_settings(client,event):
    c,store=client
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    observed=store.get(event['event_id'])
    d=observed['detections'][0]
    form={'reference_label':'can','detector_result':'incorrect','vlm_result':'correct','notes':'Draft','review_context':d['review_context']}
    store.enhance(observed,0,{**reading(),'model':'qwen'},'A')
    path='/api/detections/'+d['detection_id']+'/evaluate'
    assert c.post(path,json=form,headers={'X-RGB-Request':'evaluation'}).status_code==409
    assert store.redis.xlen('rgb:evaluations')==0
    store.update_settings({'display_mode':'materials'})
    form['review_context']=store.get(event['event_id'])['detections'][0]['review_context']
    store.update_settings({'display_mode':'objects','vlm_threshold':.95})
    result=c.post(path,json=form,headers={'X-RGB-Request':'evaluation'})
    assert result.status_code==201
    assert result.json['mode']=='materials'
    assert result.json['settings']['version']==1
    assert result.json['settings']['vlm_threshold']==.65
    assert result.json['candidate']['job_id']=='A'


def test_buffered_observation_uses_location_revision_at_capture(client,event,monkeypatch):
    c,store=client
    loc={'method':'manual','label':'Desk A','latitude':None,'longitude':None,'accuracy_m':None}
    monkeypatch.setattr('rgb.store.now',lambda:'2026-01-01T00:00:00+00:00')
    a=store.update_location('portentauno',loc)
    monkeypatch.setattr('rgb.store.now',lambda:'2026-01-01T02:00:00+00:00')
    store.update_location('portentauno',{**loc,'label':'Desk B'})
    event['observed_at']='2026-01-01T01:00:00+00:00'
    assert c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40}).status_code==201
    assert store.raw(event['event_id'])['location_record']==a
    event.update(event_id=str(uuid4()),observed_at='2025-12-31T23:00:00+00:00')
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    assert store.raw(event['event_id'])['location']=='Teaching bench'
    assert 'location_record' not in store.raw(event['event_id'])


def test_pause_api_is_durable_audited_and_preserves_other_controls(client,monkeypatch):
    c,store=client
    headers={'X-RGB-Request':'settings','X-Laminarauth-Identity':'operator'}
    store.update_settings({'detector_threshold':.39,'display_mode':'materials'})
    result=c.post('/api/settings',json={'vlm_paused':True},headers=headers)
    assert result.status_code==200
    assert result.json['detector_threshold']==.39
    assert result.json['display_mode']=='materials'
    assert c.get('/api/vlm/status').json=={'paused':True,'active':None,'enabled':True,'models':[]}
    assert json.loads(store.redis.xrevrange('rgb:settings-updates',count=1)[0][1]['value'])['actor']=='operator'
    assert c.post('/api/settings',json={'vlm_paused':'true'},headers=headers).status_code==400
    assert c.post('/api/settings',json={'vlm_paused':False},headers=headers).json['vlm_paused'] is False
    monkeypatch.setenv('ROLE','ingest')
    assert c.get('/api/vlm/status').status_code==404


def test_worker_pause_holds_cursors_and_resume_processes_same_job(monkeypatch):
    redis=Mock()
    redis.get.return_value=None
    redis.xread.return_value=[('rgb:events',[('1-0',{'event':json.dumps({'event_id':'scene'})})])]
    redis.hget.return_value=json.dumps({'status':'running','completed_boxes':4})
    settings=iter([False,True])
    store=SimpleNamespace(redis=redis,settings=lambda:{'vlm_paused':next(settings)},get=lambda _: {'analysis':{'status':'queued'}},processing_failure=Mock())
    monkeypatch.setattr('rgb.enrichment.analyze',Mock(side_effect=AnalysisPaused()))
    monkeypatch.setattr('rgb.enrichment.time.sleep',Mock(side_effect=Finished()))
    with pytest.raises(Finished):run(store,[])
    assert json.loads(redis.hset.call_args.args[2])['status']=='paused'
    assert not redis.set.called
    assert not store.processing_failure.called
    redis.xread.assert_called_once()


def test_pause_between_batches_finishes_inflight_and_resumes_missing_boxes(client,event,monkeypatch):
    _,store=client
    event['detections']=[dict(event['detections'][0]) for _ in range(5)]
    normalized,image=validate_event(event,{'portentauno':{'url':'http://camera','location':'Desk'}})
    store.save(normalized,image)
    event=store.get(event['event_id'])
    def first_request(*args,**kwargs):
        assert store.redis.get('rgb:vlm:active')
        store.update_settings({'vlm_paused':True})
        return response([reading(index=i) for i in range(4)])
    post=Mock(side_effect=first_request)
    monkeypatch.setattr('rgb.enrichment.requests.post',post)
    with pytest.raises(AnalysisPaused):analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job')
    assert post.call_count==1
    assert store.redis.get('rgb:vlm:active') is None
    assert store.redis.hget('rgb:metrics','request_success')=='1'
    assert sum(bool(d['enhancement']) for d in store.get(event['event_id'])['detections'])==4
    with pytest.raises(AnalysisPaused):analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job')
    assert post.call_count==1
    store.update_settings({'vlm_paused':False})
    post.side_effect=None;post.return_value=response([reading(index=4)])
    assert analyze(store,event,[{'url':'http://vlm','model':'qwen'}],'job')['status']=='complete'
    assert post.call_count==2
    assert all(d['enhancement'] for d in store.get(event['event_id'])['detections'])
