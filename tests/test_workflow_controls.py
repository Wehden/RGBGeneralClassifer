import json
from datetime import datetime,timedelta,timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from rgb.enrichment import analyze,latest_pending,run
from test_enrichment import Finished
from test_web import client
from test_events import event,sources
from test_identification import reading,response


def test_paused_capture_stays_manual_after_resume(client,event,monkeypatch):
    c,store=client
    start=datetime.now(timezone.utc)-timedelta(seconds=30)
    monkeypatch.setattr('rgb.store.now',lambda:start.isoformat())
    store.update_settings({'vlm_paused':True})
    event['observed_at']=(start+timedelta(seconds=5)).isoformat()
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    monkeypatch.setattr('rgb.store.now',lambda:(start+timedelta(seconds=10)).isoformat())
    store.update_settings({'vlm_paused':False})
    saved=store.get(event['event_id'])
    assert saved['analysis']['status']=='manual_only'
    assert saved['analysis_policy']['vlm_paused'] is True
    assert latest_pending(store) is None
    original=store.redis.xread
    calls=0
    def stop_after_one(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls>1:raise Finished()
        return original(*args,**kwargs)
    monkeypatch.setattr(store.redis,'xread',stop_after_one)
    model=Mock()
    monkeypatch.setattr('rgb.enrichment.analyze',model)
    with pytest.raises(Finished):run(store,[{'url':'http://vlm','model':'qwen'}])
    model.assert_not_called()
    assert store.redis.get('rgb:vlm:cursor')==store.redis.hget('rgb:ids',event['event_id'])


def test_preview_observations_are_not_stored(client,event,monkeypatch):
    c,store=client
    before=datetime.now(timezone.utc)-timedelta(seconds=2)
    monkeypatch.setattr('rgb.store.now',lambda:before.isoformat())
    result=c.post('/api/settings',json={'classification_running':True,'record_observations':False},headers={'X-RGB-Request':'settings'})
    assert result.status_code==200
    r=c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    assert r.json['recorded'] is False
    assert store.redis.xlen('rgb:events')==0
    assert store.redis.xlen('rgb:analysis:requests')==0
    assert not list(store.root.iterdir())
    assert c.post('/api/settings',json={'classification_running':'yes','record_observations':False},headers={'X-RGB-Request':'settings'}).status_code==400


def test_individual_detection_request_reads_only_that_crop(client,event,monkeypatch):
    c,store=client
    event['detections']=[dict(event['detections'][0]) for _ in range(5)]
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    saved=store.get(event['event_id'])
    key=saved['detections'][4]['detection_id']
    url='/api/detections/'+key+'/analyze'
    assert c.post(url).status_code==400
    assert c.post(url,headers={'X-RGB-Request':'analysis'}).status_code==202
    assert c.post(url,headers={'X-RGB-Request':'analysis'}).json['created'] is False
    _,fields=store.redis.xrange('rgb:analysis:requests')[0]
    assert json.loads(fields['box_indices'])==[4]
    post=Mock(return_value=response([reading(index=4)]))
    monkeypatch.setattr('rgb.enrichment.requests.post',post)
    state=analyze(store,saved,[{'url':'http://vlm','model':'qwen'}],fields['job_id'],indices=[4])
    assert post.call_count==1
    assert len(post.call_args.kwargs['json']['messages'][0]['images'])==2
    assert state['status']=='partial'
    updated=store.get(event['event_id'])
    assert updated['detections'][4]['enhancement']
    assert all(not d['enhancement'] for d in updated['detections'][:4])


def test_catch_up_snapshot_does_not_include_future_frames(client,event):
    c,store=client
    # Multiple scenes and explicit snapshot boundary.
    for _ in range(3):
        event['event_id']=str(uuid4())
        c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    first=c.post('/api/analysis/catch-up',json={'source_id':'portentauno'},headers={'X-RGB-Request':'analysis'})
    assert first.status_code==202
    assert first.json['queued']==3
    assert first.json['done'] is True
    event['event_id']=str(uuid4())
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    second=c.post('/api/analysis/catch-up',json={'source_id':'portentauno','batch_id':first.json['batch_id']},headers={'X-RGB-Request':'analysis'})
    assert second.json['queued']==3
    assert store.redis.xlen('rgb:analysis:requests')==3


def test_latest_pointer_uses_capture_time_and_does_not_advance_fifo(client,event,monkeypatch):
    c,store=client
    event['observed_at']=(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat()
    old=event['event_id']
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    event['event_id']=str(uuid4());event['observed_at']=datetime.now(timezone.utc).isoformat()
    fresh=event['event_id']
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    event['event_id']=str(uuid4());event['observed_at']=(datetime.now(timezone.utc)-timedelta(seconds=30)).isoformat()
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    assert latest_pending(store)['event_id']==fresh
    jobs=[]
    def model(s,e,endpoints,job_id,**kwargs):
        jobs.append(e['event_id'])
        s.redis.hset('rgb:analysis',e['event_id'],json.dumps({'schema_version':3,'status':'complete'}))
        if len(jobs)==2:raise Finished()
    monkeypatch.setattr('rgb.enrichment.analyze',model)
    with pytest.raises(Finished):run(store,[{'url':'http://vlm','model':'qwen'}])
    assert jobs==[fresh,old]
    # The live choice cannot jump the historical cursor over unprocessed rows.
    assert store.redis.get('rgb:vlm:cursor') is None


def test_material_detector_display_query_and_arrow_provenance(client,event):
    import io
    import zipfile
    import pyarrow as pa
    c,store=client
    event['detector']['config']={'task':'materials','prompts':['metal object'],'prompt_mode':'text'}
    event['detections'][0]['label']='metal'
    c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    store.update_settings({'display_mode':'materials'})
    d=store.get(event['event_id'])['detections'][0]
    assert d['display_label']=='metal'
    assert d['label_provider']=='detector'
    queried=c.get('/api/review/records?material=metal').json['records']
    assert len(queried)==1
    assert queried[0]['initial_label_kind']=='material'
    download=c.get('/api/review/export/arrow')
    with zipfile.ZipFile(io.BytesIO(download.data)) as archive:
        table=pa.ipc.open_file(pa.BufferReader(archive.read('material_readings.arrow'))).read_all()
        assert table.to_pylist()[0]['provider']=='detector'
        assert table.to_pylist()[0]['label']=='metal'
        assert 'analysis_actions.arrow' in archive.namelist()
    download.close()


def test_empty_frame_retains_below_threshold_proposals(client,event):
    c,store=client
    proposal=dict(event['detections'][0]);proposal['confidence']=.2
    event.update(detections=[],detector_proposals=[proposal])
    r=c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})
    assert r.status_code==201
    saved=store.get(event['event_id'])
    assert saved['detections']==[]
    assert saved['detector_proposals'][0]['confidence']==.2
    proposal['confidence']=float('nan')
    event['event_id']=str(uuid4())
    assert c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40}).status_code==400
