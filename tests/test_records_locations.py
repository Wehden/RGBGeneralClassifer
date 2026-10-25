import json
from uuid import uuid4
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_web import client
from test_events import event,sources


def ingest(c,event):
    return c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40})


def test_records_filter_pagination_and_typed_page_download(client,event):
    c,store=client
    for i in range(3):
        event['event_id']=str(uuid4());ingest(c,event)
        observation=store.get(event['event_id'])
        store.enhance(observation,0,{'label':'can','confidence':.9,'model':'qwen','materials':[{'label':'metal','confidence':.95,'reason':'Reflective lid'}]},str(uuid4()))
    first=c.get('/api/review/records?material=metal&limit=2').json
    assert len(first['records'])==2
    second=c.get('/api/review/records',query_string={'material':'metal','limit':2,'before':first['next_cursor']}).json
    assert len(second['records'])==1
    assert not {r['detection_id'] for r in first['records']} & {r['detection_id'] for r in second['records']}
    assert c.get('/api/review/records?material=glass').json['records']==[]
    assert len(c.get('/api/review/records?label=can').json['records'])==3
    assert c.get('/api/review/records?source=elsewhere').json['records']==[]
    assert c.get('/api/review/records?from=2099-01-01T00:00:00Z').json['records']==[]
    assert c.get('/api/review/records?from=2026-01-01').status_code==400
    for fmt in ['parquet','arrow']:
        r=c.get('/api/review/records?material=metal&limit=2&format='+fmt)
        buf=pa.BufferReader(r.data)
        table=pq.read_table(buf) if fmt=='parquet' else pa.ipc.open_file(buf).read_all()
        assert table.num_rows==2
        assert table.to_pylist()[0]['materials'][0]['label']=='metal'
        assert table.schema.field('observed_at').type.tz=='UTC'
        assert r.headers['X-RGB-Next-Cursor']


def test_location_is_opt_in_audited_and_only_applies_to_future_observations(client,event):
    c,store=client
    ingest(c,event);old_id=event['event_id']
    payload={'method':'manual','label':'Home desk','latitude':None,'longitude':None,'accuracy_m':None}
    assert c.post('/api/sources/portentauno/location',json=payload).status_code==400
    r=c.post('/api/sources/portentauno/location',json=payload,headers={'X-RGB-Request':'location','X-Laminarauth-Identity':'operator'})
    assert r.status_code==200
    assert r.json['actor']=='operator'
    event['event_id']=str(uuid4());ingest(c,event)
    saved=store.raw(event['event_id'])
    assert saved['location']=='Home desk'
    assert saved['location_record']['version']==1
    assert saved['location_kind']=='manual'
    assert store.raw(old_id)['location']=='Teaching bench'
    assert c.get('/api/review/records?location=home').json['records'][0]['observation_id']==event['event_id']
    disabled={**payload,'method':'disabled'}
    c.post('/api/sources/portentauno/location',json=disabled,headers={'X-RGB-Request':'location'})
    event['event_id']=str(uuid4());ingest(c,event)
    assert store.raw(event['event_id'])['location'] is None
    assert store.raw(old_id)['location']=='Teaching bench'
    assert store.redis.xlen('rgb:location-updates')==2


def test_location_coordinates_and_role_are_validated(client,monkeypatch):
    c,_=client
    payload={'method':'browser','label':'','latitude':91,'longitude':0,'accuracy_m':5}
    headers={'X-RGB-Request':'location'}
    assert c.post('/api/sources/portentauno/location',json=payload,headers=headers).status_code==400
    payload['latitude']=1.25
    assert c.post('/api/sources/portentauno/location',json=payload,headers=headers).status_code==200
    loc=c.get('/api/sources/portentauno/location').json
    assert loc['accuracy_m']==5
    monkeypatch.setenv('ROLE','ingest')
    for path in ['/records','/api/review/records','/api/sources/portentauno/location']:
        assert c.get(path).status_code==404
