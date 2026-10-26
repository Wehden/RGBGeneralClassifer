import base64
import json
import time
from unittest.mock import Mock

import fakeredis
import pytest
import requests
from board.transport import Transport
from rgb.store import Store
from rgb.web import create_app
from test_events import event, sources


@pytest.fixture
def portable(tmp_path,monkeypatch):
    monkeypatch.setenv('DEVICE_TOKENS_JSON',json.dumps({'unoq':'u'*40,'other':'o'*40}))
    store=Store(fakeredis.FakeRedis(decode_responses=True),tmp_path)
    app=create_app(store,{'unoq':{'label':'Uno Q'},'other':{}},'x'*40)
    app.config['TESTING']=True
    return app.test_client(),store


def auth(source='unoq'):
    return {'Authorization':'Bearer '+('u' if source=='unoq' else 'o')*40}


def live_frame(event):
    return {'status':{'source_id':'unoq','last_frame':time.time(),'sequence':1,
        'error':None,'detections':[],'model':'test'},'image_b64':event['image_b64'],'frame_age_s':0}


def test_source_token_not_ip_identity(portable,event):
    c,store=portable
    event['source_id']='unoq'
    for address in ['192.168.10.42','10.42.0.56','203.0.113.1','2001:db8::1']:
        r=c.post('/api/device/unoq/observations',json=event,headers=auth(),environ_base={'REMOTE_ADDR':address})
        assert r.status_code in (200,201)
        assert r.json['event_id']==event['event_id']
        assert c.get('/api/device/unoq/policy',headers=auth()).status_code==200
    assert store.redis.xlen('rgb:events')==1
    assert c.post('/api/device/other/observations',json={**event,'source_id':'other'},headers=auth()).status_code==401
    assert c.post('/api/device/unoq/observations',json={**event,'source_id':'other'},headers=auth()).status_code==400
    assert c.get('/api/device/unoq/policy',headers={'Authorization':'Bearer '+'x'*40}).status_code==401
    assert c.get('/api/device/missing/policy',headers=auth()).status_code==401


def test_live_upload_no_archive_and_disconnect(portable,event,monkeypatch):
    c,store=portable
    monkeypatch.setenv('ROLE','ingest')
    payload=live_frame(event)
    assert c.post('/api/device/unoq/live',json=payload).status_code==401
    assert c.post('/api/device/unoq/live',json=payload,headers=auth()).status_code==200
    assert store.redis.xlen('rgb:events')==0
    assert not list(store.redis.scan_iter('rgb:live:*'))
    assert not list(store.root.iterdir())
    assert c.get('/api/live/unoq/status').status_code==404
    monkeypatch.delenv('ROLE')
    assert c.get('/api/live/unoq/status').json['model']=='test'
    response=c.get('/api/live/unoq/stream',buffered=False)
    assert base64.b64decode(event['image_b64']) in next(response.response)
    response.close()
    # Retrying an old frame cannot turn it into a new live capture.
    payload['status']['last_frame']=time.time()-30
    assert c.post('/api/device/unoq/live',json=payload,headers=auth()).status_code==200
    assert c.get('/api/live/unoq/status').status_code==503
    assert c.get('/api/live/unoq/stream').status_code==503
    payload['status']['last_frame']=time.time()
    assert c.post('/api/device/unoq/live',json=payload,headers=auth()).status_code==200
    assert c.get('/api/live/unoq/status').status_code==200
    c.application.extensions['live_cache'].frames.clear()
    assert c.get('/api/live/unoq/status').status_code==503


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(image_b64='not jpeg'),
    lambda p:p['status'].update(source_id='other'),
    lambda p:p['status'].update(last_frame=float('nan')),
    lambda p:p['status'].update(last_frame=time.time()+60),
    lambda p:p.update(image_b64=base64.b64encode(b'not jpeg').decode()),
    lambda p:p.update(frame_age_s=-1),
])
def test_invalid_frame_rejected(portable,event,mutate):
    c,store=portable
    payload=live_frame(event)
    mutate(payload)
    assert c.post('/api/device/unoq/live',json=payload,headers=auth()).status_code==400
    assert store.redis.get('rgb:live:unoq') is None


@pytest.mark.parametrize('origin',['http://rgb.wehden.com','https://user:secret@rgb.wehden.com','https://rgb.wehden.com/path','https://rgb.wehden.com?redirect=elsewhere','https://rgb.wehden.com#fragment'])
def test_transport_requires_https_origin(origin):
    with pytest.raises(ValueError):Transport('unoq',{'ARCHIVE_URL':origin,'INGEST_TOKEN':'u'*40})


def test_redirect_and_html_do_not_acknowledge():
    t=Transport('unoq',{'ARCHIVE_URL':'https://rgb.wehden.com','INGEST_TOKEN':'u'*40})
    response=Mock(status_code=302)
    t.session.request=Mock(return_value=response)
    with pytest.raises(requests.HTTPError):t.request('GET','/policy')
    args=t.session.request.call_args
    assert args.kwargs['allow_redirects'] is False
    assert args.args[1]=='https://rgb.wehden.com/api/device/unoq/policy'
    response.status_code=200
    response.json.side_effect=ValueError('HTML login')
    with pytest.raises(ValueError):t.request('GET','/policy')


def test_stream_admission_and_stale_disconnect(portable,event,monkeypatch):
    from rgb import live
    c,_=portable
    payload=live_frame(event)
    c.post('/api/device/unoq/live',json=payload,headers=auth())
    held=[]
    with c.application.test_request_context():
        stream=c.application.view_functions['stream']
        held=[stream('unoq') for _ in range(6)]
        assert stream('unoq')[1]==429
        for response in held:response.close()
        response=stream('unoq')
        iterator=iter(response.response)
        assert next(iterator).startswith(b'--frame')
        c.application.extensions['live_cache'].frames.clear()
        with pytest.raises(StopIteration):next(iterator)
        response.close()
        # Camera loss does not strand all viewer slots.
        c.post('/api/device/unoq/live',json=live_frame(event),headers=auth())
        again=stream('unoq')
        assert again.status_code==200
        again.close()


def test_large_live_upload_does_not_spool_to_disk(portable,monkeypatch):
    import io
    import threading
    from PIL import Image
    from waitress import create_server
    from common.http import OPTIONS
    c,store=portable
    jpeg=io.BytesIO()
    Image.effect_noise((1280,720),100).convert('RGB').save(jpeg,'JPEG',quality=95)
    raw=jpeg.getvalue()
    assert 524288<len(raw)<1_125_000
    payload={'status':{'source_id':'unoq','last_frame':time.time()},'image_b64':base64.b64encode(raw).decode(),'frame_age_s':0}
    forbidden=Mock(side_effect=AssertionError('Preview data reached a disk buffer'))
    monkeypatch.setattr('tempfile.TemporaryFile',forbidden)
    server=create_server(c.application,host='127.0.0.1',port=0,threads=2,**OPTIONS)
    worker=threading.Thread(target=server.run,daemon=True)
    worker.start()
    try:
        result=requests.post('http://127.0.0.1:'+str(server.effective_port)+'/api/device/unoq/live',json=payload,headers=auth(),timeout=5)
        assert result.status_code==200
        assert not list(store.redis.scan_iter('rgb:live:*'))
        assert not list(store.root.iterdir())
        forbidden.assert_not_called()
    finally:
        server.close()
        server.task_dispatcher.shutdown()


def test_live_uploader_survives_stale_then_fresh_camera_frame():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from test_enrichment import Finished
    tree=ast.parse(Path('board/main.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='push_live')
    state={};sent=[]
    frames=iter([{'sequence':1,'last_frame':0,'jpeg':b'stale'}, {'sequence':2,'last_frame':99,'jpeg':b'fresh'}])
    class Condition:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def wait_for(self,*args,**kwargs):state.update(next(frames))
    def request(*args,**kwargs):
        # Exercise actual JSON encoding, then end the infinite uploader after recovery.
        json.dumps(kwargs['json']);sent.append(kwargs['json'])
        if len(sent)==2:raise Finished()
    scope={'condition':Condition(),'state':state,'time':SimpleNamespace(time=lambda:100),'SOURCE':'unoq','policy':{'settings':{}},'transport':SimpleNamespace(request=request),'requests':requests,'base64':base64}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'board/main.py','exec'),scope)
    with pytest.raises(Finished):scope['push_live']()
    assert sent[0]['image_b64'] is None
    assert sent[1]['image_b64']==base64.b64encode(b'fresh').decode()
    assert all('jpeg' not in p['status'] for p in sent)
