import hashlib,json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from board.yoloe import prepare,decode,Detector


def test_letterbox_restores_wide_image_coordinates_and_rgb_channels():
    frame=np.zeros((480,640,3),dtype=np.uint8);frame[:]=[10,20,30]
    tensor,transform=prepare(frame,320)
    assert tensor.shape==(1,3,320,320)
    assert tensor.dtype==np.float32
    assert tensor[0,0,100,100]==pytest.approx(30/255)
    assert tensor[0,2,100,100]==pytest.approx(10/255)
    assert tensor[0,0,0,0]==pytest.approx(114/255)
    out=np.array([[[0,40,320,280,.9,0],[5,0,30,20,.8,0]]],dtype=np.float32)
    detections=decode(out,['fan'],frame.shape,transform,.5)
    assert len(detections)==1
    assert detections[0]['box']==pytest.approx([0,0,1,1])
    assert detections[0]['label']=='fan'


def test_letterbox_handles_portrait_odd_padding_and_clipped_boxes():
    frame=np.zeros((641,479,3),np.uint8)
    tensor,transform=prepare(frame,320)
    assert tensor.shape==(1,3,320,320)
    out=np.array([[[-10,-10,330,330,.7,0]]],np.float32)
    assert decode(out,['object'],frame.shape,transform,.5)[0]['box']==[0,0,1,1]


def test_yoloe_drops_invalid_rows_and_respects_threshold_and_mask_columns():
    out=np.zeros((1,6,38),np.float32)
    out[0,:,:6]=[[0,0,100,100,.8,1],[0,0,100,100,.3,0],[0,0,100,100,.9,1.5],[0,0,100,100,float('nan'),0],[0,0,100,100,.9,5],[30,30,10,10,.9,0]]
    detections=decode(out,['cup','fan'],(320,320,3),(1,0,0),.39)
    assert len(detections)==1
    assert detections[0]['label']=='fan'
    with pytest.raises(ValueError):decode(np.zeros((1,84,2100)),['fan'],(320,320,3),(1,0,0),.39)


def test_pinned_yoloe_model_and_vocabulary():
    root=Path('models/yoloe-26n-pf-320')
    manifest=json.loads((root/'manifest.json').read_text())
    assert hashlib.sha256((root/'model.onnx').read_bytes()).hexdigest()==manifest['sha256']
    assert len(manifest['labels'])==4585
    assert manifest['prompt_mode']=='prompt_free'
    assert manifest['end2end'] is True
    assert manifest['agnostic_nms'] is True
    assert manifest['output_contract']==[[1,100,38],[1,32,80,80]]


def test_detector_config_preserved_in_event_and_typed_downloads(client,event):
    import pyarrow as pa
    import pyarrow.parquet as pq
    c,store=client
    config={'runtime':'onnxruntime','input_size':320,'label_count':4585,'vocabulary_sha256':'b'*64}
    event['detector']['config']=config
    assert c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40}).status_code==201
    assert store.raw(event['event_id'])['detector']['config']==config
    record=c.get('/api/review/records').json['records'][0]
    assert json.loads(record['detector_config_json'])==config
    assert record['detector_sha256']==event['detector']['sha256']
    assert record['initial_class_id']==44
    for fmt in ['parquet','arrow']:
        r=c.get('/api/review/records?format='+fmt)
        buf=pa.BufferReader(r.data)
        table=pq.read_table(buf) if fmt=='parquet' else pa.ipc.open_file(buf).read_all()
        assert json.loads(table.to_pylist()[0]['detector_config_json'])==config


@pytest.mark.parametrize('config',[[],{'bad':float('nan')},{'bad':'x'*65536}])
def test_rejects_invalid_or_unbounded_detector_config(event,sources,config):
    from rgb.events import validate_event,InvalidEvent
    event['detector']['config']=config
    with pytest.raises(InvalidEvent):validate_event(event,sources)


from test_web import client
from test_events import event,sources
