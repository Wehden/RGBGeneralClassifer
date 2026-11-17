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


def test_tracking_and_display_do_not_mutate_raw_proposals(client,event):
    from board.transport import observation_payload
    import uuid
    model=object.__new__(Detector)
    model.backend='onnxruntime'
    model.size=320;model.labels=['fan'];model.threshold=.4;model.input_name='images'
    model.session=Mock()
    model.session.run.return_value=[np.array([[[0,0,100,100,.8,0]]],dtype=np.float32)]
    detections,_=model.detect(np.zeros((320,320,3),dtype=np.uint8))
    detections[0].update(track_id=str(uuid.uuid4()),display_label='fan',label_kind='object')
    assert set(model.proposals[0])=={'class_id','label','confidence','box'}
    event.update(detections=detections,detector_proposals=model.proposals)
    c,store=client
    assert c.post('/api/ingest',json=event,headers={'Authorization':'Bearer '+'x'*40}).status_code==201
    assert len(store.raw(event['event_id'])['detector_proposals'])==1
    # Repair the formerly shared payload shape without changing the input row.
    event['detector_proposals']=[dict(detections[0])]
    original=json.dumps(event)
    repaired=observation_payload(original)
    assert set(repaired['detector_proposals'][0])=={'class_id','label','confidence','box'}
    assert 'track_id' in json.loads(original)['detector_proposals'][0]


@pytest.mark.parametrize('backend',['onnxruntime','openvino'])
def test_instance_masks_remain_aligned_when_a_selected_mask_has_bad_coefficients(backend):
    model=object.__new__(Detector)
    model.backend=backend
    model.size=32;model.labels=['cup','chair'];model.threshold=.4;model.input_name='images'
    output=np.zeros((1,3,38),np.float32)
    output[0,:,:6]=[[0,0,16,16,.8,0],[16,16,32,32,.3,1],[16,16,32,32,.9,1]]
    output[0,0,6]=float('nan')
    output[0,2,6]=1.
    prototypes=np.ones((1,32,8,8),np.float32)
    model.session=Mock()
    if backend=='onnxruntime':
        model.session.run.return_value=[output,prototypes]
    else:
        model.output_port='boxes'
        model.session.outputs=['boxes','masks']
        model.session.output.return_value='masks'
        model.session.return_value={'boxes':output,'masks':prototypes}
    found,_=model.detect(np.zeros((32,32,3),np.uint8),with_masks=True)
    assert [d['label'] for d in found]==['cup','chair']
    assert model.instance_masks[0] is None
    assert model.instance_masks[1].dtype==np.bool_
    assert model.instance_masks[1][20,20] and not model.instance_masks[1][5,5]
    assert all(set(d)=={'class_id','label','confidence','box'} for d in found)
    assert all(set(d)=={'class_id','label','confidence','box'} for d in model.proposals)


@pytest.fixture
def arm_upsample_graph(monkeypatch):
    """Evaluate the rewrite's standard graph operations without a native SDK."""
    import sys
    from types import SimpleNamespace
    class Tensor:
        def __init__(self, value):self.value=np.asarray(value)
        def set_friendly_name(self, name):self.name=name
        def output(self, index):return self
        def set_names(self, names):self.names=names
    def value(item):return item.value if isinstance(item,Tensor) else item
    ops=SimpleNamespace(
        transpose=lambda data, axes:Tensor(np.transpose(value(data),axes)),
        reshape=lambda data, shape, special_zero:Tensor(np.reshape(value(data),shape)),
        matmul=lambda left, right, ta, tb:Tensor(np.matmul(value(left),value(right))))
    monkeypatch.setitem(sys.modules,'openvino',SimpleNamespace(opset13=ops))
    pixels=np.arange(12,dtype=np.float32).reshape(1,2,2,3)/3
    weights=np.arange(24,dtype=np.float32).reshape(2,3,2,2)/7
    node=Mock()
    node.get_type_name.return_value='ConvolutionBackpropData'
    node.get_input_partial_shape.return_value.is_static=True
    node.get_output_partial_shape.return_value.is_static=True
    node.get_input_shape.side_effect=lambda index:[pixels.shape,weights.shape][index]
    node.input_value.side_effect=lambda index:[pixels,weights][index]
    node.get_output_shape.return_value=[1,3,4,6]
    node.get_attributes.return_value={'strides':[2,2],'dilations':[1,1],
        'pads_begin':[0,0],'pads_end':[0,0],'output_padding':[0,0]}
    graph=Mock();graph.get_ordered_ops.return_value=[node]
    return graph,node,pixels,weights


def test_arm_disjoint_upsampling_matches_independent_scatter_reference(arm_upsample_graph):
    from board.yoloe import _openvino_arm_upsample_contract
    graph,node,pixels,weights=arm_upsample_graph
    expected=np.zeros((1,3,4,6),np.float32)
    # Direct transposed convolution: scatter each channel's weighted 2x2 tile.
    for channel in range(2):
        for y in range(2):
            for x in range(3):
                expected[0,:,2*y:2*y+2,2*x:2*x+2]+=pixels[0,channel,y,x]*weights[channel]
    assert _openvino_arm_upsample_contract(graph)==1
    replacement=node.output.return_value.replace.call_args.args[0]
    np.testing.assert_allclose(replacement.value,expected,rtol=1e-6,atol=1e-6)
    assert replacement.names==node.output.return_value.get_names.return_value
    graph.validate_nodes_and_infer_types.assert_called_once()


@pytest.mark.parametrize('unsupported',['overlap','padding','dilation','output_padding',
                                       'dynamic_input','dynamic_output','grouped','different_output'])
def test_arm_upsampling_preserves_unsupported_graphs(arm_upsample_graph,unsupported):
    from board.yoloe import _openvino_arm_upsample_contract
    graph,node,_,_=arm_upsample_graph
    attrs=node.get_attributes.return_value
    if unsupported=='overlap':attrs['strides']=[1,1]
    elif unsupported=='padding':attrs['pads_begin']=[1,1]
    elif unsupported=='dilation':attrs['dilations']=[2,2]
    elif unsupported=='output_padding':attrs['output_padding']=[1,1]
    elif unsupported=='dynamic_input':node.get_input_partial_shape.return_value.is_static=False
    elif unsupported=='dynamic_output':node.get_output_partial_shape.return_value.is_static=False
    elif unsupported=='grouped':node.get_type_name.return_value='GroupConvolutionBackpropData'
    elif unsupported=='different_output':node.get_output_shape.return_value=[1,3,8,12]
    assert _openvino_arm_upsample_contract(graph)==0
    node.output.return_value.replace.assert_not_called()
