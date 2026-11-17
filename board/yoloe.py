"""Verified end-to-end YOLOE exports using ONNX Runtime or OpenVINO."""
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np


def prepare(frame,size):
    h,w=frame.shape[:2]
    gain=min(size/w,size/h)
    width,height=round(w*gain),round(h*gain)
    left,top=round((size-width)/2-.1),round((size-height)/2-.1)
    image=cv2.resize(frame,(width,height),interpolation=cv2.INTER_LINEAR)
    image=cv2.copyMakeBorder(image,top,size-height-top,left,size-width-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    return tensor,(gain,left,top)


def decode(output,labels,shape,transform,threshold,*,include_row_index=False):
    if output.ndim!=3 or output.shape[0]!=1 or output.shape[2] not in (6,38):
        raise ValueError('Expected YOLOE end-to-end [1,N,6 or 38] output')
    h,w=shape[:2];gain,left,top=transform
    found=[]
    for row_index,row in enumerate(output[0]):
        x1,y1,x2,y2,score,cls=row[:6]
        if not np.isfinite(row[:6]).all() or not 0<=score<=1 or score<threshold:continue
        if cls!=int(cls) or not 0<=int(cls)<len(labels):continue
        box=[float(np.clip((y1-top)/gain/h,0,1)),float(np.clip((x1-left)/gain/w,0,1)),float(np.clip((y2-top)/gain/h,0,1)),float(np.clip((x2-left)/gain/w,0,1))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        found.append({'class_id':int(cls),'label':labels[int(cls)],'confidence':round(float(score),4),'box':box})
        if include_row_index:found[-1]['row_index']=row_index
        if len(found)==100:break
    return found


def _openvino_gather_contract(graph):
    """Preserve YOLOE top-k gathers across OpenVINO's Slice elimination pass.

    OpenVINO 2026.2.1 can remove a channel Slice before GatherElements, then
    reject its own rewritten graph because the data/indices widths differ.
    An equivalent batch GatherND retains the channel coordinates explicitly.
    Apply only to the static rank-three, axis-one contract used by these exports.
    """
    import openvino as ov
    replacements=0
    for node in graph.get_ordered_ops():
        if node.get_type_name()!='GatherElements' or node.get_attributes()['axis']!=1:
            continue
        data,indices=node.input_value(0),node.input_value(1)
        if not data.get_partial_shape().is_static or not indices.get_partial_shape().is_static:
            continue
        shape=list(indices.get_shape())
        source_shape=list(data.get_shape())
        if (len(shape)!=3 or len(source_shape)!=3
                or shape[0]!=source_shape[0] or shape[2]!=source_shape[2]):
            continue
        rows=ov.opset13.unsqueeze(ov.opset13.convert(indices,'i64'),[-1])
        columns=np.broadcast_to(np.arange(shape[2],dtype=np.int64),shape).copy()[...,None]
        coordinates=ov.opset13.concat([rows,ov.opset13.constant(columns)],-1)
        replacement=ov.opset13.gather_nd(data,coordinates,batch_dims=1)
        replacement.set_friendly_name(node.get_friendly_name())
        replacement.output(0).set_names(node.output(0).get_names())
        node.output(0).replace(replacement.output(0))
        replacements+=1
    graph.validate_nodes_and_infer_types()
    return replacements


def _openvino_arm_upsample_contract(graph):
    """Avoid ARM ACL deconvolution drift for disjoint 2x2/stride-2 kernels.

    Each input pixel maps to its own 2x2 output tile: a channel matrix
    multiplication followed by interleaving is mathematically equivalent.
    Leave overlapping, padded, grouped, and dynamic operations untouched.
    """
    from openvino import opset13 as ops
    replacements=0
    for node in graph.get_ordered_ops():
        if node.get_type_name()!='ConvolutionBackpropData':continue
        if (not node.get_input_partial_shape(0).is_static or not node.get_input_partial_shape(1).is_static
            or not node.get_output_partial_shape(0).is_static):continue
        source=list(node.get_input_shape(0));weights=list(node.get_input_shape(1))
        attrs=node.get_attributes()
        if (len(source)!=4 or len(weights)!=4 or weights[2:]!=[2,2]
            or attrs['strides']!=[2,2] or attrs['dilations']!=[1,1]
            or attrs['pads_begin']!=[0,0] or attrs['pads_end']!=[0,0]
            or attrs['output_padding']!=[0,0]):continue
        n,c,h,w=source;out_channels=weights[1]
        if weights[0]!=c or list(node.get_output_shape(0))!=[n,out_channels,h*2,w*2]:continue
        pixels=ops.transpose(node.input_value(0),[0,2,3,1])
        kernel=ops.reshape(node.input_value(1),[c,out_channels*4],False)
        tiles=ops.matmul(pixels,kernel,False,False)
        tiles=ops.reshape(tiles,[n,h,w,out_channels,2,2],False)
        interleaved=ops.transpose(tiles,[0,3,1,4,2,5])
        replacement=ops.reshape(interleaved,[n,out_channels,h*2,w*2],False)
        replacement.set_friendly_name(node.get_friendly_name())
        replacement.output(0).set_names(node.output(0).get_names())
        node.output(0).replace(replacement.output(0))
        replacements+=1
    graph.validate_nodes_and_infer_types()
    return replacements


class Detector:
    def __init__(self,model,manifest,threshold=.45,backend='onnxruntime',device='CPU',threads=None):
        self.threshold=threshold
        self.manifest=json.loads(Path(manifest).read_text())
        self.sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest()
        if self.sha256!=self.manifest['sha256']:raise ValueError('YOLOE artifact checksum mismatch')
        self.labels=self.manifest['labels']
        if not isinstance(self.labels,list) or not self.labels or not all(isinstance(s,str) and 0<len(s)<=100 for s in self.labels):raise ValueError('Invalid YOLOE vocabulary')
        self.size=self.manifest['input_size']
        self.model=self.manifest['model']
        threads=threads if threads is not None else int(os.getenv('INFERENCE_THREADS','2'))
        if backend=='openvino':
            import openvino as ov
            core=ov.Core()
            graph=core.read_model(str(model))
            if list(graph.input(0).shape)!=[1,3,self.size,self.size] or graph.input(0).element_type!=ov.Type.f32:
                raise ValueError('YOLOE input contract mismatch')
            gather_replacements=_openvino_gather_contract(graph)
            upsample_replacements=0
            if device=='CPU' and platform.machine().lower() in {'aarch64','arm64'}:
                upsample_replacements=_openvino_arm_upsample_contract(graph)
            # Accuracy mode also disables reduced-precision internal optimizations on ARM.
            options={'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':1,'INFERENCE_PRECISION_HINT':'f32',
                     'EXECUTION_MODE_HINT':'ACCURACY'}
            if device=='CPU':options['INFERENCE_NUM_THREADS']=threads
            self.session=core.compile_model(graph,device,options)
            self.output_port=self.session.output(0)
            runtime_version=ov.__version__;provider=device
        elif backend=='onnxruntime':
            import onnxruntime as ort
            if device!='CPU':raise ValueError('ONNX adapter supports CPU only')
            options=ort.SessionOptions()
            options.intra_op_num_threads=threads
            options.inter_op_num_threads=1
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
            inp=self.session.get_inputs()[0]
            if inp.shape!=[1,3,self.size,self.size] or inp.type!='tensor(float)':raise ValueError('YOLOE input contract mismatch')
            self.input_name=inp.name
            runtime_version=ort.__version__;provider='CPUExecutionProvider'
        else:raise ValueError('Unsupported detector backend')
        self.backend=backend
        vocabulary=json.dumps(self.labels,separators=(',',':'),ensure_ascii=False).encode()
        self.config={**{k:v for k,v in self.manifest.items() if k!='labels'},'label_count':len(self.labels),'vocabulary_sha256':hashlib.sha256(vocabulary).hexdigest(),'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),'runtime':backend,'runtime_version':runtime_version,'provider':provider,'inference_threads':threads}
        if backend=='openvino':
            self.config.update(gather_contract_replacements=gather_replacements,
                               arm_upsample_contract_replacements=upsample_replacements,
                               execution_mode='ACCURACY',inference_precision_hint='f32')

    def detect(self,frame,*,with_masks=False):
        tensor,transform=prepare(frame,self.size)
        start=time.monotonic()
        if self.backend=='openvino':
            tensors=self.session([tensor])
            output=tensors[self.output_port]
            prototypes=tensors[self.session.output(1)] if with_masks and len(self.session.outputs)>1 else None
        else:
            tensors=self.session.run(None,{self.input_name:tensor})
            output=tensors[0]
            prototypes=tensors[1] if with_masks and len(tensors)>1 else None
        ms=(time.monotonic()-start)*1000
        self.proposals=decode(output,self.labels,frame.shape,transform,0)
        detections=[dict(p) for p in self.proposals if p['confidence']>=self.threshold]
        self.instance_masks=[None]*len(detections)
        self.mask_error=None
        if not with_masks or prototypes is None:
            return detections,round(ms,2)
        selected=decode(output,self.labels,frame.shape,transform,0,include_row_index=True)
        selected=[p for p in selected if p['confidence']>=self.threshold]
        from rgb.instance_masks import instances
        try:
            masks={d['row_index']:d['mask'] for d in
                   instances(output,prototypes,self.labels,frame.shape,transform,0,
                             row_indices={p['row_index'] for p in selected})}
            self.instance_masks=[masks.get(p['row_index']) for p in selected]
        except ValueError as exc:
            # A bad mask head must not hide otherwise valid object boxes.
            self.mask_error=str(exc)
        return detections,round(ms,2)
