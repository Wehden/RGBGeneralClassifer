"""Verified end-to-end YOLOE exports using ONNX Runtime or OpenVINO."""
import hashlib
import json
import os
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


def decode(output,labels,shape,transform,threshold):
    if output.ndim!=3 or output.shape[0]!=1 or output.shape[2] not in (6,38):
        raise ValueError('Expected YOLOE end-to-end [1,N,6 or 38] output')
    h,w=shape[:2];gain,left,top=transform
    found=[]
    for row in output[0]:
        x1,y1,x2,y2,score,cls=row[:6]
        if not np.isfinite(row[:6]).all() or not 0<=score<=1 or score<threshold:continue
        if cls!=int(cls) or not 0<=int(cls)<len(labels):continue
        box=[float(np.clip((y1-top)/gain/h,0,1)),float(np.clip((x1-left)/gain/w,0,1)),float(np.clip((y2-top)/gain/h,0,1)),float(np.clip((x2-left)/gain/w,0,1))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        found.append({'class_id':int(cls),'label':labels[int(cls)],'confidence':round(float(score),4),'box':box})
        if len(found)==100:break
    return found


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
            options={'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':1,'INFERENCE_PRECISION_HINT':'f32'}
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

    def detect(self,frame):
        tensor,transform=prepare(frame,self.size)
        start=time.monotonic()
        output=(self.session([tensor])[self.output_port] if self.backend=='openvino'
                else self.session.run(None,{self.input_name:tensor})[0])
        ms=(time.monotonic()-start)*1000
        self.proposals=decode(output,self.labels,frame.shape,transform,0)
        return [dict(p) for p in self.proposals if p['confidence']>=self.threshold],round(ms,2)
