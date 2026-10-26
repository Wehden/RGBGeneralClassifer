#!/usr/bin/env python3
"""Bake a material vocabulary into YOLOE; export off-board with pinned tools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('models/yoloe-26n-materials-320'))
a=p.parse_args()
import torch
import ultralytics
import onnx
import onnxslim
from ultralytics import YOLOE
if ultralytics.__version__!='8.4.188':raise RuntimeError('Expected ultralytics==8.4.188')
if (str(torch.__version__),onnx.__version__,onnxslim.__version__)!=('2.9.1+cpu','1.20.1','0.1.82'):raise RuntimeError('Export dependency version mismatch')
expected='1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b'
if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint mismatch')
labels=['plastic','paper','metal','film','liquid','glass','cardboard','wood','textile','rubber','ceramic','foam']
prompts=['plastic object','paper','metal object','plastic film','liquid','glass object','cardboard','wooden object','textile','rubber object','ceramic object','foam']
torch.set_num_threads(2)
m=YOLOE(str(a.checkpoint))
m.set_classes(prompts)
m.model.names=dict(enumerate(labels))
artifact=Path(m.export(format='onnx',imgsz=320,nms=False,agnostic_nms=True,dynamic=False,opset=17,simplify=True,device='cpu',max_det=100))
a.output.mkdir(parents=True,exist_ok=True)
shutil.copyfile(artifact,a.output/'model.onnx')
manifest={'torch':str(torch.__version__),'onnx':onnx.__version__,'onnxslim':onnxslim.__version__,'manifest_path':'models/yoloe-26n-materials-320/manifest.json','schema_version':1,'model':'yoloe-26n-materials-320','task':'materials','checkpoint':'yoloe-26n-seg.pt','checkpoint_url':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt','checkpoint_sha256':expected,'sha256':hashlib.sha256((a.output/'model.onnx').read_bytes()).hexdigest(),'input_size':320,'precision':'float32','prompt_mode':'text','prompts':prompts,'labels':labels,'exporter':'ultralytics==8.4.188','opset':17,'max_det':100,'agnostic_nms':True,'end2end':True,'output_contract':[[1,100,38],[1,32,80,80]],'license':'AGPL-3.0','text_encoder_sha256':hashlib.sha256(Path('mobileclip2_b.ts').read_bytes()).hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'model':manifest['model'],'sha256':manifest['sha256'],'labels':labels}))
