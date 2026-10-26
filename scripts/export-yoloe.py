#!/usr/bin/env python3
"""Re-export the pinned prompt-free checkpoint; record the resulting artifact hash."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path('models/yoloe-26n-pf-320'))
p.add_argument('--checkpoint',type=Path,help='Existing official checkpoint; otherwise download it')
a=p.parse_args()
# Export dependencies belong in a separate CPU environment, not on the board.
import torch
import ultralytics
from ultralytics import YOLOE

if ultralytics.__version__!='8.4.188':raise RuntimeError('Expected ultralytics==8.4.188')
manifest=json.loads((a.output/'manifest.json').read_text())
with tempfile.TemporaryDirectory(prefix='rgb-yoloe-export-') as directory:
    checkpoint=Path(directory)/manifest['checkpoint']
    if a.checkpoint:shutil.copyfile(a.checkpoint,checkpoint)
    else:urllib.request.urlretrieve(manifest['checkpoint_url'],checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise RuntimeError('Source checkpoint checksum mismatch')
    torch.set_num_threads(2)
    model=YOLOE(str(checkpoint))
    artifact=Path(model.export(format='onnx',imgsz=320,nms=False,agnostic_nms=True,dynamic=False,opset=17,simplify=True,device='cpu',max_det=100))
    labels=[model.names[i] for i in range(len(model.names))]
    if labels!=manifest['labels']:raise RuntimeError('Export vocabulary differs from pinned manifest')
    shutil.copyfile(artifact,a.output/'model.onnx')
manifest['sha256']=hashlib.sha256((a.output/'model.onnx').read_bytes()).hexdigest()
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'model':manifest['model'],'sha256':manifest['sha256'],'labels':len(labels)}))
