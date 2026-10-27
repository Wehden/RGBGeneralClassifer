#!/usr/bin/env python3
"""Measure paired local detector latency; retain scene readings privately."""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from board.yoloe import Detector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',nargs='+',type=Path,required=True,help='Private camera NPZ bundles')
    p.add_argument('--models-dir',type=Path,default=Path.home()/'.cache/bolt-rgb/models')
    p.add_argument('--output',type=Path)
    p.add_argument('--worker',nargs=2,metavar=('DIRECTORY','BACKEND'))
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--samples',type=int,default=30)
    a=p.parse_args()
    if a.worker:
        directory=Path(a.worker[0])
        model=Detector(directory/'model.onnx',directory/'manifest.json',
                       backend=a.worker[1],threads=a.threads)
        frames=[]
        for path in a.frames:
            with np.load(path,allow_pickle=False) as f:
                frames.append(cv2.cvtColor(f['color_rgb'],cv2.COLOR_RGB2BGR))
        for i in range(5):model.detect(frames[i%len(frames)])
        timings=[];inference=[];readings=[]
        for i in range(a.samples):
            started=time.perf_counter()
            detections,ms=model.detect(frames[i%len(frames)])
            timings.append((time.perf_counter()-started)*1000);inference.append(ms)
            if i<len(frames):readings.append(detections)
        print(json.dumps({'config':model.config,'samples':a.samples,
                          'total_p50_ms':float(np.median(timings)),
                          'total_p95_ms':float(np.percentile(timings,95)),
                          'inference_p50_ms':float(np.median(inference)),
                          'inference_p95_ms':float(np.percentile(inference,95)),
                          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          'scene_readings':readings}))
        return
    root=Path(__file__).resolve().parents[1]
    cases=[(root/'models/yoloe-26n-pf-320','onnxruntime')]
    cases.extend((a.models_dir/name,backend) for name in ['yoloe-26s-pf-640','yoloe-26m-pf-640']
                 for backend in ['onnxruntime','openvino'])
    results=[]
    for directory,backend in cases:
        command=[sys.executable,__file__,'--frames',*map(str,a.frames),'--worker',str(directory),backend,
                 '--threads',str(a.threads),'--samples',str(a.samples)]
        process=subprocess.run(command,capture_output=True,text=True)
        if process.returncode:
            result={'model':directory.name,'backend':backend,'state':'unavailable',
                    'error':process.stderr[-1600:]}
        else:
            result=json.loads(process.stdout)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='scene_readings' and k!='config'},
                         separators=(',',':')),directory.name,backend,flush=True)
    document={'frames':[{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in a.frames],
              'results':results,'quality_accuracy':'Not measured: frames have no ground-truth labels'}
    if a.output:a.output.write_text(json.dumps(document,indent=2)+'\n')


if __name__=='__main__':main()
