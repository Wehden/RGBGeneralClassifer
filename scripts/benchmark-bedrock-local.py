"""Paid, explicit replay of identical sensor/crop batches through both Nova models."""
import argparse
import io
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rgb.camera import Camera, CameraLock, make_frame
from rgb.local_runtime import LocalRuntime
from rgb.local_store import FALLBACK_MODEL, PRIMARY_MODEL, LocalStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--observations',type=int,default=10)
    parser.add_argument('--aws-profile',default='default')
    parser.add_argument('--aws-region',default='us-west-2')
    parser.add_argument('--state-dir',type=Path,required=True,help='Dedicated private benchmark archive; not the running service archive')
    parser.add_argument('--invoke-bedrock',action='store_true',help='Explicitly allow paid Converse requests')
    args=parser.parse_args()
    if not args.invoke_bedrock:parser.error('--invoke-bedrock is required; this benchmark incurs AWS charges')
    if args.observations<1:parser.error('Use at least one observation')
    if (args.state_dir.expanduser()/'archive.sqlite3').exists():parser.error('Use a new dedicated benchmark directory')
    ownership=CameraLock('archive:'+str(args.state_dir.expanduser().resolve()))
    store=LocalStore(args.state_dir)
    runtime=LocalRuntime(store,Camera(),profile=args.aws_profile,region=args.aws_region,read_timeout=30)
    events=[]
    for _ in range(args.observations):
        response=requests.get(args.url+'/api/camera/detections',timeout=5)
        if response.status_code!=200:raise RuntimeError('Wait for a complete detector result')
        result=response.json()
        response=requests.get(args.url+'/api/camera/frame',params={'frame_id':result['frame_id']},timeout=10)
        response.raise_for_status()
        with np.load(io.BytesIO(response.content),allow_pickle=False) as arrays:
            metadata=json.loads(arrays['metadata_json_utf8'].tobytes())
            frame=make_frame(cv2.cvtColor(arrays['color_rgb'],cv2.COLOR_RGB2BGR),
                             arrays['depth_native'].copy(),arrays['ir_native'].copy(),
                             arrays['depth_registered'].copy(),metadata)
        if not result['detections']:raise RuntimeError('Benchmark requires at least one detected object')
        events.append(runtime.archive(frame,result))
        time.sleep(.25)
    started_at=time.time()
    for model in [PRIMARY_MODEL,FALLBACK_MODEL]:
        store.update_settings({'selected_model':model,'model_selection_reason':'explicit_benchmark'})
        for event in events:
            for start in range(0,len(event['detections']),4):
                if not store.request_analysis(event['event_id']):raise RuntimeError('Benchmark job already active')
                job=store.job(event['event_id'])
                indices=list(range(start,min(start+4,len(event['detections']))))
                runtime._batch(job,indices)
                if store.analysis(event['event_id'])['status']=='queued':
                    store.set_job(job,'partial',error='Benchmark explicitly selected one batch')
    terminal=[json.loads(fields['value']) for _,fields in store.stream('rgb:model-attempts')]
    summary={'observations':len(events),'started_unix':started_at,'models':[]}
    for model in [PRIMARY_MODEL,FALLBACK_MODEL]:
        attempts=[a for a in terminal if a['model']==model and a['outcome']!='started']
        durations=sorted(a['latency_ms'] for a in attempts)
        summary['models'].append({'model':model,'outcomes':dict(Counter(a['outcome'] for a in attempts)),
                                  'requests':len(attempts),'request_p50_ms':durations[math.ceil(len(durations)*.5)-1],
                                  'request_p95_ms':durations[math.ceil(len(durations)*.95)-1],
                                  'input_tokens':sum(a.get('prompt_tokens') or 0 for a in attempts),
                                  'output_tokens':sum(a.get('output_tokens') or 0 for a in attempts)})
    (store.directory/'benchmark-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    ownership.close()


if __name__=='__main__':main()
