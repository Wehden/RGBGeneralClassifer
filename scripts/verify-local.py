"""Observe a running local service without creating any Bedrock requests."""
import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--viewers',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True,help='Private JSON evidence path outside the repository')
    args=parser.parse_args()
    if args.seconds<=0 or not 0<=args.viewers<=8:parser.error('Use positive duration and 0–8 viewers')
    stop=threading.Event()
    viewer_results=[]
    def viewer(index):
        channel=['color','depth','ir'][index%3]
        count=0
        try:
            with requests.get(args.url+f'/api/camera/{channel}/stream',stream=True,timeout=(5,10)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(16384):
                    count+=len(chunk)
                    if stop.is_set():break
            viewer_results.append({'channel':channel,'bytes':count})
        except Exception as exc:viewer_results.append({'channel':channel,'error':str(exc)})
    samples=[];latencies=[]
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1,args.viewers)) as pool:
        tasks=[pool.submit(viewer,n) for n in range(args.viewers)]
        try:
            while time.monotonic()-started<args.seconds:
                begin=time.monotonic()
                health=requests.get(args.url+'/healthz',timeout=5)
                latencies.append((time.monotonic()-begin)*1000)
                health.raise_for_status()
                sample=requests.get(args.url+'/api/camera/status',timeout=5).json()
                sample['elapsed_s']=time.monotonic()-started
                samples.append(sample)
                if stop.wait(1):break
        finally:stop.set()
        for task in tasks:task.result()
    ordered=sorted(latencies)
    summary={'elapsed_s':time.monotonic()-started,'samples':len(samples),'viewers':viewer_results,
             'health_p50_ms':ordered[math.ceil(len(ordered)*.5)-1],
             'health_p95_ms':ordered[math.ceil(len(ordered)*.95)-1],
             'initial':samples[0],'final':samples[-1],
             'max_cache_bytes':max(s['cache_bytes'] for s in samples),
             'max_cached_frames':max(s['frames_in_cache'] for s in samples),
             'all_three_channels':all(set(s['channels'])=={'color','depth','ir'} for s in samples),
             'samples_detail':samples}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in {'initial','final','samples_detail'}},indent=2))


if __name__=='__main__':main()
