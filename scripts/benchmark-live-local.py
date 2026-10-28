#!/usr/bin/env python3
"""Exercise existing local workers and API in process, with a separate paused archive."""
import argparse
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np

from rgb.camera import Camera
from rgb.local_runtime import LocalRuntime
from rgb.local_store import LocalStore
from rgb.web import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--detector-dir',type=Path,required=True)
    parser.add_argument('--material-dir',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--detector-fps',type=float,default=1)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 3600 or not 0 < args.detector_fps <= 5:
        parser.error('Expected 10–3600 seconds and positive detector rate ≤5 Hz')
    normal_archive = (Path.home()/'.local/share/bolt-rgb').resolve()
    if args.output_dir.resolve() == normal_archive or normal_archive in args.output_dir.resolve().parents:
        parser.error('Use a separate benchmark directory outside the operator archive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    store = LocalStore(args.output_dir/'archive',analysis_enabled=False)
    store.update_settings({'capture_paused':True,'vlm_paused':True,'detector_threshold':.45})
    camera = Camera(fps=15)
    runtime = LocalRuntime(store,camera,detector_fps=args.detector_fps,
                           detector_dir=args.detector_dir,material_dir=args.material_dir)
    class ForbiddenProvider:
        def review(self,request):
            raise AssertionError('Benchmark must not dispatch a cloud request')
    runtime.providers = {'bedrock':ForbiddenProvider(),'openrouter':ForbiddenProvider()}
    app = create_app(store,sources={'femto-bolt':{'local':True}},local_runtime=runtime)
    health_times = []; cloud_times = []; detector_times = []; result_ages = []
    observed = set();material_responses = 0;responses = 0;errors = [];point_counts = []
    try:
        runtime.start()
        startup = time.monotonic()+30
        while time.monotonic() < startup:
            if (camera.state == 'running' and camera.latest_detection is not None
                    and runtime.material_status['state'] == 'running'):
                break
            if camera.fatal_error or runtime.material_status['state'] == 'unavailable':
                raise RuntimeError(str(camera.status())+' '+str(runtime.material_status))
            time.sleep(.1)
        else:
            raise RuntimeError('Live workers failed readiness: '+str(camera.status()))
        accepted_before = camera.counters['accepted']
        started = time.monotonic();deadline = started+args.seconds
        next_tick = started
        with app.test_client() as client:
            while time.monotonic() < deadline:
                before = time.perf_counter();response = client.get('/healthz')
                health_times.append((time.perf_counter()-before)*1000)
                if response.status_code != 200:errors.append(['health',response.status_code])
                response = client.get('/api/camera/detections')
                if response.status_code == 200:
                    result = response.get_json();responses += 1
                    material_responses += int(any('local_material' in d for d in result['detections']))
                    if result['frame_id'] not in observed:
                        observed.add(result['frame_id']);detector_times.append(result['detector_ms'])
                    with camera.condition:latest = camera.latest_detection
                    result_ages.append(time.monotonic()-latest[0].metadata['received_monotonic_s'])
                else:errors.append(['detections',response.status_code])
                before = time.perf_counter();response = client.get('/api/camera/pointcloud')
                cloud_times.append((time.perf_counter()-before)*1000)
                if response.status_code == 200:
                    length = struct.unpack('<I',response.data[:4])[0]
                    header = json.loads(response.data[4:4+length])
                    point_counts.append(header['point_count'])
                else:errors.append(['pointcloud',response.status_code])
                next_tick += .25
                time.sleep(max(0,next_tick-time.monotonic()))
        elapsed = time.monotonic()-started
        status = camera.status()
        count = store.connection().execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        document = {
            'detector':args.detector_dir.name,'material':args.material_dir.name,
            'duration_s':elapsed,'target_detector_hz':args.detector_fps,
            'capture_hz':(camera.counters['accepted']-accepted_before)/elapsed,
            'unique_detection_hz':len(observed)/elapsed,
            'detector_inference_p50_ms':float(np.median(detector_times)),
            'detector_inference_p95_ms':float(np.percentile(detector_times,95)),
            'detection_result_age_p95_s':float(np.percentile(result_ages,95)),
            'health_p95_ms':float(np.percentile(health_times,95)),
            'pointcloud_request_p95_ms':float(np.percentile(cloud_times,95)),
            'pointcloud_requests_hz':len(cloud_times)/elapsed,
            'median_point_count':float(np.median(point_counts)),
            'material_responses':material_responses,'detection_responses':responses,
            'observations':count,'errors':errors,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'camera_status':status,'material_status':runtime.material_status,
            'scope':'in-process API and pointcloud generation; no network transport or WebGL rendering; cloud disabled',
        }
        (args.output_dir/'results.json').write_text(json.dumps(document,indent=2)+'\n')
        print(json.dumps({key:value for key,value in document.items()
                          if key not in ['camera_status','material_status']}),flush=True)
        if errors or count:
            raise SystemExit(1)
    finally:
        runtime.stop()


if __name__ == '__main__':
    main()
