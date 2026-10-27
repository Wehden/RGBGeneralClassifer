"""Run the local RGB camera service: python -m rgb.local."""
import argparse
import logging
import math
import os
import signal
import threading
from pathlib import Path

from waitress import create_server
from rgb.camera import Camera, CameraBusy, CameraLock, CameraUnavailable
from rgb.local_runtime import LocalRuntime
from rgb.local_store import LocalStore
from rgb.web import create_app
from rgb.vlm_providers import MODELS, model_selection


def parser():
    p=argparse.ArgumentParser(description='Local RGB camera and visual review service')
    p.add_argument('--host',default=os.environ.get('RGB_HOST','127.0.0.1'))
    p.add_argument('--port',type=int,default=int(os.environ.get('RGB_PORT','8080')))
    p.add_argument('--state-dir',default=os.environ.get('RGB_STATE_DIR',str(Path.home()/'.local/share/bolt-rgb')))
    p.add_argument('--aws-profile',default=os.environ.get('RGB_AWS_PROFILE','default'))
    p.add_argument('--aws-region',default=os.environ.get('RGB_AWS_REGION','us-west-2'))
    p.add_argument('--camera-backend',choices=['orbbec','v4l2'],default=os.environ.get('RGB_CAMERA_BACKEND','orbbec'))
    p.add_argument('--camera-serial',default=os.environ.get('RGB_CAMERA_SERIAL'))
    p.add_argument('--camera-device',default=os.environ.get('RGB_CAMERA_DEVICE','/dev/video0'))
    p.add_argument('--provider',choices=list(MODELS),default=os.environ.get('RGB_VLM_PROVIDER'))
    p.add_argument('--model',default=os.environ.get('RGB_MODEL'))
    paused=os.environ.get('RGB_VLM_PAUSED')
    if paused not in {None,'true','false'}:raise ValueError('RGB_VLM_PAUSED must be true or false')
    p.add_argument('--paused',action=argparse.BooleanOptionalAction,default=None if paused is None else paused=='true')
    p.add_argument('--detector-threshold',type=float,default=os.environ.get('RGB_DETECTOR_THRESHOLD'))
    p.add_argument('--vlm-threshold',type=float,default=os.environ.get('RGB_VLM_THRESHOLD'))
    p.add_argument('--display-mode',choices=['objects','materials'],default=os.environ.get('RGB_DISPLAY_MODE'))
    p.add_argument('--no-vlm',action='store_true')
    p.add_argument('--capture-fps',type=int,choices=[5,15,25,30],default=int(os.environ.get('RGB_CAPTURE_FPS','15')))
    p.add_argument('--detector-fps',type=float,default=float(os.environ.get('RGB_DETECTOR_FPS','2')))
    p.add_argument('--detector-model-dir',default=os.environ.get('RGB_DETECTOR_MODEL_DIR'),
                   help='Directory containing verified model.onnx and manifest.json')
    p.add_argument('--detector-backend',choices=['onnxruntime','openvino'],
                   default=os.environ.get('RGB_DETECTOR_BACKEND','onnxruntime'))
    p.add_argument('--detector-device',choices=['CPU','GPU'],default=os.environ.get('RGB_DETECTOR_DEVICE','CPU'))
    p.add_argument('--detector-threads',type=int,default=int(os.environ.get('RGB_DETECTOR_THREADS','4')))
    p.add_argument('--materials-model-dir',default=os.environ.get('RGB_MATERIALS_MODEL_DIR'))
    p.add_argument('--materials-device',choices=['CPU','GPU'],default=os.environ.get('RGB_MATERIALS_DEVICE','CPU'))
    p.add_argument('--no-local-materials',action='store_true',help='Disable the local SigLIP2 crop worker')
    p.add_argument('--archive-interval',type=float,default=os.environ.get('RGB_ARCHIVE_INTERVAL'),
                   help='Seconds between automatic observations and their live VLM analysis (default: saved setting or 60)')
    p.add_argument('--vlm-min-interval',type=float,default=float(os.environ.get('RGB_VLM_MIN_INTERVAL','0')))
    p.add_argument('--read-timeout',type=float,default=float(os.environ.get('RGB_READ_TIMEOUT','15')))
    p.add_argument('--connect-timeout',type=float,default=float(os.environ.get('RGB_CONNECT_TIMEOUT','5')))
    p.add_argument('--max-tokens',type=int,default=int(os.environ.get('RGB_MAX_TOKENS','1600')))
    p.add_argument('--max-skew-ms',type=float,default=float(os.environ.get('RGB_MAX_SKEW_MS','10')))
    p.add_argument('--freshness',type=float,default=float(os.environ.get('RGB_FRESHNESS_S','2')))
    return p


def provider_settings(args,settings):
    provider=args.provider or settings.get('selected_provider','bedrock')
    if provider not in MODELS:raise ValueError('Unsupported VLM provider')
    model=args.model or (next(iter(MODELS[provider])) if args.provider and provider!=settings.get('selected_provider','bedrock')
                        else settings['selected_model'])
    model_selection(provider,model)
    return {'selected_provider':provider,'selected_model':model,'model_selection_reason':'cli_override'} if args.provider or args.model else {}


def main():
    args=parser().parse_args()
    timings=[args.detector_fps,args.read_timeout,args.connect_timeout,args.freshness,args.vlm_min_interval]
    if not 1<=args.port<=65535 or not all(math.isfinite(v) for v in timings) or min(timings[:-1])<=0 or args.vlm_min_interval<0 or args.max_tokens<1:
        raise SystemExit('Invalid port, timing, or token limit')
    if args.archive_interval is not None and (not math.isfinite(args.archive_interval) or not 1<=args.archive_interval<=86400):
        raise SystemExit('Capture interval must be between 1 and 86400 seconds')
    if any(value is not None and (not math.isfinite(value) or not .05<=value<=.99) for value in [args.detector_threshold,args.vlm_threshold]):
        raise SystemExit('Thresholds must be finite and between 0.05 and 0.99')
    if not math.isfinite(args.max_skew_ms) or args.max_skew_ms<0:raise SystemExit('Invalid channel skew tolerance')
    if not 1<=args.detector_threads<=64:raise SystemExit('Detector threads must be between 1 and 64')
    if args.detector_backend=='onnxruntime' and args.detector_device!='CPU':
        raise SystemExit('ONNX Runtime detector requires CPU')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # Claim the archive before recovery can modify interrupted-job records.
    camera=Camera(serial=args.camera_serial,backend=args.camera_backend,device=args.camera_device,
                  fps=args.capture_fps,skew_ms=args.max_skew_ms,freshness_s=args.freshness)
    try:
        state_lock=CameraLock('archive:'+str(Path(args.state_dir).expanduser().resolve()))
        camera.reserve()
    except (CameraBusy,CameraUnavailable) as exc:raise SystemExit(str(exc))
    store=LocalStore(args.state_dir,analysis_enabled=not args.no_vlm)
    updates={}
    try:updates.update(provider_settings(args,store.settings()))
    except ValueError as exc:raise SystemExit(str(exc))
    if args.paused is not None:updates['vlm_paused']=args.paused
    if args.archive_interval is not None:updates['capture_interval_s']=args.archive_interval
    for key in ['detector_threshold','vlm_threshold','display_mode']:
        if getattr(args,key) is not None:updates[key]=getattr(args,key)
    if updates:store.update_settings(updates)
    settings=store.settings()
    if settings['display_mode'] not in {'objects','materials'} or type(settings['vlm_paused']) is not bool:
        raise SystemExit('Invalid saved model, display mode or pause setting')
    if any(type(settings[key]) not in {int,float} or not math.isfinite(settings[key]) or not .05<=settings[key]<=.99 for key in ['detector_threshold','vlm_threshold']):
        raise SystemExit('Invalid saved confidence thresholds')
    if type(settings['capture_paused']) is not bool or type(settings['capture_interval_s']) not in {int,float} or not math.isfinite(settings['capture_interval_s']) or not 1<=settings['capture_interval_s']<=86400:
        raise SystemExit('Invalid saved capture controls')
    runtime=LocalRuntime(store,camera,profile=args.aws_profile,region=args.aws_region,detector_fps=args.detector_fps,
                         min_interval=args.vlm_min_interval,
                         read_timeout=args.read_timeout,connect_timeout=args.connect_timeout,max_tokens=args.max_tokens,
                         detector_dir=args.detector_model_dir,detector_backend=args.detector_backend,
                         detector_device=args.detector_device,detector_threads=args.detector_threads,
                         material_dir=args.materials_model_dir,material_device=args.materials_device,
                         local_materials=not args.no_local_materials)
    runtime.configuration.update(host=args.host,port=args.port,camera_backend=args.camera_backend,
                                 camera_serial=args.camera_serial,capture_fps=args.capture_fps,
                                 max_channel_skew_ms=camera.skew_ms,freshness_s=camera.freshness_s,
                                 max_cached_frames=camera.max_frames,max_cache_bytes=camera.max_bytes,
                                 shutdown_deadline_s=10)
    app=create_app(store,sources={'femto-bolt':{'local':True}},local_runtime=runtime)
    # Bind and initialize the archive first; camera and ML failures remain observable through the API.
    server=create_server(app,host=args.host,port=args.port,threads=16,channel_timeout=30)
    stopped=threading.Event()
    def stop(signum=None,frame=None):
        if stopped.is_set():return
        stopped.set()
        runtime.stop(10)
        server.close()
        from waitress import wasyncore
        wasyncore.close_all(map=server._map)
        server.task_dispatcher.shutdown(timeout=0)
        state_lock.close()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    runtime.start()
    def ownership_monitor():
        while not stopped.wait(.1):
            if camera.fatal_error:
                logging.error('%s',camera.fatal_error)
                stop()
                return
    threading.Thread(target=ownership_monitor,name='ownership-monitor',daemon=True).start()
    logging.info('Local RGB runner listening on %s:%s, VLM provider %s, model %s',args.host,args.port,
                 settings['selected_provider'],settings['selected_model'])
    try:server.run()
    finally:stop()
    if camera.fatal_error:raise SystemExit(1)


if __name__=='__main__':main()
