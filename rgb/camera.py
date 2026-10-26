"""One camera owner, synchronized immutable frames, bounded shared preview cache."""
import fcntl
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from common.geometry import bounds
from rgb.local_store import encode
from rgb.store import now


class CameraUnavailable(Exception):
    pass


class FrameExpired(Exception):
    pass


class CameraBusy(Exception):
    pass


def v4l2_serial(device):
    node=Path('/sys/class/video4linux')/Path(device).resolve().name/'device'
    for parent in [node.resolve(),*node.resolve().parents]:
        if (parent/'serial').exists():return (parent/'serial').read_text().strip()
    raise CameraUnavailable('Cannot establish physical camera identity for singleton lock')


class CameraLock:
    def __init__(self,serial):
        runtime=Path(os.environ.get('XDG_RUNTIME_DIR',f'/tmp/bolt-rgb-{os.getuid()}'))/'bolt-rgb'
        runtime.mkdir(mode=0o700,parents=True,exist_ok=True)
        key=hashlib.sha256(serial.encode()).hexdigest()
        self.file=(runtime/(key+'.lock')).open('a+')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise CameraBusy(f'Camera {serial} is already owned by another runner')
        self.file.seek(0);self.file.truncate();self.file.write(str(os.getpid()));self.file.flush()

    def close(self):
        if not self.file.closed:
            fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close()


def jpg(array,quality=80):
    ok,data=cv2.imencode('.jpg',array,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:raise ValueError('Could not encode preview')
    return data.tobytes()


def png(array):
    ok,data=cv2.imencode('.png',array)
    if not ok:raise ValueError('Could not encode raw plane')
    return data.tobytes()


def depth_summary(depth,box,scale_mm,minimum=30):
    x1,y1,x2,y2=bounds(box,depth.shape[1],depth.shape[0])
    roi=depth[y1:y2,x1:x2]
    samples=roi[roi>0].astype(np.float64)*scale_mm/1000
    sufficient=samples.size>=minimum
    quantiles=np.percentile(samples,[10,50,90],method='linear').tolist() if sufficient else [None]*3
    return {'basis':'box_roi','status':'ok' if sufficient else 'insufficient',
            'valid_samples':int(samples.size),'roi_pixels':int(roi.size),
            'valid_fraction':float(samples.size/roi.size) if roi.size else 0,
            'p10_m':quantiles[0],'median_m':quantiles[1],'p90_m':quantiles[2],
            'bounds_xyxy':[x1,y1,x2,y2],'minimum_valid_samples':minimum}


@dataclass(frozen=True)
class Frame:
    metadata:dict
    color_rgb:np.ndarray
    depth_native:np.ndarray|None
    ir_native:np.ndarray|None
    depth_registered:np.ndarray|None
    previews:dict

    @property
    def id(self):return self.metadata['frame_id']

    @property
    def memory_bytes(self):
        return sum(a.nbytes for a in [self.color_rgb,self.depth_native,self.ir_native,self.depth_registered] if a is not None)+sum(map(len,self.previews.values()))

    def bundle(self):
        data=io.BytesIO()
        arrays={'color_rgb':self.color_rgb,'metadata_json_utf8':np.frombuffer(encode(self.metadata).encode(),np.uint8)}
        for name in ['depth_native','ir_native','depth_registered']:
            if getattr(self,name) is not None:arrays[name]=getattr(self,name)
        np.savez_compressed(data,**arrays)
        return data.getvalue()


def make_frame(color_bgr,depth,ir,registered,metadata,color_jpeg=None):
    color=cv2.cvtColor(color_bgr,cv2.COLOR_BGR2RGB)
    previews={'color':color_jpeg if color_jpeg is not None else jpg(color_bgr)}
    if depth is not None:
        display_depth=registered if registered is not None else depth
        scale=metadata.get('registered_depth_scale_mm',metadata['depth_scale_mm']) if registered is not None else metadata['depth_scale_mm']
        normalized=np.clip(display_depth.astype(np.float64)*scale/5000,0,1)
        preview=cv2.applyColorMap(np.rint(normalized*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        preview[display_depth==0]=0
        previews['depth']=jpg(preview)
        metadata['depth_preview']={'plane':'depth_registered' if registered is not None else 'depth_native',
                                   'range_m':[0,5],'colormap':'TURBO','invalid_value':0,'clipped':True}
    if ir is not None:
        lo,hi=np.percentile(ir,[1,99],method='linear')
        normalized=np.zeros(ir.shape,np.uint8) if hi<=lo else np.rint(np.clip((ir.astype(np.float64)-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        previews['ir']=jpg(normalized)
        metadata['ir_preview']={'plane':'ir_native','percentiles':[1,99],'range_raw':[float(lo),float(hi)],'colormap':'grayscale','constant_value':0}
    for array in [color,depth,ir,registered]:
        if array is not None:array.flags.writeable=False
    return Frame(metadata,color,depth,ir,registered,previews)


def _calibration(profiles):
    def attrs(obj,names):
        result={}
        for name in names:
            value=getattr(obj,name)
            result[name]=value if isinstance(value,(int,float,str)) else str(value)
        return result
    result={'schema_version':1,'registration':'Orbbec AlignFilter to COLOR_STREAM','planes':{}}
    for name,profile in profiles.items():
        result['planes'][name]={
            'profile':{'width':profile.get_width(),'height':profile.get_height(),'fps':profile.get_fps(),'format':str(profile.get_format())},
            'intrinsics':attrs(profile.get_intrinsic(),['width','height','fx','fy','cx','cy']),
            'distortion':attrs(profile.get_distortion(),['model','k1','k2','k3','k4','k5','k6','p1','p2'])}
    for name in ['depth','ir']:
        ex=profiles[name].get_extrinsic_to(profiles['color'])
        result['planes'][name]['extrinsics_to_color']={'rotation':np.asarray(ex.rot).reshape(-1).tolist(),
                                                     'translation_mm':np.asarray(ex.transform).reshape(-1).tolist()}
    return result


class Camera:
    def __init__(self,serial=None,backend='orbbec',device='/dev/video0',fps=15,skew_ms=10,freshness_s=2,
                 max_frames=8,max_bytes=128*1024*1024):
        self.serial=serial
        self.backend=backend
        self.device=device
        self.fps=fps
        self.skew_ms=skew_ms
        self.freshness_s=freshness_s
        self.max_frames=max_frames
        self.max_bytes=max_bytes
        self.frames=OrderedDict()
        self.condition=threading.Condition(threading.RLock())
        self.stop_event=threading.Event()
        self.thread=None
        self.lock=None
        self.state='starting'
        self.error=None
        self.counters={'received':0,'accepted':0,'incomplete':0,'skew_rejected':0,'reconnects':0}
        self.info={}
        self.session_id=None
        self.latest_detection=None
        self.detector_status={'state':'starting'}
        self.calibration=None
        self.calibration_hash=None
        self.started=time.monotonic()
        self.configuration={}
        self.fatal_error=None

    def start(self):
        self.reserve()
        self.thread=threading.Thread(target=self._run,name='camera',daemon=True)
        self.thread.start()

    def reserve(self):
        """Use passive USB identity discovery before opening the SDK/device."""
        if self.lock is not None:return
        serial=self.serial
        if self.backend=='v4l2':
            try:reported=v4l2_serial(self.device)
            except CameraUnavailable:
                if serial is None:
                    raise CameraUnavailable('Cannot identify V4L2 device; supply --camera-serial when its kernel node is unavailable')
                reported=None
            if reported and serial and reported!=serial:raise CameraUnavailable('V4L2 device does not match the selected camera serial')
            serial=reported or serial
        elif serial is None:
            candidates=[]
            for path in Path('/sys/bus/usb/devices').glob('*'):
                try:
                    if (path/'idVendor').read_text().strip().lower()=='2bc5' and (path/'idProduct').read_text().strip().lower()=='066b':
                        candidates.append((path/'serial').read_text().strip())
                except OSError:continue
            if len(candidates)>1:raise CameraUnavailable('Multiple Femto Bolts found; select --camera-serial')
            if len(candidates)==1:serial=candidates[0]
        if serial:
            self.lock=CameraLock(serial)
            self.serial=serial

    def stop(self):
        self.stop_event.set()
        with self.condition:self.condition.notify_all()
        if self.thread:self.thread.join(3)

    def get(self,frame_id=None):
        with self.condition:
            if frame_id:
                frame=self.frames.get(frame_id)
                if frame is None:raise FrameExpired('Frame no longer in the memory cache')
                return frame
            frame=next(reversed(self.frames.values())) if self.frames else None
            if frame is None or time.monotonic()-frame.metadata['received_monotonic_s']>self.freshness_s:
                raise CameraUnavailable('Camera has no fresh complete frame')
            return frame

    def wait(self,after=None,timeout=1):
        with self.condition:
            self.condition.wait_for(lambda:self.stop_event.is_set() or self.frames and next(reversed(self.frames))!=after,timeout)
            return self.get()

    def publish(self,frame):
        with self.condition:
            self.frames[frame.id]=frame
            while len(self.frames)>self.max_frames or sum(f.memory_bytes for f in self.frames.values())>self.max_bytes:
                self.frames.popitem(last=False)
            self.counters['accepted']+=1
            self.state='running';self.error=None
            self.condition.notify_all()

    def accept_timestamps(self,timestamps):
        if set(timestamps)!={'color','depth','ir'}:
            self.counters['incomplete']+=1
            return False
        if (max(timestamps.values())-min(timestamps.values()))/1000>self.skew_ms:
            self.counters['skew_rejected']+=1
            return False
        return True

    def status(self):
        with self.condition:
            latest=next(reversed(self.frames.values())) if self.frames else None
            age=time.monotonic()-latest.metadata['received_monotonic_s'] if latest else None
            return {'state':self.state if age is None or age<=self.freshness_s else 'stale',
                    'error':self.error,'backend':self.backend,'device':self.info,'session_id':self.session_id,
                    'frame_id':latest.id if latest else None,'frame_age_s':age,
                    'channels':list(latest.previews) if latest else [],
                    'profiles':latest.metadata['profiles'] if latest else None,
                    'frames_in_cache':len(self.frames),'cache_bytes':sum(f.memory_bytes for f in self.frames.values()),
                    'counters':dict(self.counters),'capture_fps':self.counters['accepted']/max(time.monotonic()-self.started,1),
                    'detector':dict(self.detector_status),'calibration_sha256':self.calibration_hash}

    def _run(self):
        delay=1
        while not self.stop_event.is_set():
            accepted_before=self.counters['accepted']
            try:
                if self.backend=='orbbec':self._orbbec()
                else:self._v4l2()
                delay=1
            except CameraBusy as exc:
                self.state='busy';self.error=str(exc)
                self.fatal_error=str(exc)
                break
            except Exception as exc:
                self.state='unavailable';self.error=f'{type(exc).__name__}: {str(exc)[:250]}'
                self.counters['reconnects']+=1
                if self.counters['accepted']>accepted_before:delay=1
                if self.stop_event.wait(delay):break
                delay=min(delay*2,30)
        if self.lock:self.lock.close()
        if self.stop_event.is_set():self.state='stopped'

    def _orbbec(self):
        self.reserve()
        import pyorbbecsdk as ob
        ctx=ob.Context();ctx.set_logger_level(ob.OBLogLevel.ERROR)
        devices=ctx.query_devices()
        found=[devices.get_device_by_index(i) for i in range(devices.get_count())]
        if self.serial:found=[d for d in found if d.get_device_info().get_serial_number()==self.serial]
        else:found=[d for d in found if 'Femto Bolt' in d.get_device_info().get_name()]
        if len(found)!=1:raise CameraUnavailable('Select --camera-serial when multiple devices are present; no matching unique device')
        device=found[0];info=device.get_device_info();serial=info.get_serial_number()
        if self.lock is None:self.lock=CameraLock(serial)
        self.serial=serial
        self.info={'name':info.get_name(),'serial':serial,'firmware':info.get_firmware_version()}
        pipeline=ob.Pipeline(device);config=ob.Config();profiles={}
        for name,sensor,w,h,fmt in [('color',ob.OBSensorType.COLOR_SENSOR,1280,720,ob.OBFormat.MJPG),
                                  ('depth',ob.OBSensorType.DEPTH_SENSOR,640,576,ob.OBFormat.Y16),
                                  ('ir',ob.OBSensorType.IR_SENSOR,640,576,ob.OBFormat.Y16)]:
            profile=pipeline.get_stream_profile_list(sensor).get_video_stream_profile(w,h,fmt,self.fps)
            config.enable_stream(profile);profiles[name]=profile
        self.calibration=_calibration(profiles)
        self.calibration_hash=hashlib.sha256(encode(self.calibration).encode()).hexdigest()
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync()
        align=ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self.session_id=str(uuid4());sequence=0;started=False;misses=0
        try:
            pipeline.start(config);started=True
            while not self.stop_event.is_set():
                frames=pipeline.wait_for_frames(1000)
                received_at=now();received_monotonic=time.monotonic()
                if frames is None:
                    self.counters['incomplete']+=1;misses+=1
                    if misses>=5:raise CameraUnavailable('No synchronized frames received')
                    continue
                self.counters['received']+=1
                cf,df,irf=frames.get_color_frame(),frames.get_depth_frame(),frames.get_ir_frame()
                if not all([cf,df,irf]):
                    self.counters['incomplete']+=1;continue
                timestamps={name:int(f.get_timestamp_us()) for name,f in [('color',cf),('depth',df),('ir',irf)]}
                system_timestamps={name:int(f.get_system_timestamp_us()) for name,f in [('color',cf),('depth',df),('ir',irf)]}
                indices={name:int(f.get_index()) for name,f in [('color',cf),('depth',df),('ir',irf)]}
                skew=(max(timestamps.values())-min(timestamps.values()))/1000
                if not self.accept_timestamps(timestamps):continue
                depth=np.frombuffer(df.get_data(),np.uint16).reshape(df.get_height(),df.get_width()).copy()
                native_scale=float(df.get_depth_scale())
                ir=np.frombuffer(irf.get_data(),np.uint16).reshape(irf.get_height(),irf.get_width()).copy()
                color_jpeg=bytes(cf.get_data())
                color=cv2.imdecode(np.frombuffer(color_jpeg,np.uint8),cv2.IMREAD_COLOR)
                if color is None:raise CameraUnavailable('MJPEG decode failed')
                aligned=align.process(frames).as_frame_set().get_depth_frame()
                registered=np.frombuffer(aligned.get_data(),np.uint16).reshape(aligned.get_height(),aligned.get_width()).copy()
                if registered.shape!=color.shape[:2]:raise CameraUnavailable('Registered depth dimensions do not match color')
                scale=float(aligned.get_depth_scale())
                if any(not np.isfinite(s) or s<=0 for s in [scale,native_scale]):raise CameraUnavailable('Invalid depth scale')
                sequence+=1;misses=0
                metadata={'frame_id':str(uuid4()),'session_id':self.session_id,'source_id':'femto-bolt',
                          'sequence':sequence,'received_at':received_at,'received_monotonic_s':received_monotonic,
                          'timestamps_us':timestamps,'timestamp_domain':'device_reported','channel_skew_ms':skew,
                          'sdk_frame_indices':indices,'sdk_system_timestamps_us':system_timestamps,
                          'sdk_system_timestamp_domain':'sdk_reported',
                          'profiles':{name:value['profile'] for name,value in self.calibration['planes'].items()},
                          'depth_scale_mm':native_scale,'registered_depth_scale_mm':scale,
                          'depth_units':'raw_uint16 * plane_depth_scale_mm / 1000 = metres',
                          'calibration_sha256':self.calibration_hash,'calibration':self.calibration,'device':self.info}
                self.publish(make_frame(color,depth,ir,registered,metadata,color_jpeg=color_jpeg))
        finally:
            if started:pipeline.stop()

    def _v4l2(self):
        # Explicit color-only fallback. Resolve a video alias to its USB serial.
        resolved=Path(self.device).resolve()
        serial=v4l2_serial(self.device)
        if self.serial is not None and serial!=self.serial:
            raise CameraUnavailable('V4L2 alias now refers to another camera; select that camera explicitly')
        if self.lock is None:self.lock=CameraLock(serial)
        capture=cv2.VideoCapture(str(resolved),cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        capture.set(cv2.CAP_PROP_FPS,self.fps);capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        fourcc=int(capture.get(cv2.CAP_PROP_FOURCC))
        format_name=''.join(chr((fourcc>>(8*i))&255) for i in range(4))
        if format_name.strip() in {'Y16','Z16','Y12'}:
            capture.release()
            raise CameraUnavailable('Selected V4L2 node is a depth plane, not a color camera')
        self.session_id=str(uuid4());sequence=0;self.info={'serial':serial,'path':str(resolved)}
        try:
            while not self.stop_event.is_set():
                ok,color=capture.read()
                if not ok:raise CameraUnavailable('V4L2 capture failed')
                self.counters['received']+=1
                sequence+=1
                self.calibration={'schema_version':1,'available':False,'backend':'v4l2','serial':serial}
                self.calibration_hash=hashlib.sha256(encode(self.calibration).encode()).hexdigest()
                metadata={'frame_id':str(uuid4()),'session_id':self.session_id,'source_id':'femto-bolt','sequence':sequence,
                          'received_at':now(),'received_monotonic_s':time.monotonic(),'timestamps_us':{},'timestamp_domain':'unknown',
                          'profiles':{'color':{'width':color.shape[1],'height':color.shape[0],'fps':capture.get(cv2.CAP_PROP_FPS),'format':'BGR'}},
                          'calibration':self.calibration,'calibration_sha256':self.calibration_hash,'device':self.info}
                self.publish(make_frame(color,None,None,None,metadata))
        finally:capture.release()
