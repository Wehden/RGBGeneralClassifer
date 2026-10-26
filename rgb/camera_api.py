"""Direct camera API. Every response refers to one immutable frame set."""
import io
import json
import sqlite3
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import cv2
from flask import Response, jsonify, request, send_file
from rgb.camera import CameraUnavailable, FrameExpired, jpg, png
from rgb.local_store import ArchiveConflict, QueueFull
from common.geometry import bounds
from rgb.pointcloud import PointCloudBuilder


class CameraAPI:
    def __init__(self,app,runtime,export_slot):
        self.runtime=runtime
        self.camera=runtime.camera
        self.slots=threading.BoundedSemaphore(8)
        self.export_slot=export_slot
        self.active=0
        self.lock=threading.Lock()
        self.pointcloud_builder=PointCloudBuilder()
        self.pointcloud_slots=threading.BoundedSemaphore(2)
        app.register_error_handler(CameraUnavailable,lambda exc:(jsonify(error=str(exc)),503))
        app.register_error_handler(FrameExpired,lambda exc:(jsonify(error=str(exc)),410))
        app.register_error_handler(ArchiveConflict,lambda exc:(jsonify(error=str(exc)),409))
        app.register_error_handler(QueueFull,lambda exc:(jsonify(error=str(exc)),429))
        app.register_error_handler(sqlite3.Error,lambda exc:(jsonify(error='Archive storage unavailable'),503))
        app.register_error_handler(OSError,lambda exc:(jsonify(error='Archive storage unavailable'),503))
        app.add_url_rule('/api/camera/status','camera_status',self.status)
        app.add_url_rule('/api/camera/calibration','camera_calibration',self.calibration)
        app.add_url_rule('/api/camera/<channel>/stream','camera_stream',self.stream_response)
        app.add_url_rule('/api/camera/<channel>/snapshot.jpg','camera_snapshot',self.snapshot)
        app.add_url_rule('/api/camera/<channel>/raw.png','camera_raw',self.raw)
        app.add_url_rule('/api/camera/frame','camera_frame',self.bundle)
        app.add_url_rule('/api/camera/pointcloud','camera_pointcloud',self.pointcloud)
        app.add_url_rule('/api/camera/detections','camera_detections',self.detections)
        app.add_url_rule('/api/camera/observations','camera_observe',self.observe,methods=['POST'])
        app.add_url_rule('/api/events/<event_id>/assets/<name>','observation_asset',self.asset)
        app.add_url_rule('/api/review/evidence','evidence_export',self.evidence)

    def _frame(self,data=None):
        frame_id=data.get('frame_id') if data is not None else request.args.get('frame_id')
        if frame_id is not None:
            try:
                if not isinstance(frame_id,str):raise ValueError()
                frame_id=str(UUID(frame_id))
            except ValueError:return None,(jsonify(error='Expected UUID frame_id'),400)
        return self.camera.get(frame_id),None

    def _headers(self,response,frame,channel,plane=None):
        response.headers.update({'X-RGB-Frame-ID':frame.id,'X-RGB-Session-ID':frame.metadata['session_id'],
                                 'X-RGB-Channel':channel,'X-RGB-Received-At':frame.metadata['received_at'],
                                 'X-RGB-Calibration-SHA256':frame.metadata['calibration_sha256'],'Cache-Control':'no-store'})
        if 'depth_scale_mm' in frame.metadata:
            scale=frame.metadata.get('registered_depth_scale_mm',frame.metadata['depth_scale_mm']) if plane=='depth_registered' else frame.metadata['depth_scale_mm']
            response.headers.update({'X-RGB-Depth-Scale-MM':str(scale),'X-RGB-Depth-Units':'raw_uint16'})
        if plane:response.headers['X-RGB-Plane']=plane
        return response

    def status(self):
        status=self.camera.status()
        settings=self.runtime.store.settings()
        return jsonify(**status,stream_viewers=self.active,
                       views=status['channels']+(['pointcloud'] if self.camera.backend=='orbbec' and 'depth' in status['channels'] else []),
                       configuration={**self.runtime.configuration,'archive_interval_s':settings['capture_interval_s']})

    def calibration(self):
        frame,error=self._frame()
        if error:return error
        return self._headers(jsonify(frame_id=frame.id,calibration_sha256=frame.metadata['calibration_sha256'],
                                     calibration=frame.metadata['calibration']),frame,'calibration')

    def snapshot(self,channel):
        if channel not in {'color','depth','ir'}:return 'Not found',404
        if self.camera.backend=='v4l2' and channel!='color':return 'Channel unsupported',404
        frame,error=self._frame()
        if error:return error
        if channel not in frame.previews:return jsonify(error='Channel unsupported by this backend'),404
        return self._headers(Response(frame.previews[channel],mimetype='image/jpeg'),frame,channel,
                             frame.metadata.get('depth_preview',{}).get('plane') if channel=='depth' else channel+'_native' if channel=='ir' else 'color')

    def raw(self,channel):
        if channel not in {'depth','ir'}:return 'Not found',404
        if self.camera.backend=='v4l2':return 'Channel unsupported',404
        frame,error=self._frame()
        if error:return error
        array=getattr(frame,channel+'_native')
        if array is None:return jsonify(error='Channel unsupported by this backend'),404
        return self._headers(Response(png(array),mimetype='image/png'),frame,channel,channel+'_native')

    def bundle(self):
        frame,error=self._frame()
        if error:return error
        return self._headers(send_file(io.BytesIO(frame.bundle()),mimetype='application/octet-stream',
                                      as_attachment=True,download_name=frame.id+'.npz'),frame,'frame_set')

    def detections(self):
        frame,error=self._frame()
        if error:return error
        with self.camera.condition:latest=self.camera.latest_detection
        if request.args.get('frame_id') and (not latest or latest[0].id!=frame.id):
            return jsonify(status='pending',frame_id=frame.id,camera_frame_id=self.camera.status()['frame_id']),202
        if not latest:return jsonify(status='pending',frame_id=frame.id),202
        return jsonify(**latest[1],camera_frame_id=self.camera.status()['frame_id'],
                       camera_session_id=self.camera.session_id,status='complete')

    def stream_response(self,channel,overlay=False):
        if channel not in {'color','depth','ir'}:return 'Not found',404
        if self.camera.backend=='v4l2' and channel!='color':return 'Channel unsupported',404
        frame=self.camera.get()
        if channel not in frame.previews:return 'Channel unsupported',404
        if not self.slots.acquire(blocking=False):return jsonify(error='Viewer limit reached'),429
        with self.lock:self.active+=1
        released=False
        def release():
            nonlocal released
            with self.lock:
                if not released:
                    released=True;self.active-=1;self.slots.release()
        def generate():
            previous=None;sent=None
            try:
                while not self.camera.stop_event.is_set():
                    try:f=self.camera.wait(previous,1)
                    except CameraUnavailable:return
                    if previous==f.id:continue
                    previous=f.id
                    image=f.previews[channel]
                    if overlay:
                        with self.camera.condition:latest=self.camera.latest_detection
                        if (latest and latest[0].metadata['session_id']==f.metadata['session_id']
                                and latest[0].metadata['calibration_sha256']==f.metadata['calibration_sha256']
                                and latest[0].color_rgb.shape==f.color_rgb.shape
                                and time.monotonic()-latest[0].metadata['received_monotonic_s']<=self.camera.freshness_s):
                            f,result=latest
                            color=cv2.cvtColor(f.color_rgb,cv2.COLOR_RGB2BGR)
                            for d in result['detections']:
                                x1,y1,x2,y2=bounds(d['box'],color.shape[1],color.shape[0])
                                label=d.get('live_value',{}).get('display_label',d['label'])
                                cv2.rectangle(color,(x1,y1),(x2,y2),(50,220,90),2)
                                cv2.putText(color,label,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,220,90),1)
                            image=jpg(color)
                    if sent==f.id:continue
                    sent=f.id
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(image)).encode()+
                           b'\r\nX-RGB-Frame-ID: '+f.id.encode()+b'\r\nX-RGB-Session-ID: '+f.metadata['session_id'].encode()+
                           b'\r\nX-RGB-Received-At: '+f.metadata['received_at'].encode()+b'\r\n\r\n'+image+b'\r\n')
            finally:release()
        response=Response(generate(),mimetype='multipart/x-mixed-replace; boundary=frame')
        response.call_on_close(release)
        return response

    def observe(self):
        if not request.is_json:return jsonify(error='Expected JSON'),400
        if request.headers.get('Origin') and urlsplit(request.headers['Origin']).netloc!=request.host:return 'Origin mismatch',403
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)-{'frame_id'}:return jsonify(error='Expected optional frame_id'),400
        frame=None
        if data.get('frame_id') is not None:
            frame,error=self._frame(data)
            if error:return error
        try:event=self.runtime.archive(frame)
        except RuntimeError as exc:return jsonify(error=str(exc)),429
        return jsonify(observation_id=event['event_id'],event_id=event['event_id'],frame_id=event['frame_id']),201

    def pointcloud(self):
        if self.camera.backend=='v4l2':return 'Registered depth unsupported',404
        try:
            stride=int(request.args.get('stride','6'))
            max_range=float(request.args.get('max_range_m','5'))
            if not 2<=stride<=32 or not 0<max_range<=20:raise ValueError()
        except (ValueError,TypeError):
            return jsonify(error='Expected stride 2–32 and max_range_m greater than 0 and at most 20'),400
        frame,error=self._frame()
        if error:return error
        if not self.pointcloud_slots.acquire(blocking=False):
            return jsonify(error='Point cloud conversion busy; retry the latest frame'),429
        try:payload=self.pointcloud_builder.build(frame,stride,max_range)
        finally:self.pointcloud_slots.release()
        response=Response(payload,mimetype='application/octet-stream')
        response.headers['Content-Disposition']='attachment; filename="bolt-pointcloud.bin"'
        self._headers(response,frame,'pointcloud','depth_registered')
        response.headers['X-RGB-Depth-Units']='metres'
        response.headers['X-RGB-Coordinate-Frame']='color_camera'
        return response

    def asset(self,event_id,name):
        event=self.runtime.store.raw(event_id)
        if not event:return 'Not found',404
        assets={**event.get('sensor_assets',{}),'calibration':event.get('calibration_asset'),'metadata':event.get('metadata_asset')}
        asset=assets.get(name)
        if not asset:return 'Not found',404
        path=self.runtime.store.root/asset['file']
        if not path.exists():return jsonify(error='Archived evidence missing'),503
        return send_file(path,as_attachment=name in {'calibration','metadata'} or name.endswith('_native') or name=='depth_registered')

    def evidence(self):
        ids=request.args.get('event_ids','').split(',')
        if not 1<=len(ids)<=100:return jsonify(error='Select 1 to 100 observation IDs'),400
        try:ids=[str(UUID(key)) for key in ids]
        except ValueError:return jsonify(error='Expected event_ids UUID list'),400
        if not self.export_slot.acquire(blocking=False):return jsonify(error='Export already running'),429
        directory=tempfile.TemporaryDirectory(prefix='rgb-evidence-')
        def cleanup():directory.cleanup();self.export_slot.release()
        try:
            root=Path(directory.name);path=root/'evidence.zip';manifest={'schema_version':1,'observations':[],'assets':{}}
            with self.runtime.store.export_snapshot(['rgb:model-attempts']) as (store,stops,_):
                selected=set(ids);asset_files=set()
                for key in ids:
                    event=store.raw(key)
                    if event is None:cleanup();return 'Observation not found',404
                    manifest['observations'].append(event)
                    asset_files.add(event['image_sha256']+'.jpg')
                    for asset in [*event.get('sensor_assets',{}).values(),event.get('calibration_asset'),event.get('metadata_asset')]:
                        if asset:asset_files.add(asset['file'])
                manifest['attempts']=[]
                for _,fields in store.stream('rgb:model-attempts',stops['rgb:model-attempts']):
                    attempt=json.loads(fields['value'])
                    if attempt['scene_id'] in selected:
                        manifest['attempts'].append(attempt)
                        asset_files.update(item['asset']['file'] for item in attempt.get('inputs',[]))
                with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
                    for name in sorted(asset_files):
                        file=store.root/name
                        if not file.exists():raise OSError('Archived evidence missing')
                        archive.write(file,'assets/'+name);manifest['assets'][name]={'sha256':name.split('.')[0],'bytes':file.stat().st_size}
                    archive.writestr('manifest.json',json.dumps(manifest,indent=2,allow_nan=False))
            response=send_file(path,mimetype='application/zip',as_attachment=True,download_name='rgb-evidence.zip')
            response.direct_passthrough=False;response.call_on_close(cleanup)
            return response
        except Exception:
            cleanup();raise
