"""Bounded RGB-D point-cloud frames from immutable, registered camera arrays."""
import json
import math
import struct
import threading
from collections import OrderedDict

import cv2
import numpy as np

from rgb.camera import CameraUnavailable

POINT_LIMIT=30000
POINT_DTYPE=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,)),('padding','u1')])


class PointCloudBuilder:
    def __init__(self):
        self.lock=threading.Lock()
        self.cache=OrderedDict()
        self.ray_key=None
        self.rays=None

    def build(self,frame,stride=6,max_range=5):
        key=(frame.id,stride,max_range)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            payload=self._build(frame,stride,max_range)
            self.cache[key]=payload
            while len(self.cache)>2:self.cache.popitem(last=False)
            return payload

    def _build(self,frame,stride,max_range):
        depth=frame.depth_registered
        if depth is None:raise CameraUnavailable('Registered depth is required for the 3D view')
        height,width=depth.shape
        if frame.color_rgb.shape[:2]!=(height,width):
            raise CameraUnavailable('Registered depth and color dimensions do not match')
        try:
            plane=frame.metadata['calibration']['planes']['color']
            intrinsic=plane['intrinsics']
            fx,fy,cx,cy=[float(intrinsic[name]) for name in ['fx','fy','cx','cy']]
            if (not all(math.isfinite(v) for v in [fx,fy,cx,cy]) or min(fx,fy)<=0
                    or (intrinsic['width'],intrinsic['height'])!=(width,height)):
                raise ValueError()
            distortion=plane.get('distortion',{'model':'NONE'})
            model=str(distortion.get('model','NONE')).upper()
            if 'BROWN_CONRADY' not in model and model not in {'NONE','OBCAMERADISTORTIONMODEL.NONE'}:
                raise ValueError()
            coefficients=np.array([float(distortion.get(name,0)) for name in
                                   ['k1','k2','p1','p2','k3','k4','k5','k6']],np.float64)
            scale=float(frame.metadata.get('registered_depth_scale_mm',frame.metadata['depth_scale_mm']))/1000
            if not np.isfinite(coefficients).all() or not math.isfinite(scale) or scale<=0:raise ValueError()
        except (KeyError,TypeError,ValueError):
            raise CameraUnavailable('Valid color calibration is required for the 3D view') from None
        stride=max(stride,math.ceil(math.sqrt(width*height/POINT_LIMIT)))
        while math.ceil(width/stride)*math.ceil(height/stride)>POINT_LIMIT:stride+=1
        ray_key=(frame.metadata['calibration_sha256'],width,height,stride)
        yy,xx=np.mgrid[0:height:stride,0:width:stride]
        if ray_key!=self.ray_key:
            pixels=np.stack([xx,yy],axis=-1).reshape(-1,1,2).astype(np.float32)
            matrix=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64)
            self.rays=cv2.undistortPoints(pixels,matrix,coefficients).reshape(-1,2)
            self.ray_key=ray_key
        z=depth[yy,xx].reshape(-1).astype(np.float32)*scale
        valid=(z>0)&(z<=max_range)&np.isfinite(self.rays).all(axis=1)
        z=z[valid];rays=self.rays[valid]
        points=np.zeros(len(z),POINT_DTYPE)
        points['xyz'][:,0]=rays[:,0]*z
        points['xyz'][:,1]=rays[:,1]*z
        points['xyz'][:,2]=z
        points['rgb']=frame.color_rgb[yy,xx].reshape(-1,3)[valid]
        header={'schema_version':1,'encoding':'xyz-f32le-rgb-u8-pad1','point_stride_bytes':16,
                'point_count':len(points),'frame_id':frame.id,'session_id':frame.metadata['session_id'],
                'received_at':frame.metadata['received_at'],'calibration_sha256':frame.metadata['calibration_sha256'],
                'coordinate_frame':'color_camera','units':'metres','axes':['right','down','forward'],
                'depth_plane':'depth_registered','color_plane':'color_rgb','sample_stride':stride,
                'max_range_m':max_range,'point_limit':POINT_LIMIT,
                'median_z_m':float(np.median(z)) if len(z) else None,
                'projection':'Brown-Conrady undistorted color rays times registered optical depth'}
        encoded=json.dumps(header,separators=(',',':'),allow_nan=False).encode()
        return struct.pack('<I',len(encoded))+encoded+points.tobytes()
