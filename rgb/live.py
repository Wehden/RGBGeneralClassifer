"""Live relay in process memory. Frames never enter Redis, disk, or an outbox."""
import base64
import binascii
import io
import json
import math
import time
import threading
from PIL import Image

FRESHNESS = 15


def validate(payload, source):
    if not isinstance(payload, dict) or set(payload) != {'status','image_b64','frame_age_s'}:
        raise ValueError('Invalid live payload')
    age = payload['frame_age_s']
    if type(age) not in (int,float) or not math.isfinite(age) or age < 0:
        raise ValueError('Invalid frame age')
    try:json.dumps(payload,allow_nan=False)
    except (ValueError,TypeError):raise ValueError('Invalid frame values')
    status = payload['status']
    if not isinstance(status, dict) or status.get('source_id') != source:
        raise ValueError('Invalid source')
    captured = status.get('last_frame')
    if type(captured) not in (int,float) or not math.isfinite(captured) or captured > time.time()+30:
        raise ValueError('Invalid frame timestamp')
    encoded = payload['image_b64']
    if encoded is not None:
        if not isinstance(encoded,str) or len(encoded)>1_500_000:
            raise ValueError('Invalid frame')
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as img:
                if img.format!='JPEG' or img.width*img.height>2_000_000:
                    raise ValueError('Invalid JPEG')
                img.verify()
        except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Invalid JPEG') from exc
    # Upload time and retries cannot refresh the original frame's capture time.
    return {'status':status,'image_b64':encoded,'received_at':time.time(), 'expires_at':min(captured+FRESHNESS,time.time()+max(0,FRESHNESS-age))}


class Cache:
    def __init__(self):
        self.frames={}
        self.lock=threading.Lock()

    def put(self,source,frame):
        with self.lock:self.frames[source]=frame

    def read(self,source):
        with self.lock:frame=self.frames.get(source)
        if not frame or not frame['image_b64'] or time.time()>=frame['expires_at']:
            with self.lock:
                if self.frames.get(source) is frame:self.frames.pop(source,None)
            return None
        return frame


def response(cache, source, slots):
    from flask import Response
    if not slots.acquire(blocking=False):
        return 'Viewer limit reached',429
    released = False
    def close():
        nonlocal released
        if not released:
            released = True
            slots.release()
    try:
        first = cache.read(source)
        if first is None:
            close()
            return 'Camera unavailable',503
    except Exception:
        close()
        raise
    def frames():
        previous = None
        current = cache.read(source)
        try:
            while current is not None:
                marker = current['status']['last_frame']
                if marker != previous:
                    jpeg = base64.b64decode(current['image_b64'])
                    yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'
                    previous = marker
                time.sleep(.2)
                current = cache.read(source)
        finally:
            close()
    result = Response(frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    result.call_on_close(close)
    return result
