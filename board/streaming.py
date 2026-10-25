"""Shared MJPEG response with admission cleanup even before iteration."""
import threading
import time
from flask import Response


def response(condition,state,viewers):
    if not viewers.acquire(blocking=False):
        return 'Viewer limit reached',429
    released=False
    lock=threading.Lock()
    def release():
        nonlocal released
        with lock:
            if not released:
                released=True
                viewers.release()
    def frames():
        sequence=-1
        try:
            while True:
                with condition:
                    condition.wait_for(lambda:state['sequence']!=sequence,timeout=10)
                    sequence,jpeg=state['sequence'],state['jpeg']
                    fresh=time.time()-state['last_frame']<15
                if not fresh or jpeg is None:return
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
        finally:release()
    result=Response(frames(),mimetype='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})
    result.call_on_close(release)
    return result
