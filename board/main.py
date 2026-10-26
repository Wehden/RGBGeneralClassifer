"""Single camera owner, shared JPEG cache and persistent event outbox."""
import base64
import glob
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import requests
from flask import Flask, Response, jsonify
from waitress import serve
from common.http import OPTIONS as HTTP_OPTIONS
from board.detector import Detector, overlay
from board.refinements import match
from board.tracking import Tracker
from board.transport import Transport
from board.streaming import response as stream_response
from common.display import value
from common.geometry import bounds, fingerprint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('rgb.board')
app = Flask(__name__)
condition = threading.Condition()
state = {'jpeg': None, 'sequence': 0, 'last_frame': 0, 'inference_ms': None, 'detections': [], 'camera': None, 'error': 'starting'}
viewers = threading.BoundedSemaphore(4)
SOURCE = os.getenv('SOURCE_ID', 'portentauno')
LOCATION = os.getenv('SOURCE_LOCATION', '')
MAX_FPS = float(os.getenv('MAX_FPS', '2'))
if MAX_FPS!=0 and not 0.1 <= MAX_FPS <= 30:
    raise ValueError('MAX_FPS must be zero (uncapped) or between 0.1 and 30')
cv2.setNumThreads(1)
BACKEND = os.getenv('DETECTOR_BACKEND','litert')
if BACKEND=='yoloe':
    from board.yoloe import Detector as YOLOEDetector
    detector = YOLOEDetector(os.environ['YOLOE_MODEL'],os.environ['YOLOE_MANIFEST'])
elif BACKEND=='litert':
    detector = Detector('/app/models/efficientdet-lite0.tflite','/app/models/labelmap.txt')
else:raise ValueError('Expected litert or yoloe detector backend')
MODEL = getattr(detector,'model','efficientdet-lite0-coco-uint8')
MODEL_SHA = getattr(detector,'sha256',None) or hashlib.sha256(open('/app/models/efficientdet-lite0.tflite','rb').read()).hexdigest()
DETECTOR_CONFIG = getattr(detector,'config',{'runtime':'litert','runtime_version':'2.1.6','input_size':320,'inference_threads':int(os.getenv('INFERENCE_THREADS','2'))})
DETECTOR_CONFIG={**DETECTOR_CONFIG,'capture_limit_fps':MAX_FPS or None}
detectors={'objects':detector}
if BACKEND=='yoloe':
    detectors['materials']=YOLOEDetector(os.getenv('MATERIALS_MODEL','/app/models/yoloe-26n-materials-320/model.onnx'),os.getenv('MATERIALS_MANIFEST','/app/models/yoloe-26n-materials-320/manifest.json'))


def identity(selected):
    return {'model':getattr(selected,'model',MODEL), 'model_sha256':getattr(selected,'sha256',MODEL_SHA),
            'detector_config':{**getattr(selected,'config',DETECTOR_CONFIG),'capture_limit_fps':MAX_FPS or None}}

policy = {'settings': {'detector_threshold': .45, 'vlm_threshold': .65, 'display_mode': 'objects', 'classification_running': False, 'record_observations': False, 'version': 0}, 'refinements': None}
DB = os.getenv('OUTBOX_DB', '/data/outbox.sqlite3')
transport = Transport(SOURCE)


def database():
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute('CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    return conn


def camera():
    for path in sorted(glob.glob('/dev/video*')):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read()
        if ok and frame is not None:
            log.info('Camera selected: %s', path)
            return cap, path
        cap.release()
    raise RuntimeError('No usable V4L2 capture device')


def capture():
    tracker = Tracker()
    cap = None
    last_event = 0
    last_mode = None
    while True:
        try:
            frame_start = time.monotonic()
            captured_at = datetime.now(timezone.utc).isoformat()
            if cap is None:
                cap, path = camera()
                state['camera'] = path
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            current_policy = policy
            mode=current_policy['settings'].get('display_mode','objects')
            selected=detectors.get(mode,detectors['objects'])
            if mode!=last_mode:
                tracker=Tracker()
                last_mode=mode
                last_event=0
            selected.threshold=current_policy['settings']['detector_threshold']
            running=current_policy['settings'].get('classification_running',True)
            recording=current_policy['settings'].get('record_observations',True)
            detections,ms=selected.detect(frame) if running else ([],None)
            provenance=identity(selected)
            for detection in detections:
                detection['label_kind']='material' if selected is detectors.get('materials') else 'object'
            tracker.update(detections, time.monotonic())
            for d in detections:
                reading = None
                if current_policy['refinements']:
                    x1,y1,x2,y2 = bounds(d['box'],frame.shape[1],frame.shape[0])
                    crop = frame[y1:y2,x1:x2]
                    digest = fingerprint(lambda x,y: crop[y,x][::-1],crop.shape[1],crop.shape[0])
                    reading = match(d,digest,current_policy['refinements'],time.time(),current_policy['settings']['vlm_threshold'])
                    if reading:
                        d.update(detection_id=reading['detection_id'],live_value={key:reading[key] for key in ['label','confidence','detection_id','reference_event_id','model','match_iou','match_hash_distance','match_age_s','match_method']})
                        d['live_value'].update(materials=reading.get('materials',[]),display_mode=current_policy['settings'].get('display_mode','objects'))
                displayed = value(d, reading, current_policy['settings'])
                d.update(display_label=displayed['label'],display_confidence=displayed['confidence'],label_provider=displayed['provider'])
                if 'live_value' in d:
                    d['live_value'].update(display_label=displayed['label'],display_confidence=displayed['confidence'])
            ok, jpeg = cv2.imencode('.jpg', overlay(frame, detections), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            now = time.time()
            with condition:
                state.update(jpeg=jpeg.tobytes(), sequence=state['sequence']+1, last_frame=now, inference_ms=ms, detections=detections, error=None, **provenance, classification_running=running, record_observations=recording, applied_settings_version=current_policy['settings']['version'], proposal_max=max((p['confidence'] for p in getattr(selected,'proposals',[]) ),default=None) if running else None)
                condition.notify_all()
            # Observation snapshots, not unique-item counts. Cap outbox at 500 rows.
            if recording and running and now-last_event >= 15:
                raw_ok, raw = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if raw_ok:
                    event = {'event_id': str(uuid.uuid4()), 'observed_at': captured_at, 'source_id': SOURCE, 'location': LOCATION or None, 'location_kind': 'configured' if LOCATION else 'unknown', 'detector': {'model': provenance['model'], 'sha256': provenance['model_sha256'], 'config': provenance['detector_config'], 'threshold': selected.threshold, 'settings_version': current_policy['settings']['version']}, 'detections': detections, 'detector_proposals':getattr(selected,'proposals',[]), 'image_b64': base64.b64encode(raw).decode()}
                    with database() as conn:
                        size = conn.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]
                        if size < 500:
                            conn.execute('INSERT INTO outbox VALUES (?, ?)', (event['event_id'], json.dumps(event)))
                        else:
                            log.error('Outbox full; new observation skipped')
                    last_event = now
            if MAX_FPS:time.sleep(max(0, 1/MAX_FPS-(time.monotonic()-frame_start)))
        except Exception as exc:
            log.warning('Capture: %s', exc)
            state['error'] = str(exc)
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(2)


def poll_policy():
    global policy
    while True:
        try:
            received = transport.request('GET', '/policy')
            for key in ['detector_threshold','vlm_threshold']:
                if not .05 <= float(received['settings'][key]) <= .99:
                    raise ValueError('Invalid threshold')
            policy = received
        except (requests.RequestException,KeyError,TypeError,ValueError):
            pass
        time.sleep(3)


def deliver():
    while True:
        try:
            with database() as conn:
                rows = conn.execute('SELECT id,payload FROM outbox ORDER BY rowid LIMIT 1').fetchall()
            if not rows:
                time.sleep(1)
                continue
            event_id, payload = rows[0]
            result = transport.request('POST', '/observations', json=json.loads(payload))
            if result.get('event_id') != event_id:
                raise ValueError('Observation acknowledgement mismatch')
            with database() as conn:
                conn.execute('DELETE FROM outbox WHERE id=?', (event_id,))
        except Exception as exc:
            log.warning('Delivery retry: %s', type(exc).__name__)
            time.sleep(5)


def push_live():
    # At most one pending frame: network outages never build a live-frame backlog.
    previous = None
    while True:
        with condition:
            condition.wait_for(lambda: state['sequence'] != previous, timeout=3)
            current = dict(state)
        previous = current['sequence']
        age = max(0, time.time()-current['last_frame'])
        jpeg = current.pop('jpeg') if age < 15 else None
        current.update(source_id=SOURCE, settings=policy['settings'])
        try:
            transport.request('POST', '/live', json={'status':current,
                'image_b64':base64.b64encode(jpeg).decode() if jpeg else None,
                'frame_age_s':age})
        except (requests.RequestException, ValueError):
            time.sleep(3)


@app.get('/healthz')
def health():
    with condition:
        result = {k:v for k,v in state.items() if k != 'jpeg'}
    result.update(source_id=SOURCE, settings=policy['settings'])
    return jsonify(result), 200 if time.time()-result['last_frame'] < 15 else 503


@app.get('/snapshot.jpg')
def snapshot():
    with condition:
        jpeg = state['jpeg'] if time.time()-state['last_frame'] < 15 else None
    return Response(jpeg, mimetype='image/jpeg', headers={'Cache-Control':'no-store'}) if jpeg else ('Camera unavailable', 503)


@app.get('/stream')
def stream():
    return stream_response(condition,state,viewers)


if __name__ == '__main__':
    database().close()
    threading.Thread(target=capture, daemon=True).start()
    threading.Thread(target=deliver, daemon=True).start()
    threading.Thread(target=poll_policy, daemon=True).start()
    threading.Thread(target=push_live, daemon=True).start()
    serve(app, host='0.0.0.0', port=8080, threads=10, channel_timeout=20, **HTTP_OPTIONS)
