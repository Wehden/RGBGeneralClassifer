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
from board.detector import Detector, overlay
from board.refinements import match
from board.tracking import Tracker
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
if not 0.1 <= MAX_FPS <= 30:
    raise ValueError('MAX_FPS must be between 0.1 and 30')
cv2.setNumThreads(1)
MODEL = 'efficientdet-lite0-coco-uint8'
MODEL_SHA = hashlib.sha256(open('/app/models/efficientdet-lite0.tflite', 'rb').read()).hexdigest()
policy = {'settings': {'detector_threshold': .45, 'vlm_threshold': .65, 'display_mode': 'objects', 'version': 0}, 'refinements': None}
DB = os.getenv('OUTBOX_DB', '/data/outbox.sqlite3')


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
    detector = Detector('/app/models/efficientdet-lite0.tflite', '/app/models/labelmap.txt')
    tracker = Tracker()
    cap = None
    last_event = 0
    while True:
        try:
            frame_start = time.monotonic()
            if cap is None:
                cap, path = camera()
                state['camera'] = path
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            current_policy = policy
            detector.threshold = current_policy['settings']['detector_threshold']
            detections, ms = detector.detect(frame)
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
                state.update(jpeg=jpeg.tobytes(), sequence=state['sequence']+1, last_frame=now, inference_ms=ms, detections=detections, error=None)
                condition.notify_all()
            # Observation snapshots, not unique-item counts. Cap outbox at 500 rows.
            if detections and now-last_event >= 15:
                raw_ok, raw = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if raw_ok:
                    event = {'event_id': str(uuid.uuid4()), 'observed_at': datetime.now(timezone.utc).isoformat(), 'source_id': SOURCE, 'location': LOCATION or None, 'location_kind': 'configured' if LOCATION else 'unknown', 'detector': {'model': MODEL, 'sha256': MODEL_SHA, 'threshold': detector.threshold, 'settings_version': current_policy['settings']['version']}, 'detections': detections, 'image_b64': base64.b64encode(raw).decode()}
                    with database() as conn:
                        size = conn.execute('SELECT COUNT(*) FROM outbox').fetchone()[0]
                        if size < 500:
                            conn.execute('INSERT INTO outbox VALUES (?, ?)', (event['event_id'], json.dumps(event)))
                        else:
                            log.error('Outbox full; new observation skipped')
                    last_event = now
            time.sleep(max(0, 1/MAX_FPS-(time.monotonic()-frame_start)))
        except Exception as exc:
            log.warning('Capture: %s', exc)
            state['error'] = str(exc)
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(2)


def poll_policy():
    global policy
    url = os.environ['ARCHIVE_URL'].rstrip('/')+'/api/board-policy/'+SOURCE
    while True:
        try:
            response = requests.get(url,headers={'Authorization':'Bearer '+os.environ['INGEST_TOKEN']},timeout=5)
            response.raise_for_status()
            received = response.json()
            for key in ['detector_threshold','vlm_threshold']:
                if not .05 <= float(received['settings'][key]) <= .99:
                    raise ValueError('Invalid threshold')
            policy = received
        except (requests.RequestException,KeyError,TypeError,ValueError):
            pass
        time.sleep(3)


def deliver():
    url = os.environ['ARCHIVE_URL'].rstrip('/') + '/api/ingest'
    token = os.environ['INGEST_TOKEN']
    while True:
        try:
            with database() as conn:
                rows = conn.execute('SELECT id,payload FROM outbox ORDER BY rowid LIMIT 1').fetchall()
            if not rows:
                time.sleep(1)
                continue
            event_id, payload = rows[0]
            result = requests.post(url, data=payload, headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'}, timeout=10)
            result.raise_for_status()
            with database() as conn:
                conn.execute('DELETE FROM outbox WHERE id=?', (event_id,))
        except Exception as exc:
            log.warning('Delivery retry: %s', type(exc).__name__)
            time.sleep(5)


@app.get('/healthz')
def health():
    with condition:
        result = {k:v for k,v in state.items() if k != 'jpeg'}
    result.update(source_id=SOURCE, model=MODEL, model_sha256=MODEL_SHA, settings=policy['settings'])
    return jsonify(result), 200 if time.time()-result['last_frame'] < 15 else 503


@app.get('/snapshot.jpg')
def snapshot():
    with condition:
        jpeg = state['jpeg'] if time.time()-state['last_frame'] < 15 else None
    return Response(jpeg, mimetype='image/jpeg', headers={'Cache-Control':'no-store'}) if jpeg else ('Camera unavailable', 503)


@app.get('/stream')
def stream():
    if not viewers.acquire(blocking=False):
        return 'Viewer limit reached', 429
    def frames():
        sequence = -1
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: state['sequence'] != sequence, timeout=10)
                    sequence, jpeg = state['sequence'], state['jpeg']
                    fresh = time.time()-state['last_frame'] < 15
                if not fresh or jpeg is None:
                    return
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg+b'\r\n'
        finally:
            viewers.release()
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control':'no-store'})


if __name__ == '__main__':
    database().close()
    threading.Thread(target=capture, daemon=True).start()
    threading.Thread(target=deliver, daemon=True).start()
    threading.Thread(target=poll_policy, daemon=True).start()
    serve(app, host='0.0.0.0', port=8080, threads=10, channel_timeout=20)
