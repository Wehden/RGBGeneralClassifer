"""Executed inside the shipped service image, on a private Docker network."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
from uuid import UUID, uuid5

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from redis import Redis
import requests

from rgb.store import Store

ORIGIN = 'http://service:8080'
TOKEN = os.environ['INGEST_TOKEN']
EVENT_ID = os.environ['CI_EVENT_ID']


def call(path, *, method='GET', **kwargs):
    return requests.request(method, ORIGIN + path, timeout=15, **kwargs)


def ready():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if call('/healthz').status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(.2)
    raise AssertionError('Service did not become healthy')


def initial():
    ready()
    assert os.getuid() == 1000, 'Tests must use the deployment image user'
    for path in ['/', '/records', '/static/app.js', '/static/records.js']:
        response = call(path)
        assert response.status_code == 200, path
        assert response.headers['X-Content-Type-Options'] == 'nosniff'
        assert "default-src 'self'" in response.headers['Content-Security-Policy']
    image = io.BytesIO()
    Image.new('RGB', (48, 32), (30, 160, 80)).save(image, 'JPEG')
    event = {
        'event_id': EVENT_ID, 'source_id': 'synthetic-ci',
        'observed_at': datetime.now(timezone.utc).isoformat(), 'location': 'untrusted',
        'detector': {'model': 'synthetic-ci', 'sha256': 'a' * 64},
        'detections': [{'label': 'bottle', 'class_id': 44, 'confidence': .85,
                        'box': [.1, .2, .8, .9]}],
        'image_b64': base64.b64encode(image.getvalue()).decode(),
    }
    for headers in [{}, {'Authorization': 'Bearer incorrect'}]:
        assert call('/api/ingest', method='POST', json=event, headers=headers).status_code == 401
    headers = {'Authorization': 'Bearer ' + TOKEN}
    # Race the real Lua append/initialization scripts, not fakeredis.
    with ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda _: call('/api/ingest', method='POST', json=event, headers=headers),
                                range(16)))
    assert sum(response.status_code == 201 for response in replies) == 1
    assert all(response.status_code in {200, 201} for response in replies)
    event['detections'][0]['label'] = 'forged replay'
    assert call('/api/ingest', method='POST', json=event, headers=headers).status_code == 200
    saved = call('/api/events/' + EVENT_ID).json()
    assert saved['detections'][0]['label'] == 'bottle'
    assert saved['location'] == 'Synthetic CI bench'
    assert call('/api/events/' + EVENT_ID + '/image.jpg').content == image.getvalue()
    assert len(call('/api/events').json()) == 1
    redis = Redis.from_url(os.environ['REDIS_URL'], decode_responses=True)
    assert redis.xlen('rgb:events') == 1 and redis.xlen('rgb:detection-updates') == 1
    store = Store(redis, '/tmp/ci-contract-images', analysis_enabled=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        queued = list(pool.map(lambda _: store.request_analysis(EVENT_ID, [0]), range(16)))
    assert sum(queued) == 1 and redis.xlen('rgb:analysis:requests') == 1
    candidate = {'label': 'glass bottle', 'confidence': .9, 'comparison': 'refines',
                 'reason': 'synthetic fixture', 'materials': [
                     {'label': 'glass', 'confidence': .9, 'reason': 'synthetic fixture'}]}
    job = str(uuid5(UUID(EVENT_ID), 'enhancement'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        enriched = list(pool.map(lambda _: store.enhance(saved, 0, candidate, job), range(16)))
    assert all(item == enriched[0] for item in enriched)
    assert redis.xlen('rgb:detection-updates') == 2
    assert call('/api/settings', method='POST', json={'vlm_paused': True}).status_code == 400
    controls = {'X-RGB-Request': 'settings'}
    assert call('/api/settings', method='POST', json={'vlm_paused': True},
                headers={**controls, 'Origin': 'https://elsewhere.invalid'}).status_code == 403
    response = call('/api/settings', method='POST', json={'vlm_paused': True}, headers=controls)
    assert response.status_code == 200 and response.json()['vlm_paused'] is True
    records = call('/api/review/records').json()['records']
    assert len(records) == 1 and records[0]['observation_id'] == EVENT_ID
    for fmt in ['arrow', 'parquet']:
        response = call('/api/review/records?format=' + fmt)
        assert response.status_code == 200
        reader = pa.BufferReader(response.content)
        table = pq.read_table(reader) if fmt == 'parquet' else pa.ipc.open_file(reader).read_all()
        assert table.num_rows == 1
        assert table.column('observation_id').to_pylist() == [EVENT_ID]
    assert call('/api/review/records?format=invalid').status_code == 400
    redis.close()
    print('PASS container HTTP/auth, concurrent Redis Lua idempotence, controls and typed exports')


def recovered():
    ready()
    assert len(call('/api/events').json()) == 1
    saved = call('/api/events/' + EVENT_ID).json()
    assert saved['detections'][0]['label'] == 'bottle'
    assert saved['detections'][0]['enhancement']['label'] == 'glass bottle'
    assert call('/api/events/' + EVENT_ID + '/image.jpg').content.startswith(b'\xff\xd8')
    assert call('/api/settings').json()['vlm_paused'] is True
    print('PASS container and Redis restart preserve observations, images, enhancements and settings')


def outage():
    response = call('/healthz')
    assert response.status_code == 503 and response.json()['error'] == 'Archive storage unavailable'
    assert call('/api/events').status_code == 503
    print('PASS real Redis outage fails closed with 503')


def board():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            response = requests.get('http://board:8080/healthz', timeout=3)
            payload = response.json()
            if response.status_code == 503 and 'No usable V4L2' in payload.get('error', ''):
                assert payload['camera'] is None and payload['sequence'] == 0
                assert requests.get('http://board:8080/snapshot.jpg', timeout=3).status_code == 503
                # This endpoint closes an empty MJPEG response when no frame exists.
                # Repeated requests also exercise real WSGI admission cleanup.
                for _ in range(8):
                    stream = requests.get('http://board:8080/stream', timeout=3)
                    assert stream.status_code == 200 and stream.content == b''
                print('PASS real board CMD and outbox initialization; absent camera stays unavailable')
                return
        except (requests.RequestException, ValueError):
            pass
        time.sleep(.2)
    raise AssertionError('Board entry point did not start with truthful absent-camera readiness')


if __name__ == '__main__':
    import sys
    {'initial': initial, 'recovered': recovered, 'outage': outage, 'board': board}[sys.argv[1]]()
