import hmac
import json
import os
import re
import threading
from uuid import UUID
from urllib.parse import urlsplit

import requests
from flask import Flask, Response, jsonify, render_template, request, send_file
from redis.exceptions import RedisError
from rgb.events import InvalidEvent, validate_event


def create_app(store, sources=None, ingest_token=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2_000_000
    sources = sources if sources is not None else json.loads(os.environ['SOURCES_JSON'])
    token = ingest_token if ingest_token is not None else os.environ['INGEST_TOKEN']
    if len(token) < 32:
        raise ValueError('INGEST_TOKEN must have at least 32 characters')
    slots = threading.BoundedSemaphore(6)
    export_slot = threading.BoundedSemaphore(1)

    @app.before_request
    def ingest_only():
        if os.getenv('ROLE') == 'ingest' and request.endpoint not in {'ingest', 'health', 'board_policy'}:
            return 'Not found', 404

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.errorhandler(RedisError)
    def redis_error(exc):
        app.logger.error('Archive storage unavailable: %s', type(exc).__name__)
        return jsonify(error='Archive storage unavailable'), 503

    @app.get('/healthz')
    def health():
        store.redis.ping()
        return jsonify(status='ok')

    @app.get('/')
    def dashboard():
        return render_template('index.html')

    @app.get('/api/sources')
    def source_list():
        return jsonify([{'id':key, 'label':s.get('label',key), 'location':s.get('location') or None} for key,s in sources.items()])

    @app.post('/api/ingest')
    def ingest():
        if not hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+token):
            return jsonify(error='Unauthorized'), 401
        try:
            event, image = validate_event(request.get_json(silent=True), sources)
        except InvalidEvent:
            return jsonify(error='Invalid observation'), 400
        row_id, created = store.save(event, image)
        return jsonify(event_id=event['event_id'], cursor=row_id, created=created), 201 if created else 200

    @app.get('/api/settings')
    def settings():
        return jsonify(store.settings())

    @app.post('/api/settings')
    def change_settings():
        if not request.is_json or request.headers.get('X-RGB-Request') != 'settings':
            return jsonify(error='Expected settings request'), 400
        if request.headers.get('Origin') and urlsplit(request.headers['Origin']).netloc != request.host:
            return jsonify(error='Origin mismatch'), 403
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)!={'detector_threshold','vlm_threshold'}:
            return jsonify(error='Expected both confidence thresholds'),400
        if not all(type(v) in (int,float) and .05<=v<=.99 for v in data.values()):
            return jsonify(error='Thresholds must be between 0.05 and 0.99'),400
        return jsonify(store.update_settings(data,actor=request.headers.get('X-Laminarauth-Identity')))

    @app.get('/api/board-policy/<source_id>')
    def board_policy(source_id):
        if not hmac.compare_digest(request.headers.get('Authorization',''),'Bearer '+token):
            return jsonify(error='Unauthorized'),401
        if source_id not in sources:return 'Not found',404
        raw=store.redis.hget('rgb:refinements',source_id)
        return jsonify(settings=store.settings(),refinements=json.loads(raw) if raw else None)

    @app.post('/api/events/<event_id>/analyze')
    def reanalyze(event_id):
        if request.headers.get('X-RGB-Request')!='analysis':return jsonify(error='Expected analysis request'),400
        if not store.analysis_enabled:return jsonify(error='VLM disabled'),503
        try:event_id=str(UUID(event_id))
        except ValueError:return 'Not found',404
        event=store.get(event_id)
        if not event:return 'Not found',404
        queued=store.request_analysis(event_id)
        return jsonify(status='queued',created=queued),202

    @app.get('/api/detections/<key>/history')
    def detection_history(key):
        try:key=str(UUID(key))
        except ValueError:return 'Not found',404
        return jsonify(store.history(key))

    @app.get('/api/review/export/<format>')
    def export_data(format):
        if format not in {'parquet','arrow'}:return 'Not found',404
        if not export_slot.acquire(blocking=False):return jsonify(error='Export already running'),429
        try:
            from rgb.export import build_export
            directory,path=build_export(store,format)
            response=send_file(path,mimetype='application/zip',as_attachment=True,download_name='rgb-review-'+format+'.zip')
            response.direct_passthrough=False
        except Exception:
            export_slot.release()
            raise
        def cleanup():
            directory.cleanup()
            export_slot.release()
        response.call_on_close(cleanup)
        return response

    @app.get('/api/review/metrics')
    def review_metrics():
        return jsonify(store.redis.hgetall('rgb:metrics'))

    @app.post('/api/detections/<key>/evaluate')
    def evaluate_detection(key):
        if not request.is_json or request.headers.get('X-RGB-Request')!='evaluation':return jsonify(error='Expected evaluation request'),400
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)!={'reference_label','detector_result','vlm_result','notes'}:return jsonify(error='Invalid evaluation'),400
        if not isinstance(data['reference_label'],str) or not 1<=len(data['reference_label'])<=120 or not isinstance(data['notes'],str) or len(data['notes'])>4000:return jsonify(error='Invalid evaluation'),400
        if data['detector_result'] not in {'correct','incorrect','unscored'} or data['vlm_result'] not in {'correct','incorrect','unscored'}:return jsonify(error='Invalid outcome'),400
        try:key=str(UUID(key))
        except ValueError:return 'Not found',404
        review=store.evaluate(key,data,actor=request.headers.get('X-Laminarauth-Identity'))
        return (jsonify(review),201) if review else ('Not found',404)

    @app.get('/api/events')
    def events():
        before = request.args.get('before')
        if before and not re.fullmatch(r'\d+-\d+', before):
            return jsonify(error='Invalid cursor'), 400
        return jsonify(store.list(before=before))

    @app.get('/api/events/<event_id>')
    def detail(event_id):
        try:
            event_id = str(UUID(event_id))
        except ValueError:
            return jsonify(error='Not found'), 404
        event = store.get(event_id)
        if event:store.ensure_detections(store.raw(event_id))
        return (jsonify(event) if event else (jsonify(error='Not found'), 404))

    @app.get('/api/events/<event_id>/image.jpg')
    def image(event_id):
        try:
            event_id = str(UUID(event_id))
        except ValueError:
            return 'Not found', 404
        event = store.get(event_id)
        if not event:
            return 'Not found', 404
        path = store.root / (event['image_sha256']+'.jpg')
        if not path.exists():
            return 'Image unavailable', 503
        return send_file(path, mimetype='image/jpeg')

    @app.get('/api/live/<source_id>/status')
    def status(source_id):
        if source_id not in sources:
            return 'Not found', 404
        try:
            response = requests.get(sources[source_id]['url'].rstrip('/')+'/healthz', timeout=3)
            return jsonify(response.json()), response.status_code
        except (requests.RequestException, ValueError):
            return jsonify(error='Camera unavailable'), 503

    @app.get('/api/live/<source_id>/stream')
    def stream(source_id):
        if source_id not in sources:
            return 'Not found', 404
        if not slots.acquire(blocking=False):
            return 'Viewer limit reached', 429
        try:
            upstream = requests.get(sources[source_id]['url'].rstrip('/')+'/stream', timeout=(3,15), stream=True)
            upstream.raise_for_status()
        except requests.RequestException:
            slots.release()
            return 'Camera unavailable', 503
        released = False
        def close_upstream():
            nonlocal released
            if not released:
                released = True
                upstream.close()
                slots.release()
        def generate():
            try:
                for chunk in upstream.iter_content(16384):
                    yield chunk
            except requests.RequestException:
                return
            finally:
                close_upstream()
        response = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        response.call_on_close(close_upstream)
        return response

    return app
