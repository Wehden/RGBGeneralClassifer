import hmac
import json
import math
import os
import re
import threading
from uuid import UUID
from urllib.parse import urlsplit

import requests
from flask import Flask, Response, jsonify, render_template, request, send_file
from redis.exceptions import RedisError
from rgb.events import InvalidEvent, validate_event
from rgb.store import StaleEvaluation


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

    @app.get('/records')
    def records_page():
        return render_template('records.html')

    @app.get('/api/review/records')
    def records_query():
        from rgb.records import query,download
        format=request.args.get('format','json')
        if format not in {'json','parquet','arrow'}:return jsonify(error='Expected json, parquet or arrow'),400
        try:result=query(store,request.args)
        except (TypeError,ValueError):return jsonify(error='Invalid records query; use source, from/to with timezone, label, material, status, limit 1-200, and before cursor'),400
        if format=='json':return jsonify(result)
        response=send_file(download(result['records'],format),mimetype='application/octet-stream',as_attachment=True,download_name='rgb-records-page.'+format)
        if result['next_cursor']:response.headers['X-RGB-Next-Cursor']=result['next_cursor']
        return response

    @app.get('/api/sources')
    def source_list():
        result=[]
        for key,s in sources.items():
            loc=store.location(key)
            label=loc['label'] if loc else s.get('location')
            if loc and not label and loc.get('latitude') is not None:label=f"{loc['latitude']:.5f}, {loc['longitude']:.5f}"
            result.append({'id':key,'label':s.get('label',key),'location':label or None,'location_method':loc['method'] if loc else 'configured' if label else 'disabled'})
        return jsonify(result)

    @app.get('/api/sources/<source_id>/location')
    def source_location(source_id):
        if source_id not in sources:return 'Not found',404
        return jsonify(store.location(source_id) or {'source_id':source_id,'method':'manual' if sources[source_id].get('location') else 'disabled','label':sources[source_id].get('location') or None,'latitude':None,'longitude':None,'accuracy_m':None,'version':0})

    @app.post('/api/sources/<source_id>/location')
    def change_location(source_id):
        if source_id not in sources:return 'Not found',404
        if not request.is_json or request.headers.get('X-RGB-Request')!='location':return jsonify(error='Expected location request'),400
        if request.headers.get('Origin') and urlsplit(request.headers['Origin']).netloc!=request.host:return jsonify(error='Origin mismatch'),403
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)!={'method','label','latitude','longitude','accuracy_m'}:return jsonify(error='Invalid location'),400
        if data['method'] not in ['manual','browser','disabled'] or not isinstance(data['label'],str) or len(data['label'])>120:return jsonify(error='Invalid location method or label'),400
        lat,lon,accuracy=data['latitude'],data['longitude'],data['accuracy_m']
        def number(n,lo,hi):return type(n) in (int,float) and math.isfinite(n) and lo<=n<=hi
        if (lat is None)!=(lon is None) or lat is not None and not(number(lat,-90,90) and number(lon,-180,180)):return jsonify(error='Invalid coordinates'),400
        if accuracy is not None and not number(accuracy,0,1e7):return jsonify(error='Invalid accuracy'),400
        if data['method']=='browser' and lat is None:return jsonify(error='Browser location requires coordinates'),400
        if data['method']=='disabled':data.update(label='',latitude=None,longitude=None,accuracy_m=None)
        return jsonify(store.update_location(source_id,data,request.headers.get('X-Laminarauth-Identity')))

    @app.post('/api/ingest')
    def ingest():
        if not hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+token):
            return jsonify(error='Unauthorized'), 401
        try:
            data=request.get_json(silent=True)
            event, image = validate_event(data, sources)
            loc=store.location_at(event['source_id'],event['observed_at'])
            if loc:event.update(location=loc['label'] or None,location_kind=loc['method'],location_record=loc)
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
        if not isinstance(data,dict) or set(data) not in [{'detector_threshold','vlm_threshold'},{'display_mode'},{'detector_threshold','vlm_threshold','display_mode'},{'vlm_paused'}]:
            return jsonify(error='Expected confidence thresholds, display mode or vlm_paused'),400
        if 'display_mode' in data and data['display_mode'] not in ['objects','materials']:
            return jsonify(error='Expected objects or materials mode'),400
        if 'vlm_paused' in data and type(data['vlm_paused']) is not bool:
            return jsonify(error='vlm_paused must be boolean'),400
        if not all(type(v) in (int,float) and .05<=v<=.99 for k,v in data.items() if k in {'detector_threshold','vlm_threshold'}):
            return jsonify(error='Thresholds must be between 0.05 and 0.99'),400
        return jsonify(store.update_settings(data,actor=request.headers.get('X-Laminarauth-Identity')))

    @app.get('/api/vlm/status')
    def vlm_status():
        active=store.redis.get('rgb:vlm:active')
        return jsonify(paused=store.settings()['vlm_paused'],active=json.loads(active) if active else None,enabled=store.analysis_enabled)

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
        if not isinstance(data,dict) or set(data)!={'reference_label','detector_result','vlm_result','notes','review_context'}:return jsonify(error='Expected evaluation and displayed review_context'),400
        context=data.pop('review_context')
        if not isinstance(context,dict) or set(context)!={'candidate_revision','settings_version','mode'} or type(context['settings_version']) is not int or context['mode'] not in ['objects','materials']:return jsonify(error='Invalid review context'),400
        if not isinstance(data['reference_label'],str) or not 1<=len(data['reference_label'])<=120 or not isinstance(data['notes'],str) or len(data['notes'])>4000:return jsonify(error='Invalid evaluation'),400
        if data['detector_result'] not in ['correct','incorrect','unscored'] or data['vlm_result'] not in ['correct','incorrect','unscored']:return jsonify(error='Invalid outcome'),400
        try:key=str(UUID(key))
        except ValueError:return 'Not found',404
        try:review=store.evaluate(key,data,context,actor=request.headers.get('X-Laminarauth-Identity'))
        except StaleEvaluation as exc:return jsonify(error=str(exc)),409
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
