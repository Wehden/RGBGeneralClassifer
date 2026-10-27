import hmac
import json
import math
import os
import re
import threading
from uuid import UUID
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, render_template, request, send_file
from redis.exceptions import RedisError
from rgb.events import InvalidEvent, validate_event
from rgb.store import StaleEvaluation
from rgb import live


def create_app(store, sources=None, ingest_token=None, analysis_models=None, local_runtime=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2_000_000
    sources = sources if sources is not None else json.loads(os.environ['SOURCES_JSON'])
    token = None if local_runtime else ingest_token if ingest_token is not None else os.environ['INGEST_TOKEN']
    if token is not None and len(token) < 32:
        raise ValueError('INGEST_TOKEN must have at least 32 characters')
    device_tokens = json.loads(os.getenv('DEVICE_TOKENS_JSON','{}'))
    if any(source not in sources or not isinstance(secret,str) or len(secret)<32
           for source,secret in device_tokens.items()):
        raise ValueError('Invalid device credentials')
    live_cache=live.Cache()
    app.extensions['live_cache']=live_cache
    slots = threading.BoundedSemaphore(6)
    export_slot = threading.BoundedSemaphore(1)

    def actor():
        return None if local_runtime else request.headers.get('X-Laminarauth-Identity')

    @app.before_request
    def ingest_only():
        if not local_runtime and os.getenv('ROLE') == 'ingest' and request.endpoint not in {'ingest', 'health', 'board_policy', 'device_observations', 'device_policy', 'device_live'}:
            return 'Not found', 404

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.errorhandler(RedisError)
    def redis_error(exc):
        app.logger.error('Archive storage unavailable: %s', type(exc).__name__)
        return jsonify(error='Archive storage unavailable'), 503

    @app.get('/healthz')
    def health():
        store.ping()
        return jsonify(status='ok',**({'camera':local_runtime.camera.status(),'detector':local_runtime.camera.detector_status} if local_runtime else {}))

    @app.get('/')
    def dashboard():
        return render_template('index.html',local=bool(local_runtime))

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
            name=local_runtime.camera.name if local_runtime and s.get('local') else s.get('label',key)
            result.append({'id':key,'label':name,'location':label or None,'location_method':loc['method'] if loc else 'configured' if label else 'disabled','local':bool(s.get('local'))})
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
        return jsonify(store.update_location(source_id,data,actor()))

    @app.post('/api/ingest')
    def ingest():
        if token is None:return 'Not found',404
        if not hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+token):
            return jsonify(error='Unauthorized'), 401
        return save_observation(request.get_json(silent=True))

    def save_observation(data):
        try:
            event, image = validate_event(data, sources)
            loc=store.location_at(event['source_id'],event['observed_at'])
            if loc:event.update(location=loc['label'] or None,location_kind=loc['method'],location_record=loc)
        except InvalidEvent:
            return jsonify(error='Invalid observation'), 400
        row_id, created = store.save(event, image)
        return jsonify(event_id=event['event_id'], cursor=row_id, created=created, recorded=row_id is not None), 201 if created else 200

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
        allowed=[{'detector_threshold','vlm_threshold'},{'display_mode'},{'detector_threshold','vlm_threshold','display_mode'},{'vlm_paused'}]
        if not local_runtime:allowed.append({'classification_running','record_observations'})
        if local_runtime:allowed.extend([{'selected_model','model_selection_reason'},
                                        {'selected_provider','selected_model','model_selection_reason'},{'capture_paused'},
                                        {'capture_interval_s'},{'capture_paused','capture_interval_s'}])
        if not isinstance(data,dict) or set(data) not in allowed:
            return jsonify(error='Expected confidence thresholds, display mode or vlm_paused'),400
        if 'display_mode' in data and data['display_mode'] not in ['objects','materials']:
            return jsonify(error='Expected objects or materials mode'),400
        if 'vlm_paused' in data and type(data['vlm_paused']) is not bool:
            return jsonify(error='vlm_paused must be boolean'),400
        if 'capture_paused' in data and type(data['capture_paused']) is not bool:
            return jsonify(error='capture_paused must be boolean'),400
        if 'capture_interval_s' in data and (type(data['capture_interval_s']) not in {int,float}
                or not math.isfinite(data['capture_interval_s']) or not 1<=data['capture_interval_s']<=86400):
            return jsonify(error='Capture interval must be 1 to 86400 seconds'),400
        if 'selected_model' in data:
            from rgb.vlm_providers import model_selection
            provider=data.get('selected_provider',store.settings()['selected_provider'])
            try:_,reason=model_selection(provider,data['selected_model'])
            except ValueError as exc:return jsonify(error=str(exc)),400
            if data['model_selection_reason']!=reason:
                return jsonify(error='Expected supported provider/model with its selection reason'),400
            data['selected_provider']=provider
        if any(type(data[k]) is not bool for k in ['classification_running','record_observations'] if k in data):return jsonify(error='Classification controls must be boolean'),400
        if not all(type(v) in (int,float) and .05<=v<=.99 for k,v in data.items() if k in {'detector_threshold','vlm_threshold'}):
            return jsonify(error='Thresholds must be between 0.05 and 0.99'),400
        if 'capture_paused' in data or 'capture_interval_s' in data:
            return jsonify(local_runtime.update_capture_settings(data,actor=actor()))
        return jsonify(store.update_settings(data,actor=actor()))

    @app.get('/api/vlm/status')
    def vlm_status():
        if local_runtime:return jsonify(local_runtime.status())
        return jsonify(paused=store.settings()['vlm_paused'],active=store.active_attempt(),enabled=store.analysis_enabled,models=analysis_models or [],manual_pending=store.redis.hlen('rgb:manual:pending'))

    @app.get('/api/board-policy/<source_id>')
    def board_policy(source_id):
        if token is None:return 'Not found',404
        if not hmac.compare_digest(request.headers.get('Authorization',''),'Bearer '+token):
            return jsonify(error='Unauthorized'),401
        if source_id not in sources:return 'Not found',404
        return policy_for(source_id)

    def policy_for(source_id):
        return jsonify(settings=store.settings(),refinements=store.refinements(source_id))

    @app.post('/api/events/<event_id>/analyze')
    def reanalyze(event_id):
        if request.headers.get('X-RGB-Request')!='analysis':return jsonify(error='Expected analysis request'),400
        if not store.analysis_enabled:return jsonify(error='VLM disabled'),503
        try:event_id=str(UUID(event_id))
        except ValueError:return 'Not found',404
        event=store.get(event_id)
        if not event:return 'Not found',404
        queued=(store.request_analysis(event_id) if local_runtime else
                store.request_analysis(event_id,actor=actor()))
        return jsonify(status='queued',created=queued),202

    @app.post('/api/detections/<key>/analyze')
    def analyze_detection(key):
        if local_runtime:return jsonify(error='Use observation analysis for the local runner'),404
        if request.headers.get('X-RGB-Request')!='analysis':return jsonify(error='Expected analysis request'),400
        if not store.analysis_enabled:return jsonify(error='VLM disabled'),503
        try:key=str(UUID(key))
        except ValueError:return 'Not found',404
        raw=store.redis.hget('rgb:detections',key)
        if not raw:return 'Not found',404
        detection=json.loads(raw)
        created=store.request_analysis(detection['scene_id'],[detection['index']],actor=request.headers.get('X-Laminarauth-Identity'))
        return jsonify(status='queued',created=created,detection_id=key),202

    @app.post('/api/analysis/catch-up')
    def catch_up():
        if local_runtime:return jsonify(error='Catch-up is available for board observations'),404
        if request.headers.get('X-RGB-Request')!='analysis':return jsonify(error='Expected analysis request'),400
        if not store.analysis_enabled:return jsonify(error='VLM disabled'),503
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or set(data)-{'source_id','batch_id'} or data.get('source_id') not in sources:
            return jsonify(error='Expected configured source'),400
        try:result=store.catch_up(data['source_id'],data.get('batch_id'),request.headers.get('X-Laminarauth-Identity'))
        except ValueError:return jsonify(error='Invalid catch-up request'),400
        return jsonify(result),202

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
        return jsonify(store.metrics())

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
        try:review=store.evaluate(key,data,context,actor=actor())
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

    def device_authorized(source_id):
        expected = device_tokens.get(source_id)
        return expected and hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+expected)

    @app.post('/api/device/<source_id>/observations')
    def device_observations(source_id):
        if not device_authorized(source_id):return jsonify(error='Unauthorized'),401
        data=request.get_json(silent=True)
        if not isinstance(data,dict) or data.get('source_id')!=source_id:
            return jsonify(error='Source mismatch'),400
        return save_observation(data)

    @app.get('/api/device/<source_id>/policy')
    def device_policy(source_id):
        if not device_authorized(source_id):return jsonify(error='Unauthorized'),401
        return policy_for(source_id)

    @app.post('/api/device/<source_id>/live')
    def device_live(source_id):
        if not device_authorized(source_id):return jsonify(error='Unauthorized'),401
        try:frame=live.validate(request.get_json(silent=True),source_id)
        except ValueError:return jsonify(error='Invalid live frame'),400
        live_cache.put(source_id,frame)
        return jsonify(source_id=source_id,accepted=True)

    @app.get('/api/live/<source_id>/status')
    def status(source_id):
        if source_id not in sources:
            return 'Not found', 404
        if local_runtime:
            result=local_runtime.camera.status()
            return jsonify({**result,'inference_ms':result['detector'].get('detector_ms'),'fps':result['capture_fps'],
                             'model':result['detector'].get('model'),'detector_config':result['detector'].get('config',{'runtime':'ONNX CPU'}),
                            'detections':result['detector'].get('count',0)})
        frame=live_cache.read(source_id)
        if frame is None:return jsonify(error='Camera unavailable'),503
        return jsonify(frame['status'])

    @app.get('/api/live/<source_id>/stream')
    def stream(source_id):
        if source_id not in sources:return 'Not found',404
        if local_runtime:return camera_api.stream_response('color',overlay=True)
        return live.response(live_cache,source_id,slots)

    if local_runtime:
        from rgb.camera_api import CameraAPI
        camera_api=CameraAPI(app,local_runtime,export_slot)
    return app
