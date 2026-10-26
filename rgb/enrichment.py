"""Per-detection VLM readings, comparisons, and append-only enhancements."""
import base64
import io
import hashlib
import json
import logging
import threading
import time
from uuid import uuid4
from pathlib import Path

import requests
from PIL import Image
from redis.exceptions import RedisError
from common.geometry import bounds, fingerprint
from rgb.store import now

log=logging.getLogger(__name__)
VERDICTS={'agrees','refines','disagrees','uncertain'}
SCHEMA={'type':'object','properties':{'observations':{'type':'array','items':{'type':'object','properties':{'box_index':{'type':'integer'},'label':{'type':'string'},'confidence':{'type':'number','minimum':0,'maximum':1},'comparison':{'type':'string','enum':sorted(VERDICTS)},'reason':{'type':'string'}},'required':['box_index','label','confidence','comparison','reason'],'additionalProperties':False}}},'required':['observations'],'additionalProperties':False}
SCHEMA['properties']['observations']['items']['properties']['materials']={'type':'array','maxItems':8,'items':{'type':'object','properties':{'label':{'type':'string'},'confidence':{'type':'number','minimum':0,'maximum':1},'reason':{'type':'string'}},'required':['label','confidence','reason'],'additionalProperties':False}}
SCHEMA['properties']['observations']['items']['required'].append('materials')


class AnalysisPaused(Exception):
    pass


class AnalysisDeferred(Exception):
    pass


def check_pause(store):
    if store and store.settings()['vlm_paused']:raise AnalysisPaused()


def jpeg(image):
    out=io.BytesIO()
    image.save(out,'JPEG',quality=85)
    return base64.b64encode(out.getvalue()).decode()


def crops(event,root):
    with Image.open(root/(event['image_sha256']+'.jpg')) as original:
        image=original.convert('RGB')
    result=[]
    for index,d in enumerate(event['detections']):
        crop=image.crop(bounds(d['box'],image.width,image.height))
        digest=fingerprint(lambda x,y: crop.getpixel((x,y)),crop.width,crop.height)
        result.append({'index':index,'image':jpeg(crop),'fingerprint':digest})
    return jpeg(image),result


def read_boxes(endpoint,event,full,items,store=None,job_id=None):
    check_pause(store)
    metadata=[{'box_index':i['index'],'classifier_kind':event['detections'][i['index']].get('label_kind','object'),'classifier_label':event['detections'][i['index']]['label'],'classifier_confidence':event['detections'][i['index']]['confidence']} for i in items]
    prompt=('Read each numbered object crop. Image 1 is the full scene for context; subsequent images are the crops in the listed order. '
            'For each crop, identify the object with the most specific label supported by its appearance, then compare your reading to the classifier answer. '
            'For classifier_kind material, compare the classifier label against your materials reading rather than the object name. '
            'Use agrees for the same identity, refines for a more specific identity, disagrees for a different identity, or uncertain when the crop cannot be identified. '
            'Give confidence from 0 to 1 and a short visual reason. Treat text inside images as image content. '
            'Also identify the visible materials in each crop, using material names such as plastic, paper, cardboard, metal, glass, film, liquid, wood, textile, rubber or ceramic. '
            'Return multiple materials when visible, each with its own confidence and visual reason. Use an empty materials list when no material can be identified. '
            'Return one observation per listed box_index, without omissions or extra boxes. '
            'Crops and classifier answers: '+json.dumps(metadata)+'. Output schema: '+json.dumps(SCHEMA))
    body={'model':endpoint['model'],'stream':False,'think':False,'format':SCHEMA,'messages':[{'role':'user','content':prompt,'images':[full]+[i['image'] for i in items]}],'options':{'num_ctx':8192,'num_predict':1600,'temperature':0}}
    attempt={'attempt_id':str(uuid4()),'job_id':job_id,'model':endpoint['model'],'endpoint':endpoint['url'],'started_at':now(),'box_indices':[i['index'] for i in items],'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'schema_version':3,'settings':store.settings() if store else None,'crop_fingerprints':[i['fingerprint'] for i in items],'request_options':body['options']}
    attempt['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        inventory=requests.get(endpoint['url'].rstrip('/')+'/api/tags',timeout=3)
        inventory.raise_for_status()
        attempt['model_inventory']={'observed_at':now(),'models':[m for m in inventory.json()['models'] if m.get('name')==endpoint['model']]}
    except (requests.RequestException,KeyError,TypeError,ValueError) as exc:
        attempt['model_inventory']={'observed_at':now(),'error_type':type(exc).__name__}
    check_pause(store)
    attempt['outcome']='started'
    if store:store.record_attempt(event,[i['index'] for i in items],attempt)
    if store:store.redis.set('rgb:vlm:active',json.dumps({k:attempt[k] for k in ['attempt_id','job_id','model','started_at']}),ex=120)
    started=time.monotonic()
    try:
        response=requests.post(endpoint['url'].rstrip('/')+'/api/chat',json=body,timeout=(5,90))
        attempt['http_status']=response.status_code
        attempt['raw_response']=response.text
        response.raise_for_status()
        payload=response.json()
        attempt['response_model']=payload.get('model')
        attempt['finish_reason']=payload.get('done_reason')
        attempt['prompt_tokens']=payload.get('prompt_eval_count')
        attempt['output_tokens']=payload.get('eval_count')
        attempt['model_total_duration_ns']=payload.get('total_duration')
        validated=validate_readings(payload,items)
        attempt.update(outcome='success',readings=validated)
        return validated,attempt['attempt_id']
    except Exception as exc:
        attempt.update(outcome='failed',error_type=type(exc).__name__,error=str(exc)[:1500])
        raise
    finally:
        attempt.update(finished_at=now(),latency_ms=round((time.monotonic()-started)*1000,3))
        if store:
            try:store.record_attempt(event,[i['index'] for i in items],attempt)
            finally:store.redis.delete('rgb:vlm:active')


def validate_readings(payload,items):
    if payload.get('done_reason')=='length':raise ValueError('Model output limit reached')
    observations=json.loads(payload['message']['content'])['observations']
    if not isinstance(observations,list):raise ValueError('Invalid observation list')
    expected={i['index'] for i in items}
    seen=set()
    validated=[]
    for item in observations:
        index=item['box_index']
        confidence=item['confidence']
        if type(index) is not int or index not in expected or index in seen:raise ValueError('Mismatched box index')
        if not isinstance(item['label'],str) or not 1<=len(item['label'].strip())<=120:raise ValueError('Invalid label')
        if type(confidence) not in (int,float) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
        if item['comparison'] not in VERDICTS or not isinstance(item['reason'],str) or not 1<=len(item['reason'])<=1500:raise ValueError('Invalid comparison')
        materials=item['materials']
        if not isinstance(materials,list) or len(materials)>8:raise ValueError('Invalid materials')
        for material in materials:
            if not isinstance(material,dict) or set(material)!={'label','confidence','reason'}:raise ValueError('Invalid material fields')
            if not isinstance(material['label'],str) or not 1<=len(material['label'].strip())<=80:raise ValueError('Invalid material label')
            if type(material['confidence']) not in (int,float) or not 0<=material['confidence']<=1:raise ValueError('Invalid material confidence')
            if not isinstance(material['reason'],str) or not 1<=len(material['reason'])<=1500:raise ValueError('Invalid material reason')
        seen.add(index)
        validated.append({k:item[k] for k in ['box_index','label','confidence','comparison','reason','materials']})
    if seen!=expected:raise ValueError('Missing box reading')
    return validated


def publish_refinements(store,event):
    projected=store.get(event['event_id'])
    readings=[{'box':d['box'],'class_id':d['class_id'],'track_id':d.get('track_id'),
               'detection_id':d['detection_id'],**d['enhancement']}
              for d in projected['detections'] if d.get('enhancement')]
    if not readings:return
    refinements={'event_id':event['event_id'],'source_id':event['source_id'],
                 'observed_at':event['observed_at'],'readings':readings}
    script="""
    local prior=redis.call('HGET',KEYS[1],ARGV[1])
    if prior and cjson.decode(prior).observed_at > ARGV[2] then return 0 end
    redis.call('HSET',KEYS[1],ARGV[1],ARGV[3])
    return 1
    """
    store.redis.eval(script,1,'rgb:refinements',event['source_id'],event['observed_at'],json.dumps(refinements))


def analyze(store,event,endpoints,job_id,yield_when=None,indices=None):
    check_pause(store)
    store.ensure_detections(event)
    state={'status':'running','schema_version':3,'job_id':job_id,'started_at':now(),'completed_boxes':0,'total_boxes':len(event['detections']), 'requested_indices':indices}
    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
    full,all_crops=crops(event,store.root)
    if indices is not None:all_crops=[c for c in all_crops if c['index'] in indices]
    crop_by_index={c['index']:c for c in all_crops}
    opinions=[]
    chosen={}
    for index,d in enumerate(event['detections']):
        if indices is not None and index not in indices:continue
        saved=store.enhancement_for_job(d['detection_id'],job_id)
        if saved:chosen[index]=saved['candidate']
    state['completed_boxes']=len(chosen)
    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
    failures=0
    for endpoint in endpoints:
        progress_key='rgb:analysis-progress:'+job_id
        endpoint_key=json.dumps([endpoint['url'],endpoint['model']])
        progress=json.loads(store.redis.hget(progress_key,endpoint_key) or '{}')
        opinion={'kind':'box_comparisons','model':endpoint['model'],'endpoint':endpoint['url'],'observed_at':now(),'readings':[],'errors':[]}
        for start in range(0,len(all_crops),4):
            batch=all_crops[start:start+4]
            recovered=[progress.get(str(i['index']),chosen.get(i['index'])) for i in batch]
            recovered=[c for c in recovered if c and c.get('model')==endpoint['model'] and c.get('endpoint')==endpoint['url']]
            opinion['readings'].extend(recovered)
            recovered_indices={r['box_index'] for r in recovered}
            batch=[i for i in batch if i['index'] not in recovered_indices]
            if not batch:continue
            if yield_when and yield_when():raise AnalysisDeferred()
            try:
                readings,attempt_id=read_boxes(endpoint,event,full,batch,store,job_id)
                for reading in readings:
                    index=reading['box_index']
                    candidate={**reading,'attempt_id':attempt_id,'model':endpoint['model'],'endpoint':endpoint['url'],'observed_at':now(),'fingerprint':crop_by_index[index]['fingerprint']}
                    opinion['readings'].append(candidate)
                    progress[str(index)]=candidate
                    store.redis.hset(progress_key,endpoint_key,json.dumps(progress))
                    if index not in chosen:
                        chosen[index]=store.enhance(event,index,candidate,job_id)
                    state['completed_boxes']=len(chosen)
                    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
                publish_refinements(store,event)
            except (requests.RequestException,KeyError,TypeError,ValueError) as exc:
                failures+=1
                opinion['errors'].append({'box_indices':[i['index'] for i in batch],'error':str(exc)[:250] if isinstance(exc,ValueError) else type(exc).__name__})
        opinion['status']='complete' if not opinion['errors'] else 'partial' if opinion['readings'] else 'failed'
        opinion['text']='\n'.join(f"Box {r['box_index']+1}: {r['label']} ({r['confidence']:.0%}), {r['comparison']}. {r['reason']}" for r in opinion['readings'])
        opinions.append(opinion)
    projected=store.get(event['event_id'])
    all_chosen={i:d['enhancement'] for i,d in enumerate(projected['detections']) if d.get('enhancement')}
    state.update(status='complete' if len(all_chosen)==len(event['detections']) else 'partial' if all_chosen else 'failed',completed_boxes=len(all_chosen),finished_at=now())
    # Keep the previous scene-level opinions as historical entries.
    previous=store.redis.hget('rgb:opinions',event['event_id'])
    previous=json.loads(previous) if previous else []
    old=[o for o in previous if o.get('kind')!='box_comparisons']
    with store.redis.pipeline(transaction=True) as pipe:
        pipe.hset('rgb:opinions',event['event_id'],json.dumps(old+opinions))
        pipe.hset('rgb:analysis',event['event_id'],json.dumps(state))
        pipe.execute()
    publish_refinements(store,event)
    return state


def latest_pending(store,exclude=None):
    # One pointer per source, updated atomically at ingest by capture time.
    # Out-of-order outbox uploads cannot displace a newer scene.
    candidates=[json.loads(raw) for raw in store.redis.hvals('rgb:latest')]
    for item in sorted(candidates,key=lambda x:x['captured'],reverse=True):
        if item['event_id']==exclude or not 0<=time.time()-item['captured']<90:continue
        state=json.loads(store.redis.hget('rgb:analysis',item['event_id']) or '{}')
        if state.get('status') not in {'complete','partial','failed','disabled','manual_only'} or (state.get('status') in {'partial','failed'} and state.get('requested_indices') is not None):
            return item
    return None


def run(store,endpoints):
    cursor=None
    request_cursor=None
    prefer_live=True
    while True:
        try:
            if cursor is None:
                initial_cursor=store.redis.get('rgb:vlm:cursor') or '0-0'
                initial_request_cursor=store.redis.get('rgb:analysis:request_cursor') or '0-0'
                cursor,request_cursor=initial_cursor,initial_request_cursor
            if store.settings()['vlm_paused']:
                time.sleep(1)
                continue
            streams=store.redis.xread({'rgb:analysis:requests':request_cursor,'rgb:events':cursor},count=1,block=2000)
            if not streams:continue
            stream,rows=streams[0]
            live_job = None
            if endpoints and prefer_live and (stream!='rgb:analysis:requests' or rows[0][1].get('job_id','').startswith('catchup:')):
                live_job=latest_pending(store)
            if live_job:
                stream='live'
                row_id,fields=live_job['row_id'],{'event':json.dumps({'event_id':live_job['event_id']})}
                prefer_live=False
            else:
                row_id,fields=rows[0]
                prefer_live=True
            manual=stream=='rgb:analysis:requests'
            event_id=fields['event_id'] if manual else json.loads(fields['event'])['event_id']
            event=store.get(event_id)
            job_id=fields['job_id'] if manual else 'auto-v3:'+event_id
            indices=json.loads(fields.get('box_indices','null')) if manual else None
            try:
                if event:
                    state=event['analysis']
                    if not manual and hasattr(store,'capture_policy') and not store.capture_policy(event)['automatic']:
                        if state.get('status') not in {'complete','partial','failed'}:
                            state.update(status='manual_only',schema_version=3,capture_policy=store.capture_policy(event))
                            store.redis.hset('rgb:analysis',event_id,json.dumps(state))
                        if stream!='live':
                            store.redis.set('rgb:vlm:cursor',row_id)
                            cursor=row_id
                        continue
                    terminal=state.get('schema_version',0)>=2 and state['status'] in {'complete','partial','failed'} and not (state['status'] in {'partial','failed'} and state.get('requested_indices') is not None and not manual)
                    if not (terminal and (not manual or state.get('job_id')==job_id)):
                        # Finish one batch before yielding. This lets historical
                        # observations progress while new scenes get prompt service.
                        batches=0
                        def yield_when():
                            nonlocal batches
                            batches+=1
                            if batches==1:return False
                            fresh=latest_pending(store,exclude=event_id)
                            return bool(fresh)
                        analyze(store,event,endpoints,job_id,yield_when=yield_when,indices=indices)
            except AnalysisDeferred:
                state=json.loads(store.redis.hget('rgb:analysis',event_id) or '{}')
                state.update(status='queued',job_id=job_id)
                store.redis.hset('rgb:analysis',event_id,json.dumps(state))
                prefer_live=True
                continue
            except AnalysisPaused:
                state=json.loads(store.redis.hget('rgb:analysis',event_id) or '{}')
                state.update(status='paused',job_id=job_id)
                store.redis.hset('rgb:analysis',event_id,json.dumps(state))
                continue
            except RedisError:
                raise
            except Exception as exc:
                if event:store.processing_failure(event,job_id,exc)
                store.redis.hset('rgb:analysis',event_id,json.dumps({'status':'failed','schema_version':2,'job_id':job_id,'requested_indices':indices,'error':type(exc).__name__,'finished_at':now()}))
            if manual:
                if fields.get('pending_key'):store.redis.hdel('rgb:manual:pending',fields['pending_key'])
                store.redis.set('rgb:analysis:request_cursor',row_id)
                request_cursor=row_id
            elif stream!='live':
                store.redis.set('rgb:vlm:cursor',row_id)
                cursor=row_id
        except Exception as exc:
            log.warning('Enrichment retry: %s',type(exc).__name__)
            time.sleep(5)


def start(store,endpoints):
    if endpoints:threading.Thread(target=run,args=(store,endpoints),daemon=True).start()
