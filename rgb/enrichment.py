"""Per-detection VLM readings, comparisons, and append-only enhancements."""
import base64
import io
import hashlib
import json
import math
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
for fields,limits in [
    (SCHEMA['properties']['observations']['items']['properties'],{'label':120,'reason':1500}),
    (SCHEMA['properties']['observations']['items']['properties']['materials']['items']['properties'],{'label':80,'reason':1500}),
]:
    for field,limit in limits.items():fields[field].update(minLength=1,maxLength=limit)


class AnalysisPaused(Exception):
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
    metadata=[{'box_index':i['index'],'classifier_label':event['detections'][i['index']]['label'],'classifier_confidence':event['detections'][i['index']]['confidence']} for i in items]
    prompt=('Read each numbered object crop. Image 1 is the full scene for context; subsequent images are the crops in the listed order. '
            'For each crop, identify the object with the most specific label supported by its appearance, then compare your reading to the classifier answer. '
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
    text=payload['message']['content'].strip()
    if text.startswith('```') and text.endswith('```'):
        lines=text.splitlines()
        if lines[0] not in {'```','```json'}:raise ValueError('Invalid JSON fence')
        text='\n'.join(lines[1:-1])
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    parsed=json.loads(text,object_pairs_hook=unique_object)
    if not isinstance(parsed,dict) or set(parsed)!={'observations'}:raise ValueError('Invalid root fields')
    observations=parsed['observations']
    if not isinstance(observations,list):raise ValueError('Invalid observation list')
    expected={i['index'] for i in items}
    seen=set()
    validated=[]
    for item in observations:
        if not isinstance(item,dict) or set(item)!={'box_index','label','confidence','comparison','reason','materials'}:raise ValueError('Invalid observation fields')
        index=item['box_index']
        confidence=item['confidence']
        if type(index) is not int or index not in expected or index in seen:raise ValueError('Mismatched box index')
        if not isinstance(item['label'],str) or not item['label'].strip() or not 1<=len(item['label'])<=120:raise ValueError('Invalid label')
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
        if not isinstance(item['comparison'],str) or item['comparison'] not in VERDICTS or not isinstance(item['reason'],str) or not item['reason'].strip() or not 1<=len(item['reason'])<=1500:raise ValueError('Invalid comparison')
        materials=item['materials']
        if not isinstance(materials,list) or len(materials)>8:raise ValueError('Invalid materials')
        for material in materials:
            if not isinstance(material,dict) or set(material)!={'label','confidence','reason'}:raise ValueError('Invalid material fields')
            if not isinstance(material['label'],str) or not material['label'].strip() or not 1<=len(material['label'])<=80:raise ValueError('Invalid material label')
            if type(material['confidence']) not in (int,float) or not math.isfinite(material['confidence']) or not 0<=material['confidence']<=1:raise ValueError('Invalid material confidence')
            if not isinstance(material['reason'],str) or not material['reason'].strip() or not 1<=len(material['reason'])<=1500:raise ValueError('Invalid material reason')
        seen.add(index)
        validated.append({k:item[k] for k in ['box_index','label','confidence','comparison','reason','materials']})
    if seen!=expected:raise ValueError('Missing box reading')
    return validated


def analyze(store,event,endpoints,job_id):
    check_pause(store)
    store.ensure_detections(event)
    state={'status':'running','schema_version':3,'job_id':job_id,'started_at':now(),'completed_boxes':0,'total_boxes':len(event['detections'])}
    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
    full,all_crops=crops(event,store.root)
    opinions=[]
    chosen={}
    for index,d in enumerate(event['detections']):
        saved=store.enhancement_for_job(d['detection_id'],job_id)
        if saved:chosen[index]=saved['candidate']
    state['completed_boxes']=len(chosen)
    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
    failures=0
    for endpoint in endpoints:
        opinion={'kind':'box_comparisons','model':endpoint['model'],'endpoint':endpoint['url'],'observed_at':now(),'readings':[],'errors':[]}
        for start in range(0,len(all_crops),4):
            batch=all_crops[start:start+4]
            recovered=[chosen[i['index']] for i in batch if i['index'] in chosen and chosen[i['index']].get('model')==endpoint['model'] and chosen[i['index']].get('endpoint')==endpoint['url']]
            opinion['readings'].extend(recovered)
            recovered_indices={r['box_index'] for r in recovered}
            batch=[i for i in batch if i['index'] not in recovered_indices]
            if not batch:continue
            try:
                readings,attempt_id=read_boxes(endpoint,event,full,batch,store,job_id)
                for reading in readings:
                    index=reading['box_index']
                    candidate={**reading,'attempt_id':attempt_id,'model':endpoint['model'],'endpoint':endpoint['url'],'observed_at':now(),'fingerprint':all_crops[index]['fingerprint']}
                    opinion['readings'].append(candidate)
                    if index not in chosen:
                        chosen[index]=store.enhance(event,index,candidate,job_id)
                    state['completed_boxes']=len(chosen)
                    store.redis.hset('rgb:analysis',event['event_id'],json.dumps(state))
            except (requests.RequestException,KeyError,TypeError,ValueError) as exc:
                failures+=1
                opinion['errors'].append({'box_indices':[i['index'] for i in batch],'error':str(exc)[:250] if isinstance(exc,ValueError) else type(exc).__name__})
        opinion['status']='complete' if not opinion['errors'] else 'partial' if opinion['readings'] else 'failed'
        opinion['text']='\n'.join(f"Box {r['box_index']+1}: {r['label']} ({r['confidence']:.0%}), {r['comparison']}. {r['reason']}" for r in opinion['readings'])
        opinions.append(opinion)
    state.update(status='complete' if len(chosen)==len(all_crops) else 'partial' if chosen else 'failed',finished_at=now())
    # Keep the previous scene-level opinions as historical entries.
    previous=store.redis.hget('rgb:opinions',event['event_id'])
    previous=json.loads(previous) if previous else []
    old=[o for o in previous if o.get('kind')!='box_comparisons']
    refinements={'event_id':event['event_id'],'source_id':event['source_id'],'observed_at':event['observed_at'],'readings':[{'box':event['detections'][i]['box'],'class_id':event['detections'][i]['class_id'],'track_id':event['detections'][i].get('track_id'),'detection_id':event['detections'][i]['detection_id'],**candidate} for i,candidate in chosen.items()]}
    with store.redis.pipeline(transaction=True) as pipe:
        pipe.hset('rgb:opinions',event['event_id'],json.dumps(old+opinions))
        pipe.hset('rgb:analysis',event['event_id'],json.dumps(state))
        # Manual readings of old snapshots must not replace newer live results.
        current=store.redis.hget('rgb:refinements',event['source_id'])
        if not current or json.loads(current)['observed_at']<=event['observed_at']:
            pipe.hset('rgb:refinements',event['source_id'],json.dumps(refinements))
        pipe.execute()
    return state


def run(store,endpoints):
    cursor=None
    request_cursor=None
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
            row_id,fields=rows[0]
            manual=stream=='rgb:analysis:requests'
            event_id=fields['event_id'] if manual else json.loads(fields['event'])['event_id']
            event=store.get(event_id)
            job_id=fields['job_id'] if manual else 'auto-v3:'+event_id
            try:
                if event:
                    state=event['analysis']
                    terminal=state.get('schema_version',0)>=2 and state['status'] in {'complete','partial','failed'}
                    if not (terminal and (not manual or state.get('job_id')==job_id)):
                        analyze(store,event,endpoints,job_id)
            except AnalysisPaused:
                state=json.loads(store.redis.hget('rgb:analysis',event_id) or '{}')
                state.update(status='paused',job_id=job_id)
                store.redis.hset('rgb:analysis',event_id,json.dumps(state))
                continue
            except RedisError:
                raise
            except Exception as exc:
                if event:store.processing_failure(event,job_id,exc)
                store.redis.hset('rgb:analysis',event_id,json.dumps({'status':'failed','schema_version':2,'job_id':job_id,'error':type(exc).__name__,'finished_at':now()}))
            if manual:
                store.redis.set('rgb:analysis:request_cursor',row_id)
                request_cursor=row_id
            else:
                store.redis.set('rgb:vlm:cursor',row_id)
                cursor=row_id
        except Exception as exc:
            log.warning('Enrichment retry: %s',type(exc).__name__)
            time.sleep(5)


def start(store,endpoints):
    if endpoints:threading.Thread(target=run,args=(store,endpoints),daemon=True).start()
