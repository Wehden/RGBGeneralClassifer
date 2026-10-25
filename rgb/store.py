"""Immutable initial observations, detection updates, and current projections."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4, uuid5
from common.display import value

DEFAULT_SETTINGS={'detector_threshold':.45,'vlm_threshold':.65,'display_mode':'objects','vlm_paused':False,'version':0}

APPEND = """
local existing = redis.call('HGET', KEYS[1], ARGV[1])
if existing then return {existing, 0} end
local id = redis.call('XADD', KEYS[2], '*', 'event', ARGV[2])
redis.call('HSET', KEYS[1], ARGV[1], id)
redis.call('HSET', KEYS[3], ARGV[1], ARGV[3])
return {id, 1}
"""
INITIAL = """
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('XADD', KEYS[2], '*', 'detection_id', ARGV[1], 'kind', 'initial', 'value', ARGV[2])
redis.call('XADD', KEYS[3], '*', 'kind', 'initial', 'value', ARGV[2])
return 1
"""
REQUEST = """
local state = redis.call('HGET', KEYS[1], ARGV[1])
if state then
 local s = cjson.decode(state)
 if s.status == 'queued' or s.status == 'running' or s.status == 'paused' then return 0 end
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('XADD', KEYS[2], '*', 'event_id', ARGV[1], 'job_id', ARGV[3])
return 1
"""
ENHANCE = """
local saved = redis.call('HGET', KEYS[5], ARGV[1])
if saved then return saved end
if redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 1 then return false end
redis.call('HSET', KEYS[1], ARGV[2], ARGV[3])
redis.call('XADD', KEYS[2], '*', 'detection_id', ARGV[2], 'kind', 'enhanced', 'value', ARGV[3])
redis.call('XADD', KEYS[4], '*', 'kind', 'enhanced', 'value', ARGV[3])
redis.call('SADD', KEYS[3], ARGV[1])
redis.call('HSET', KEYS[5], ARGV[1], ARGV[3])
return ARGV[3]
"""


class StaleEvaluation(ValueError):
    pass


def candidate_revision(candidate):
    return hashlib.sha256(json.dumps(candidate,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def detection_id(event_id,index):
    return str(uuid5(UUID(event_id),str(index)))


class Store:
    def __init__(self, redis, root, analysis_enabled=True):
        self.redis=redis
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.analysis_enabled=analysis_enabled

    def settings(self):
        raw=self.redis.get('rgb:settings')
        return {**DEFAULT_SETTINGS,**(json.loads(raw) if raw else {})}

    def location(self,source_id):
        raw=self.redis.hget('rgb:locations',source_id)
        return json.loads(raw) if raw else None

    def location_at(self,source_id,observed_at):
        observed=datetime.fromisoformat(observed_at)
        latest=self.location(source_id)
        if not latest or datetime.fromisoformat(latest['changed_at'])<=observed:return latest
        applicable=None
        for _,fields in self.stream('rgb:location-updates'):
            loc=json.loads(fields['value'])
            if loc['source_id']==source_id and datetime.fromisoformat(loc['changed_at'])<=observed:
                if not applicable or loc['version']>applicable['version']:applicable=loc
        return applicable

    def update_location(self,source_id,values,actor=None):
        script="""
        local value=cjson.decode(ARGV[2])
        value.version=redis.call('INCR',KEYS[1])
        local encoded=cjson.encode(value)
        redis.call('HSET',KEYS[2],ARGV[1],encoded)
        redis.call('XADD',KEYS[3],'*','value',encoded)
        return encoded
        """
        return json.loads(self.redis.eval(script,3,'rgb:location:version','rgb:locations','rgb:location-updates',source_id,json.dumps({'source_id':source_id,'changed_at':now(),'actor':actor,**values})))

    def update_settings(self, values, actor=None):
        # One Redis Lua operation preserves concurrent setting revisions and audit.
        script="""
        local version=redis.call('INCR',KEYS[1])
        local settings=cjson.decode(redis.call('GET',KEYS[2]) or ARGV[2])
        local updates=cjson.decode(ARGV[1])
        for key,value in pairs(updates) do settings[key]=value end
        settings.version=version
        local value=cjson.encode(settings)
        redis.call('SET',KEYS[2],value)
        redis.call('XADD',KEYS[3],'*','kind','thresholds','value',value)
        return value
        """
        return json.loads(self.redis.eval(script,3,'rgb:settings:version','rgb:settings','rgb:settings-updates',json.dumps({**values,'changed_at':now(),'actor':actor}),json.dumps(DEFAULT_SETTINGS)))

    def save(self,event,image):
        digest=hashlib.sha256(image).hexdigest()
        path=self.root/(digest+'.jpg')
        if not path.exists():
            tmp=self.root/(digest+'.'+os.urandom(6).hex()+'.tmp')
            try:
                with tmp.open('xb') as f:
                    f.write(image);f.flush();os.fsync(f.fileno())
                os.replace(tmp,path)
            finally:
                tmp.unlink(missing_ok=True)
        event['image_sha256']=digest
        state={'status':'queued' if self.analysis_enabled else 'disabled','schema_version':2}
        row_id,created=self.redis.eval(APPEND,3,'rgb:ids','rgb:events','rgb:analysis',event['event_id'],json.dumps(event),json.dumps(state))
        # Initial scene is durable first; deterministic IDs let a retry repair any
        # interrupted per-detection initialization without duplicating stream entries.
        original=self.raw(event['event_id'])
        self.ensure_detections(original)
        return row_id,bool(created)

    def raw(self,event_id):
        row=self.redis.hget('rgb:ids',event_id)
        if not row:return None
        rows=self.redis.xrange('rgb:events',row,row)
        return json.loads(rows[0][1]['event']) if rows else None

    def ensure_detections(self,event):
        event=self.raw(event['event_id']) or event
        for index,d in enumerate(event['detections']):
            key=detection_id(event['event_id'],index)
            initial={'detection_id':key,'scene_id':event['event_id'],'source_id':event['source_id'],'index':index,'observed_at':event['observed_at'],'recorded_at':now(),'initial':d,'candidate':None}
            self.redis.eval(INITIAL,3,'rgb:detections','rgb:detection-updates','rgb:detection:'+key+':events',key,json.dumps(initial))

    def get(self,event_id):
        event=self.raw(event_id)
        if not event:return None
        settings=self.settings()
        projected=[]
        for index,d in enumerate(event['detections']):
            key=detection_id(event_id,index)
            raw=self.redis.hget('rgb:detections',key)
            projection=json.loads(raw) if raw else {'candidate':None}
            candidate=projection.get('candidate')
            accepted=bool(candidate and candidate['confidence']>=settings['vlm_threshold'])
            displayed=value(d,candidate,settings)
            context={'candidate_revision':candidate_revision(candidate),'settings_version':settings['version'],'mode':settings['display_mode']}
            projected.append({**d,'detection_id':key,'display_label':displayed['label'],'display_confidence':displayed['confidence'],'label_provider':displayed['provider'],'enhancement':candidate,'enhancement_accepted':accepted,'review_context':context})
        event['detections']=projected
        opinions=self.redis.hget('rgb:opinions',event_id)
        event['opinions']=json.loads(opinions) if opinions else []
        state=self.redis.hget('rgb:analysis',event_id)
        event['analysis']=json.loads(state) if state else {'status':'not_analyzed','schema_version':1}
        event['settings']=settings
        return event

    def list(self,before=None,limit=50):
        rows=self.redis.xrevrange('rgb:events',max='('+before if before else '+',count=limit)
        return [{'cursor':row,**self.get(json.loads(fields['event'])['event_id'])} for row,fields in rows]

    def request_analysis(self,event_id):
        state={'status':'queued','schema_version':2,'queued_at':now()}
        return bool(self.redis.eval(REQUEST,2,'rgb:analysis','rgb:analysis:requests',event_id,json.dumps(state),str(uuid4())))

    def history(self,key):
        return [{'cursor':row,**json.loads(fields['value']),'kind':fields['kind']} for row,fields in reversed(self.redis.xrevrange('rgb:detection:'+key+':events',count=100))]

    def enhance(self,event,index,candidate,job_id):
        key=detection_id(event['event_id'],index)
        saved=self.enhancement_for_job(key,job_id)
        if saved:return saved['candidate']
        raw=self.redis.hget('rgb:detections',key)
        projection=json.loads(raw)
        projection['candidate']={**candidate,'job_id':job_id}
        projection['decision_settings']=self.settings()
        projection['updated_at']=now()
        saved=self.redis.eval(ENHANCE,5,'rgb:detections','rgb:detection-updates','rgb:enhancement-ids','rgb:detection:'+key+':events','rgb:enhancement-values',job_id+':'+key,key,json.dumps(projection))
        if not saved:
            saved=self.enhancement_for_job(key,job_id)
            if not saved:raise RuntimeError('Enhancement deduplication record missing')
            return saved['candidate']
        return json.loads(saved)['candidate']

    def enhancement_for_job(self,key,job_id):
        identity=job_id+':'+key
        saved=self.redis.hget('rgb:enhancement-values',identity)
        if saved:return json.loads(saved)
        # Recover enhancement values written before the per-job value index existed.
        if self.redis.sismember('rgb:enhancement-ids',identity):
            for _,fields in self.stream('rgb:detection:'+key+':events'):
                if fields['kind']=='enhanced':
                    saved=json.loads(fields['value'])
                    if saved['candidate'].get('job_id')==job_id:
                        self.redis.hsetnx('rgb:enhancement-values',identity,fields['value'])
                        return saved
        return None

    def record_attempt(self,event,indices,attempt):
        attempt['scene_id']=event['event_id']
        attempt['source_id']=event['source_id']
        attempt['image_sha256']=event['image_sha256']
        attempt['detection_ids']=[detection_id(event['event_id'],i) for i in indices]
        encoded=json.dumps(attempt)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.xadd('rgb:model-attempts',{'value':encoded})
            if attempt['outcome']=='started':
                pipe.hincrby('rgb:metrics','model_requests',1)
            else:
                pipe.hincrby('rgb:metrics','request_'+attempt['outcome'],1)
                pipe.hincrbyfloat('rgb:metrics','latency_ms_sum',attempt['latency_ms'])
            for key in attempt['detection_ids']:
                pipe.xadd('rgb:detection:'+key+':events',{'kind':'model_attempt','value':encoded})
                pipe.xadd('rgb:detection-updates',{'detection_id':key,'kind':'model_attempt','value':encoded})
            pipe.execute()

    def evaluate(self,key,values,context,actor=None):
        projection=self.redis.hget('rgb:detections',key)
        if not projection:return None
        projection=json.loads(projection)
        if context.get('candidate_revision')!=candidate_revision(projection['candidate']):
            raise StaleEvaluation('The model reading changed; refresh before evaluating it')
        version=context.get('settings_version')
        settings=DEFAULT_SETTINGS if version==0 else None
        if version!=0:
            for _,fields in self.stream('rgb:settings-updates'):
                recorded=json.loads(fields['value'])
                if recorded['version']==version:settings={**DEFAULT_SETTINGS,**recorded};break
        if settings is None or context.get('mode')!=settings['display_mode']:
            raise StaleEvaluation('The displayed settings could not be resolved')
        review={'evaluation_id':str(uuid4()),'detection_id':key,'scene_id':projection['scene_id'],'source_id':projection['source_id'],'reviewed_at':now(),'actor':actor,'initial':projection['initial'],'candidate':projection['candidate'],'settings':settings,'mode':settings['display_mode'],'review_context':context,**values}
        encoded=json.dumps(review)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.xadd('rgb:evaluations',{'value':encoded})
            pipe.xadd('rgb:detection:'+key+':events',{'kind':'evaluation','value':encoded})
            pipe.xadd('rgb:detection-updates',{'detection_id':key,'kind':'evaluation','value':encoded})
            pipe.execute()
        return review

    def stream(self,key,stop='+'):
        cursor='-'
        while True:
            rows=self.redis.xrange(key,min=cursor,max=stop,count=256)
            if not rows:return
            yield from rows
            cursor='('+rows[-1][0]

    def processing_failure(self,event,job_id,exc):
        value={'job_id':job_id,'scene_id':event['event_id'],'recorded_at':now(),'error_type':type(exc).__name__,'error':str(exc)[:1500]}
        encoded=json.dumps(value)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.hincrby('rgb:metrics','processing_failed',1)
            for i in range(len(event['detections'])):
                key=detection_id(event['event_id'],i)
                pipe.xadd('rgb:detection:'+key+':events',{'kind':'processing_failure','value':encoded})
                pipe.xadd('rgb:detection-updates',{'detection_id':key,'kind':'processing_failure','value':encoded})
            pipe.execute()
