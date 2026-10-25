"""Immutable initial observations, detection updates, and current projections."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4, uuid5
from common.display import value

DEFAULT_SETTINGS={'detector_threshold':.45,'vlm_threshold':.65,'display_mode':'objects','version':0}

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
 if s.status == 'queued' or s.status == 'running' then return 0 end
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('XADD', KEYS[2], '*', 'event_id', ARGV[1], 'job_id', ARGV[3])
return 1
"""
ENHANCE = """
if redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], ARGV[2], ARGV[3])
redis.call('XADD', KEYS[2], '*', 'detection_id', ARGV[2], 'kind', 'enhanced', 'value', ARGV[3])
redis.call('XADD', KEYS[4], '*', 'kind', 'enhanced', 'value', ARGV[3])
redis.call('SADD', KEYS[3], ARGV[1])
return 1
"""


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
            projected.append({**d,'detection_id':key,'display_label':displayed['label'],'display_confidence':displayed['confidence'],'label_provider':displayed['provider'],'enhancement':candidate,'enhancement_accepted':accepted})
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
        raw=self.redis.hget('rgb:detections',key)
        projection=json.loads(raw)
        projection['candidate']={**candidate,'job_id':job_id}
        projection['decision_settings']=self.settings()
        projection['updated_at']=now()
        self.redis.eval(ENHANCE,4,'rgb:detections','rgb:detection-updates','rgb:enhancement-ids','rgb:detection:'+key+':events',job_id+':'+key,key,json.dumps(projection))

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

    def evaluate(self,key,values,actor=None):
        projection=self.redis.hget('rgb:detections',key)
        if not projection:return None
        projection=json.loads(projection)
        settings=self.settings()
        review={'evaluation_id':str(uuid4()),'detection_id':key,'scene_id':projection['scene_id'],'source_id':projection['source_id'],'reviewed_at':now(),'actor':actor,'initial':projection['initial'],'candidate':projection['candidate'],'settings':settings,'mode':settings['display_mode'],**values}
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
