"""SQLite archive with immutable assets and transactional review projections."""
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from rgb.store import DEFAULT_SETTINGS, Store, StaleEvaluation, candidate_revision, detection_id, now
from rgb.vlm_providers import PRIMARY_MODEL, FALLBACK_MODEL, model_selection

LOCAL_DEFAULTS={**DEFAULT_SETTINGS,'selected_provider':'bedrock','selected_model':PRIMARY_MODEL,'model_selection_reason':'initial',
                'capture_paused':False,'capture_interval_s':60}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class ArchiveConflict(ValueError):
    pass


class QueueFull(ValueError):
    pass


class LocalStore(Store):
    """Semantic storage API. No Redis connection or Redis command emulation."""
    def __init__(self,root,analysis_enabled=True):
        self.directory=Path(root).expanduser().resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.root=self.directory/'assets'
        self.root.mkdir(exist_ok=True)
        self.analysis_enabled=analysis_enabled
        self.database=self.directory/'archive.sqlite3'
        self._local=threading.local()
        self._snapshot_connection=None
        self._write_lock=threading.RLock()
        with self.connection() as db:
            if db.execute('PRAGMA user_version').fetchone()[0]>2:raise ValueError('Archive schema is newer than this application')
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY, frame_id TEXT UNIQUE, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS journal (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL, fields TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS journal_topic ON journal(topic,seq);
                CREATE TABLE IF NOT EXISTS projections (id TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS state (kind TEXT NOT NULL,id TEXT NOT NULL,value TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,status TEXT NOT NULL,policy TEXT NOT NULL,
                    created_at TEXT NOT NULL,progress INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS enhancements (
                    job_id TEXT NOT NULL,detection_id TEXT NOT NULL,value TEXT NOT NULL,
                    PRIMARY KEY(job_id,detection_id));
                CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY,value REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS assets (file TEXT PRIMARY KEY,bytes INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_queued ON jobs(kind,status,created_at);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,status TEXT NOT NULL,started_value TEXT);
                PRAGMA user_version=2;
            """)
            for path in self.root.iterdir():
                if path.is_file() and re.fullmatch(r'[a-f0-9]{64}\.(jpg|png|json)',path.name):
                    db.execute('INSERT OR IGNORE INTO assets VALUES (?,?)',(path.name,path.stat().st_size))
        with self.transaction() as db:
            if not self._state('migration','attempt_index',False,db):
                for _,fields in self.stream('rgb:model-attempts'):
                    self._index_attempt(db,json.loads(fields['value']))
                self._set(db,'migration','attempt_index',True)
        self.recover()

    def connection(self):
        if self._snapshot_connection is not None:return self._snapshot_connection
        if not hasattr(self._local,'db'):
            self._local.db=sqlite3.connect(self.database,timeout=5,isolation_level=None)
            self._local.db.execute('PRAGMA synchronous=FULL')
            self._local.db.execute('PRAGMA foreign_keys=ON')
            self._local.db.execute('PRAGMA busy_timeout=5000')
        return self._local.db

    @contextmanager
    def transaction(self):
        with self._write_lock:
            db=self.connection()
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def _state(self,kind,key,default=None,db=None):
        row=(db or self.connection()).execute('SELECT value FROM state WHERE kind=? AND id=?',(kind,key)).fetchone()
        return json.loads(row[0]) if row else default

    def _set(self,db,kind,key,value):
        db.execute('INSERT INTO state VALUES (?,?,?) ON CONFLICT(kind,id) DO UPDATE SET value=excluded.value',
                   (kind,key,encode(value)))

    def _append(self,db,topic,fields):
        seq=db.execute('INSERT INTO journal(topic,fields) VALUES (?,?)',(topic,encode(fields))).lastrowid
        return f'{seq}-0'

    def _update(self,db,key,kind,value):
        fields={'detection_id':key,'kind':kind,'value':encode(value)}
        self._append(db,'rgb:detection-updates',fields)
        self._append(db,'rgb:detection:'+key+':events',fields)

    def _increment(self,db,key,value=1):
        db.execute('INSERT INTO metrics VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value',
                   (key,value))

    def ping(self):
        return self.connection().execute('SELECT 1').fetchone()[0]==1

    def asset(self,data,suffix):
        digest=hashlib.sha256(data).hexdigest()
        path=self.root/(digest+suffix)
        if not path.exists():
            tmp=self.root/(digest+'.'+uuid4().hex+'.tmp')
            try:
                with tmp.open('xb') as file:
                    file.write(data);file.flush();os.fsync(file.fileno())
                os.replace(tmp,path)
                fd=os.open(self.root,os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            finally:tmp.unlink(missing_ok=True)
        with self._write_lock:
            self.connection().execute('INSERT OR IGNORE INTO assets VALUES (?,?)',(path.name,len(data)))
        return {'sha256':digest,'file':path.name,'bytes':len(data)}

    def settings(self):
        return {**LOCAL_DEFAULTS,**self._state('settings','current',{})}

    def update_settings(self,values,actor=None):
        with self.transaction() as db:
            settings={**LOCAL_DEFAULTS,**self._state('settings','current',{},db)}
            settings.update(values,version=settings['version']+1,changed_at=now(),actor=actor)
            model_selection(settings['selected_provider'],settings['selected_model'])
            self._set(db,'settings','current',settings)
            self._append(db,'rgb:settings-updates',{'kind':'settings','value':encode(settings)})
            return settings

    def location(self,source_id):
        return self._state('location',source_id)

    def update_location(self,source_id,values,actor=None):
        with self.transaction() as db:
            version=self._state('location-version','current',0,db)+1
            self._set(db,'location-version','current',version)
            location={**values,'source_id':source_id,'version':version,'changed_at':now(),'actor':actor}
            self._set(db,'location',source_id,location)
            self._append(db,'rgb:location-updates',{'value':encode(location)})
            return location

    def save(self,event,image):
        event={**event,'image_sha256':self.asset(image,'.jpg')['sha256']}
        with self.transaction() as db:
            existing=db.execute('SELECT value FROM observations WHERE id=? OR frame_id=?',
                                (event['event_id'],event.get('frame_id'))).fetchone()
            if existing:
                original=json.loads(existing[0])
                # Repeated frame capture reuses the first immutable observation.
                if original['event_id']==event['event_id'] and encode(original)!=encode(event):
                    raise ArchiveConflict('Observation ID already has different evidence')
                row=self._state('event-row',original['event_id'],None,db)
                if row is None:
                    sequence=db.execute("SELECT seq FROM journal WHERE topic='rgb:events' AND json_extract(fields,'$.event')=?",
                                        (existing[0],)).fetchone()[0]
                    row=f'{sequence}-0'
                    self._set(db,'event-row',original['event_id'],row)
                return row,False
            encoded=encode(event)
            db.execute('INSERT INTO observations VALUES (?,?,?)',(event['event_id'],event.get('frame_id'),encoded))
            row=self._append(db,'rgb:events',{'event':encoded})
            self._set(db,'event-row',event['event_id'],row)
            self._set(db,'analysis',event['event_id'],{'status':'not_analyzed' if self.analysis_enabled else 'disabled','schema_version':3})
            for index,detection in enumerate(event['detections']):
                key=detection_id(event['event_id'],index)
                initial={'detection_id':key,'scene_id':event['event_id'],'source_id':event['source_id'],'index':index,
                         'observed_at':event['observed_at'],'recorded_at':now(),'initial':detection,'candidate':None}
                db.execute('INSERT INTO projections VALUES (?,?)',(key,encode(initial)))
                self._update(db,key,'initial',initial)
            return row,True

    def event_for_frame(self,frame_id):
        row=self.connection().execute('SELECT id FROM observations WHERE frame_id=?',(frame_id,)).fetchone()
        return self.raw(row[0]) if row else None

    def raw(self,event_id):
        row=self.connection().execute('SELECT value FROM observations WHERE id=?',(event_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def ensure_detections(self,event):
        # Initial detections are committed in the same transaction as the observation.
        return None

    def projection(self,key):
        row=self.connection().execute('SELECT value FROM projections WHERE id=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def opinions(self,event_id):
        return self._state('opinions',event_id,[])

    def analysis(self,event_id):
        return self._state('analysis',event_id,{'status':'not_analyzed','schema_version':3})

    def metrics(self):
        return dict(self.connection().execute('SELECT key,value FROM metrics'))

    def active_attempt(self):
        return self._state('active','current')

    def refinements(self,source_id):
        return self._state('refinements',source_id)

    def event_rows(self,maximum='+',count=501):
        exclusive=maximum.startswith('(')
        seq=min(int(maximum.lstrip('(').split('-')[0]),2**63-1) if maximum not in {'+','-'} else 2**63-1
        rows=self.connection().execute(
            f"SELECT seq,fields FROM journal WHERE topic='rgb:events' AND seq{'<' if exclusive else '<='}? ORDER BY seq DESC LIMIT ?",
            (seq,count)).fetchall()
        return [(f'{number}-0',json.loads(fields)) for number,fields in rows]

    def stream(self,key,stop='+'):
        upper=int(stop.split('-')[0]) if stop!='+' else 2**63-1
        cursor=0
        while True:
            rows=self.connection().execute('SELECT seq,fields FROM journal WHERE topic=? AND seq>? AND seq<=? ORDER BY seq LIMIT 256',
                                           (key,cursor,upper)).fetchall()
            if not rows:return
            for seq,fields in rows:yield f'{seq}-0',json.loads(fields)
            cursor=rows[-1][0]

    def history(self,key):
        rows=list(self.connection().execute('SELECT seq,fields FROM journal WHERE topic=? ORDER BY seq DESC LIMIT 100',
                                           ('rgb:detection:'+key+':events',)))
        return [{'cursor':f'{seq}-0',**json.loads(json.loads(fields)['value']),'kind':json.loads(fields)['kind']}
                for seq,fields in reversed(rows)]

    def request_analysis(self,event_id,kind='manual',policy=None):
        if not self.raw(event_id):return False
        policy=policy or self.settings()
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM jobs WHERE event_id=? AND status IN ('queued','running')",(event_id,)).fetchone():return False
            if kind=='manual' and db.execute("SELECT count(*) FROM jobs WHERE kind='manual' AND status='queued'").fetchone()[0]>=100:
                raise QueueFull('Manual analysis queue is full')
            previous=db.execute("SELECT id,status FROM jobs WHERE event_id=? ORDER BY rowid DESC LIMIT 1",(event_id,)).fetchone()
            if previous and previous[1] in {'partial','failed'}:
                row=db.execute("SELECT fields FROM journal WHERE topic='rgb:model-attempts' AND json_extract(json_extract(fields,'$.value'),'$.job_id')=? ORDER BY seq DESC LIMIT 1",(previous[0],)).fetchone()
                policy={**policy,'_recovery_of':{'job_id':previous[0],'attempt_id':json.loads(json.loads(row[0])['value'])['attempt_id'] if row else None}}
            job_id=str(uuid4())
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,0)',(job_id,event_id,kind,'queued',encode(policy),now()))
            self._set(db,'analysis',event_id,{'status':'queued','schema_version':3,'job_id':job_id,'queued_at':now()})
            return True

    def queued_manual(self):
        row=self.connection().execute("SELECT id,event_id,kind,policy,progress FROM jobs WHERE kind='manual' AND status='queued' ORDER BY created_at LIMIT 1").fetchone()
        return dict(zip(['id','event_id','kind','policy','progress'],[*row[:3],json.loads(row[3]),row[4]])) if row else None

    def job(self,event_id):
        row=self.connection().execute('SELECT id,event_id,kind,policy,progress FROM jobs WHERE event_id=? ORDER BY rowid DESC LIMIT 1',(event_id,)).fetchone()
        return dict(zip(['id','event_id','kind','policy','progress'],[*row[:3],json.loads(row[3]),row[4]])) if row else None

    def set_job(self,job,status,progress=None,error=None):
        event=self.raw(job['event_id'])
        with self.transaction() as db:
            if progress is not None:job['progress']=progress
            db.execute('UPDATE jobs SET status=?,progress=? WHERE id=?',(status,job['progress'],job['id']))
            self._set(db,'analysis',job['event_id'],{'status':status,'schema_version':3,'job_id':job['id'],
                      'completed_boxes':job['progress'],'total_boxes':len(event['detections']),'updated_at':now(),'error':error})

    def record_attempt(self,event,indices,attempt):
        with self.transaction() as db:self._attempt(db,event,indices,attempt)

    def _attempt(self,db,event,indices,attempt):
        attempt.update(scene_id=event['event_id'],source_id=event['source_id'],image_sha256=event['image_sha256'],
                       detection_ids=[detection_id(event['event_id'],index) for index in indices])
        self._append(db,'rgb:model-attempts',{'value':encode(attempt)})
        self._index_attempt(db,attempt)
        self._increment(db,'model_requests' if attempt['outcome']=='started' else 'request_'+attempt['outcome'])
        if attempt['outcome']=='started':self._set(db,'active','current',attempt)
        else:self._set(db,'active','current',None)
        for key in attempt['detection_ids']:self._update(db,key,'model_attempt',attempt)

    def _index_attempt(self,db,attempt):
        db.execute("""INSERT INTO attempts VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,started_value=COALESCE(excluded.started_value,attempts.started_value)""",
                   (attempt['attempt_id'],attempt['outcome'],encode(attempt) if attempt['outcome']=='started' else None))

    def append_validated_readings(self,event,indices,attempt,readings,job):
        """Terminal success and all batch enhancements are one atomic commit."""
        with self.transaction() as db:
            self._attempt(db,event,indices,attempt)
            created=0
            for reading in readings:
                key=detection_id(event['event_id'],reading['box_index'])
                candidate={**reading,'job_id':job['id'],'attempt_id':attempt['attempt_id'],
                           'model':attempt['model'],'endpoint':attempt.get('endpoint','bedrock'),'observed_at':now()}
                projection=self.projection(key)
                projection.update(candidate=candidate,decision_settings=job['policy'],updated_at=now())
                if db.execute('SELECT 1 FROM enhancements WHERE job_id=? AND detection_id=?',(job['id'],key)).fetchone():continue
                db.execute('INSERT INTO enhancements VALUES (?,?,?)',(job['id'],key,encode(projection)))
                created+=1
                db.execute('UPDATE projections SET value=? WHERE id=?',(encode(projection),key))
                self._update(db,key,'enhanced',projection)
            progress=job['progress']+created
            status='complete' if progress==len(event['detections']) else 'queued'
            db.execute('UPDATE jobs SET progress=?,status=? WHERE id=?',(progress,status,job['id']))
            self._set(db,'analysis',event['event_id'],{'status':status,'schema_version':3,'job_id':job['id'],
                      'completed_boxes':progress,'total_boxes':len(event['detections']),'updated_at':now(),'error':None})
        job['progress']=progress

    def enhancement_for_job(self,key,job_id):
        row=self.connection().execute('SELECT value FROM enhancements WHERE job_id=? AND detection_id=?',(job_id,key)).fetchone()
        return json.loads(row[0]) if row else None

    def evaluate(self,key,values,context,actor=None):
        with self.transaction() as db:
            projection=self.projection(key)
            if not projection:return None
            if context.get('candidate_revision')!=candidate_revision(projection['candidate']):
                raise StaleEvaluation('The model reading changed; refresh before evaluating it')
            settings=LOCAL_DEFAULTS if context.get('settings_version')==0 else None
            for _,fields in self.stream('rgb:settings-updates'):
                recorded=json.loads(fields['value'])
                if recorded['version']==context.get('settings_version'):settings=recorded;break
            if settings is None or context.get('mode')!=settings['display_mode']:
                raise StaleEvaluation('The displayed settings could not be resolved')
            review={**values,'evaluation_id':str(uuid4()),'detection_id':key,'scene_id':projection['scene_id'],
                    'source_id':projection['source_id'],'reviewed_at':now(),'actor':actor,'initial':projection['initial'],
                    'candidate':projection['candidate'],'settings':settings,'mode':settings['display_mode'],'review_context':context}
            self._append(db,'rgb:evaluations',{'value':encode(review)})
            self._update(db,key,'evaluation',review)
            return review

    def processing_failure(self,event,job_id,exc):
        with self.transaction() as db:
            value={'job_id':job_id,'scene_id':event['event_id'],'recorded_at':now(),'error_type':type(exc).__name__,'error':str(exc)[:1500]}
            self._increment(db,'processing_failed')
            for index in range(len(event['detections'])):self._update(db,detection_id(event['event_id'],index),'processing_failure',value)

    def recover(self):
        with self.transaction() as db:
            for (value,) in db.execute("SELECT started_value FROM attempts WHERE status='started'").fetchall():
                attempt=json.loads(value)
                attempt.update(outcome='interrupted',finished_at=now(),error='Runner stopped before recording the result')
                self._attempt(db,self.raw(attempt['scene_id']),attempt['box_indices'],attempt)
            for job_id,event_id,progress in db.execute("SELECT id,event_id,progress FROM jobs WHERE status='running' OR status='queued' AND (progress>0 OR kind='live')").fetchall():
                status='partial' if progress else 'failed'
                db.execute('UPDATE jobs SET status=? WHERE id=?',(status,job_id))
                self._set(db,'analysis',event_id,{'status':status,'schema_version':3,'job_id':job_id,
                          'completed_boxes':progress,'error':'interrupted','updated_at':now()})
            self._set(db,'active','current',None)
        for tmp in self.root.glob('*.tmp'):tmp.unlink(missing_ok=True)

    @contextmanager
    def export_snapshot(self,streams):
        import copy
        snapshot=copy.copy(self)
        db=sqlite3.connect(self.database,timeout=5,isolation_level=None)
        snapshot._snapshot_connection=db
        db.execute('BEGIN')
        try:
            stops={key:f"{db.execute('SELECT COALESCE(max(seq),0) FROM journal WHERE topic=?',(key,)).fetchone()[0]}-0" for key in streams}
            yield snapshot,stops,snapshot.settings()
        finally:
            db.rollback();db.close()

    def storage_status(self):
        size=self.connection().execute('SELECT COALESCE(sum(bytes),0) FROM assets').fetchone()[0]
        return {'asset_bytes':size,'free_bytes':shutil.disk_usage(self.directory).free,
                'observations':self.connection().execute('SELECT count(*) FROM observations').fetchone()[0],
                'manual_queue':self.connection().execute("SELECT count(*) FROM jobs WHERE kind='manual' AND status='queued'").fetchone()[0]}
