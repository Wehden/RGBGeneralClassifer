"""Typed, bounded-batch Arrow and Parquet review exports."""
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rgb.store import detection_id, now
from common.display import value

T=pa.timestamp('us',tz='UTC')
SCHEMAS={
 'observations':pa.schema([('observation_id',pa.string()),('source_id',pa.string()),('observed_at',T),('received_at',T),('location',pa.string()),('location_method',pa.string()),('location_version',pa.int64()),('latitude',pa.float64()),('longitude',pa.float64()),('location_accuracy_m',pa.float64()),('image_sha256',pa.string()),('detector_model',pa.string()),('detector_sha256',pa.string()),('capture_threshold',pa.float64()),('capture_settings_version',pa.int64())]),
 'detections':pa.schema([('detection_id',pa.string()),('track_id',pa.string()),('observation_id',pa.string()),('source_id',pa.string()),('observed_at',T),('display_mode',pa.string()),('box_index',pa.int32()),('ymin',pa.float64()),('xmin',pa.float64()),('ymax',pa.float64()),('xmax',pa.float64()),('initial_label',pa.string()),('initial_confidence',pa.float64()),('enhanced_label',pa.string()),('enhanced_confidence',pa.float64()),('display_label',pa.string()),('display_confidence',pa.float64()),('display_provider',pa.string()),('comparison',pa.string()),('reason',pa.string()),('model',pa.string()),('attempt_id',pa.string()),('vlm_threshold',pa.float64()),('settings_version',pa.int64()),('analysis_status',pa.string()),('live_label',pa.string()),('live_display_mode',pa.string()),('live_match_method',pa.string()),('live_confidence',pa.float64()),('live_reference_detection_id',pa.string()),('live_match_iou',pa.float64()),('live_hash_distance',pa.int32()),('live_age_s',pa.float64())]),
 'material_readings':pa.schema([('detection_id',pa.string()),('track_id',pa.string()),('observation_id',pa.string()),('source_id',pa.string()),('observed_at',T),('material_index',pa.int32()),('label',pa.string()),('confidence',pa.float64()),('reason',pa.string()),('accepted',pa.bool_()),('model',pa.string()),('attempt_id',pa.string())]),
 'location_updates':pa.schema([('stream_id',pa.string()),('source_id',pa.string()),('version',pa.int64()),('changed_at',T),('actor',pa.string()),('method',pa.string()),('label',pa.string()),('latitude',pa.float64()),('longitude',pa.float64()),('accuracy_m',pa.float64())]),
 'legacy_opinions':pa.schema([('observation_id',pa.string()),('source_id',pa.string()),('model',pa.string()),('endpoint',pa.string()),('observed_at',T),('status',pa.string()),('text',pa.string()),('reason',pa.string()),('error',pa.string()),('value_json',pa.string())]),
 'detection_updates':pa.schema([('stream_id',pa.string()),('detection_id',pa.string()),('kind',pa.string()),('value_json',pa.string())]),
 'model_attempts':pa.schema([('stream_id',pa.string()),('attempt_id',pa.string()),('job_id',pa.string()),('observation_id',pa.string()),('detection_ids',pa.list_(pa.string())),('source_id',pa.string()),('model',pa.string()),('endpoint',pa.string()),('started_at',T),('finished_at',T),('latency_ms',pa.float64()),('outcome',pa.string()),('http_status',pa.int32()),('prompt_tokens',pa.int64()),('output_tokens',pa.int64()),('finish_reason',pa.string()),('error_type',pa.string()),('error',pa.string()),('prompt_sha256',pa.string()),('prompt',pa.string()),('raw_response',pa.string()),('value_json',pa.string())]),
 'settings':pa.schema([('stream_id',pa.string()),('version',pa.int64()),('changed_at',T),('actor',pa.string()),('detector_threshold',pa.float64()),('vlm_threshold',pa.float64()),('display_mode',pa.string())]),
 'evaluations':pa.schema([('stream_id',pa.string()),('evaluation_id',pa.string()),('detection_id',pa.string()),('observation_id',pa.string()),('reviewed_at',T),('actor',pa.string()),('mode',pa.string()),('reference_label',pa.string()),('detector_result',pa.string()),('vlm_result',pa.string()),('notes',pa.string()),('value_json',pa.string())])}


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')) if value else None


def rows(store,table,stops,settings,projections):
    if table in {'observations','detections','material_readings','legacy_opinions'}:
        for row,fields in store.stream('rgb:events',stops['rgb:events']):
            raw=json.loads(fields['event'])
            if table=='legacy_opinions':
                opinions=store.redis.hget('rgb:opinions',raw['event_id'])
                for opinion in json.loads(opinions) if opinions else []:
                    if opinion.get('kind')!='box_comparisons':yield {'observation_id':raw['event_id'],'source_id':raw['source_id'],**{k:opinion.get(k) for k in ['model','endpoint','status','text','reason','error']},'observed_at':timestamp(opinion.get('observed_at')),'value_json':json.dumps(opinion)}
            elif table=='observations':
                d=raw['detector']
                yield dict(observation_id=raw['event_id'],source_id=raw['source_id'],observed_at=timestamp(raw['observed_at']),received_at=timestamp(raw['received_at']),location=raw['location'],location_method=raw.get('location_record',{}).get('method'),location_version=raw.get('location_record',{}).get('version'),latitude=raw.get('location_record',{}).get('latitude'),longitude=raw.get('location_record',{}).get('longitude'),location_accuracy_m=raw.get('location_record',{}).get('accuracy_m'),image_sha256=raw['image_sha256'],detector_model=d['model'],detector_sha256=d['sha256'],capture_threshold=d.get('threshold'),capture_settings_version=d.get('settings_version'))
            else:
                event=raw
                for i,d in enumerate(event['detections']):
                    d['detection_id']=detection_id(event['event_id'],i)
                    projection=projections.execute('SELECT candidate,status FROM projection WHERE detection_id=?',(d['detection_id'],)).fetchone()
                    c=json.loads(projection[0]) if projection and projection[0] else {}
                    displayed=value(d,c,settings)
                    if table=='material_readings':
                        for j,m in enumerate(c.get('materials',[])):
                            yield dict(detection_id=d['detection_id'],track_id=d.get('track_id'),observation_id=event['event_id'],source_id=event['source_id'],observed_at=timestamp(event['observed_at']),material_index=j,label=m['label'],confidence=m['confidence'],reason=m['reason'],accepted=m['confidence']>=settings['vlm_threshold'],model=c.get('model'),attempt_id=c.get('attempt_id'))
                        continue
                    yield dict(detection_id=d['detection_id'],track_id=d.get('track_id'),display_mode=settings['display_mode'],observation_id=event['event_id'],source_id=event['source_id'],observed_at=timestamp(event['observed_at']),box_index=i,ymin=d['box'][0],xmin=d['box'][1],ymax=d['box'][2],xmax=d['box'][3],initial_label=d['label'],initial_confidence=d['confidence'],enhanced_label=c.get('label'),enhanced_confidence=c.get('confidence'),display_label=displayed['label'],display_confidence=displayed['confidence'],display_provider=displayed['provider'],comparison=c.get('comparison'),reason=c.get('reason'),model=c.get('model'),attempt_id=c.get('attempt_id'),vlm_threshold=settings['vlm_threshold'],settings_version=settings['version'],analysis_status=projection[1] if projection else 'not_analyzed',live_label=d.get('live_value',{}).get('display_label',d.get('live_value',{}).get('label')),live_display_mode=d.get('live_value',{}).get('display_mode'),live_match_method=d.get('live_value',{}).get('match_method'),live_confidence=d.get('live_value',{}).get('display_confidence',d.get('live_value',{}).get('confidence')),live_reference_detection_id=d.get('live_value',{}).get('detection_id'),live_match_iou=d.get('live_value',{}).get('match_iou'),live_hash_distance=d.get('live_value',{}).get('match_hash_distance'),live_age_s=d.get('live_value',{}).get('match_age_s'))
    else:
        key={'detection_updates':'rgb:detection-updates','model_attempts':'rgb:model-attempts','settings':'rgb:settings-updates','evaluations':'rgb:evaluations','location_updates':'rgb:location-updates'}[table]
        for row,fields in store.stream(key,stops[key]):
            v=json.loads(fields['value'])
            if table=='detection_updates':yield {'stream_id':row,'detection_id':fields['detection_id'],'kind':fields['kind'],'value_json':fields['value']}
            elif table=='model_attempts':
                yield {**{k:v.get(k) for k in ['attempt_id','job_id','detection_ids','source_id','model','endpoint','latency_ms','outcome','http_status','prompt_tokens','output_tokens','finish_reason','error_type','error','prompt_sha256','prompt','raw_response']},'stream_id':row,'observation_id':v.get('scene_id'),'started_at':timestamp(v.get('started_at')),'finished_at':timestamp(v.get('finished_at')),'value_json':fields['value']}
            elif table=='location_updates':yield {'stream_id':row,**{k:v.get(k) for k in ['source_id','version','actor','method','label','latitude','longitude','accuracy_m']},'changed_at':timestamp(v.get('changed_at'))}
            elif table=='settings':yield {'stream_id':row,**{k:v.get(k) for k in ['version','actor','detector_threshold','vlm_threshold','display_mode']},'changed_at':timestamp(v.get('changed_at'))}
            else:yield {'stream_id':row,**{k:v.get(k) for k in ['evaluation_id','detection_id','actor','mode','reference_label','detector_result','vlm_result','notes']},'observation_id':v['scene_id'],'reviewed_at':timestamp(v['reviewed_at']),'value_json':fields['value']}


def build_export(store,format):
    if format not in {'parquet','arrow'}:raise ValueError('Unknown export format')
    directory=tempfile.TemporaryDirectory(prefix='rgb-export-')
    root=Path(directory.name)
    streams=['rgb:events','rgb:detection-updates','rgb:model-attempts','rgb:settings-updates','rgb:evaluations','rgb:location-updates']
    with store.redis.pipeline(transaction=True) as pipe:
        for key in streams:pipe.xrevrange(key,count=1)
        pipe.get('rgb:settings')
        snapshot=pipe.execute()
    stops={key:(result or [('0-0',{})])[0][0] for key,result in zip(streams,snapshot[:-1])}
    from rgb.store import DEFAULT_SETTINGS
    settings={**DEFAULT_SETTINGS,**(json.loads(snapshot[-1]) if snapshot[-1] else {})}
    projections=sqlite3.connect(root/'projection.sqlite3')
    projections.execute('CREATE TABLE projection (detection_id TEXT PRIMARY KEY,candidate TEXT,status TEXT)')
    for _,fields in store.stream('rgb:detection-updates',stops['rgb:detection-updates']):
        key=fields['detection_id']; value=json.loads(fields['value']); kind=fields['kind']
        if kind=='initial':projections.execute('INSERT OR IGNORE INTO projection VALUES (?,NULL,?)',(key,'queued'))
        elif kind=='enhanced':projections.execute('INSERT INTO projection VALUES (?,?,?) ON CONFLICT(detection_id) DO UPDATE SET candidate=excluded.candidate,status=excluded.status',(key,json.dumps(value['candidate']),'complete'))
        elif kind=='processing_failure':projections.execute('UPDATE projection SET status=? WHERE detection_id=?',('failed',key))
        elif kind=='model_attempt':projections.execute('UPDATE projection SET status=? WHERE detection_id=?',('read_complete' if value['outcome']=='success' else 'failed',key))
    projections.commit()
    manifest={'schema_version':4,'exported_at':now(),'format':format,'stream_watermarks':stops,'settings':settings,'tables':{}}
    try:
        for name,schema in SCHEMAS.items():
            path=root/(name+'.'+format)
            writer=pq.ParquetWriter(path,schema,compression='zstd') if format=='parquet' else pa.ipc.new_file(str(path),schema,options=pa.ipc.IpcWriteOptions(compression='zstd'))
            count=0
            batch=[]
            try:
                for row in rows(store,name,stops,settings,projections):
                    batch.append(row)
                    if len(batch)==256:
                        writer.write_table(pa.Table.from_pylist(batch,schema=schema));count+=len(batch);batch=[]
                if batch:writer.write_table(pa.Table.from_pylist(batch,schema=schema));count+=len(batch)
            finally:writer.close()
            manifest['tables'][name]={'file':path.name,'rows':count,'schema':str(schema)}
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        archive=root/'rgb-review.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as bundle:
            for file in [root/'manifest.json']+[root/(name+'.'+format) for name in SCHEMAS]:bundle.write(file,file.name)
        projections.close()
        return directory,archive
    except Exception:
        projections.close()
        directory.cleanup()
        raise
