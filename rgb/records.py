"""Bounded, paginated detection queries and typed page downloads."""
import io
import json
import re
from datetime import datetime, timezone

from common.display import value
from rgb.store import detection_id, now

STATUSES={'manual_only','paused','queued','running','complete','partial','failed','not_analyzed','disabled'}


def query(store, args):
    allowed={'source','location','from','to','label','material','status','limit','before','format'}
    if set(args)-allowed:raise ValueError('Unknown query parameter')
    limit=int(args.get('limit','50'))
    if not 1<=limit<=200:raise ValueError('Limit must be between 1 and 200')
    before=args.get('before','')
    if before and not re.fullmatch(r'\d+-\d+:\d+',before):raise ValueError('Invalid cursor')
    row_id,index=before.rsplit(':',1) if before else ('+', '-1')
    index=int(index)
    def moment(name):
        raw=args.get(name)
        if not raw:return None
        parsed=datetime.fromisoformat(raw.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError('Date filters require a timezone')
        return parsed
    start,end=moment('from'),moment('to')
    if start and end and start>end:raise ValueError('From must precede To')
    status=args.get('status','')
    if status and status not in STATUSES:raise ValueError('Invalid analysis status')
    label=args.get('label','').casefold().strip()
    material=args.get('material','').casefold().strip()
    if any(len(args.get(k,''))>120 for k in ['source','label','material']):raise ValueError('Filter too long')
    settings=store.settings()
    result=[]
    last=None
    scenes=store.event_rows(maximum=row_id,count=501)
    for stream_id,fields in scenes[:500]:
        raw=json.loads(fields['event'])
        event=store.get(raw['event_id'])
        observed=datetime.fromisoformat(raw['observed_at'])
        if not event['detections']:last=stream_id+':0'
        for i,d in enumerate(event['detections']):
            if stream_id==row_id and i<=index:continue
            last=stream_id+':'+str(i)
            if args.get('source') and raw['source_id']!=args['source']:continue
            if start and observed<start or end and observed>end:continue
            if status and event['analysis']['status']!=status:continue
            if args.get('location') and args['location'].casefold() not in (raw.get('location') or '').casefold():continue
            c=d['enhancement'] or {}
            if label and not any(label in str(v).casefold() for v in [d['label'],c.get('label','')]):continue
            if material and not (d.get('label_kind')=='material' and material in d['label'].casefold() or any(material in m['label'].casefold() for m in c.get('materials',[]))):continue
            displayed=value(d,c,settings)
            result.append({'detection_id':detection_id(raw['event_id'],i),'observation_id':raw['event_id'],'track_id':d.get('track_id'),'source_id':raw['source_id'],'location':raw.get('location'),'observed_at':raw['observed_at'],'box':d['box'],'detector_model':raw['detector']['model'],'detector_sha256':raw['detector']['sha256'],'detector_config_json':json.dumps(raw['detector']['config'],sort_keys=True) if 'config' in raw['detector'] else None,'initial_label':d['label'],'initial_label_kind':d.get('label_kind','object'),'initial_class_id':d['class_id'],'initial_confidence':d['confidence'],'enhanced_label':c.get('label'),'enhanced_confidence':c.get('confidence'),'display_label':displayed['label'],'display_confidence':displayed['confidence'],'display_mode':settings['display_mode'],'materials':c.get('materials'),'model':c.get('model'),'attempt_id':c.get('attempt_id'),'analysis_status':event['analysis']['status']})
            result[-1].update(frame_id=raw.get('frame_id'),session_id=raw.get('session_id'),depth=d.get('depth'),
                              camera_name=raw.get('frame_metadata',{}).get('device',{}).get('name'),
                              calibration_sha256=raw.get('frame_metadata',{}).get('calibration_sha256'),
                              sensor_assets=raw.get('sensor_assets'))
            if len(result)==limit:
                return {'records':result,'next_cursor':last,'queried_at':now(),'settings':settings}
    return {'records':result,'next_cursor':last if len(scenes)>500 else None,'queried_at':now(),'settings':settings}


def download(records, format):
    import pyarrow as pa
    import pyarrow.parquet as pq
    material=pa.struct([('label',pa.string()),('confidence',pa.float64()),('reason',pa.string())])
    schema=pa.schema([(key,pa.string()) for key in ['detection_id','observation_id','track_id','source_id','location','detector_model','detector_sha256','detector_config_json']]+[('observed_at',pa.timestamp('us',tz='UTC')),('box',pa.list_(pa.float64(),4)),('initial_label',pa.string()),('initial_label_kind',pa.string()),('initial_class_id',pa.int32()),('initial_confidence',pa.float64()),('enhanced_label',pa.string()),('enhanced_confidence',pa.float64()),('display_label',pa.string()),('display_confidence',pa.float64()),('display_mode',pa.string()),('materials',pa.list_(material)),('model',pa.string()),('attempt_id',pa.string()),('analysis_status',pa.string())])
    from rgb.export import DEPTH
    schema=pa.schema(list(schema)+[('frame_id',pa.string()),('session_id',pa.string()),('camera_name',pa.string()),('calibration_sha256',pa.string()),('depth',DEPTH),('sensor_assets_json',pa.string())])
    rows=[{**r,'observed_at':datetime.fromisoformat(r['observed_at']),'sensor_assets_json':json.dumps(r['sensor_assets']) if r.get('sensor_assets') else None} for r in records]
    table=pa.Table.from_pylist(rows,schema=schema)
    out=io.BytesIO()
    if format=='parquet':pq.write_table(table,out,compression='zstd')
    else:
        with pa.ipc.new_file(out,schema) as writer:writer.write_table(table)
    out.seek(0)
    return out
