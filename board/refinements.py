"""Match completed detection values to the current live object."""
from datetime import datetime
from common.geometry import iou


def match(detection,digest,refinements,now_ts,threshold):
    if not refinements:return None
    age=now_ts-datetime.fromisoformat(refinements['observed_at']).timestamp()
    if age<0:return None
    candidates=[]
    for reading in refinements['readings']:
        if reading['class_id']!=detection['class_id']:continue
        if reading['confidence']<threshold and not any(m['confidence']>=threshold for m in reading.get('materials',[])):continue
        if not detection.get('track_id') or reading.get('track_id')!=detection['track_id']:continue
        overlap=iou(detection['box'],reading['box'])
        distance=(int(digest,16)^int(reading['fingerprint'],16)).bit_count()
        candidates.append((overlap,-distance,{**reading,'match_iou':overlap,'match_hash_distance':distance,'match_age_s':age,'match_method':'continuous_track','reference_event_id':refinements['event_id']}))
    return max(candidates,key=lambda c:(c[0],c[1]))[2] if candidates else None
