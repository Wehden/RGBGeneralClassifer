"""Match completed detection values to the current live object."""
from datetime import datetime
from common.geometry import iou


def match(detection,digest,refinements,now_ts,threshold):
    if not refinements:return None
    age=now_ts-datetime.fromisoformat(refinements['observed_at']).timestamp()
    if not 0<=age<=90:return None
    candidates=[]
    for reading in refinements['readings']:
        if reading['class_id']!=detection['class_id'] or reading['confidence']<threshold:continue
        overlap=iou(detection['box'],reading['box'])
        distance=(int(digest,16)^int(reading['fingerprint'],16)).bit_count()
        if overlap>=.75 and distance<=4:candidates.append((overlap,-distance,{**reading,'match_iou':overlap,'match_hash_distance':distance,'match_age_s':age,'reference_event_id':refinements['event_id']}))
    return max(candidates,key=lambda c:(c[0],c[1]))[2] if candidates else None
