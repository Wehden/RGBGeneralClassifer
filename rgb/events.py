"""Validate untrusted board observations before persisting or displaying them."""
import base64
import binascii
import io
import json
import math
import re
from datetime import datetime, timezone
from uuid import UUID

from PIL import Image


class InvalidEvent(ValueError):
    pass


def validate_event(data, sources):
    try:
        if not isinstance(data, dict):
            raise ValueError('Expected object')
        event_id = str(UUID(data['event_id']))
        source_id = data['source_id']
        source = sources[source_id]
        observed = datetime.fromisoformat(data['observed_at'].replace('Z', '+00:00'))
        if observed.tzinfo is None or observed.timestamp() > datetime.now(timezone.utc).timestamp()+300:
            raise ValueError('Invalid observation time')
        detections = data['detections']
        if not isinstance(detections, list) or not 1 <= len(detections) <= 100:
            raise ValueError('Expected 1 to 100 detections')
        normalized = []
        for d in detections:
            label, score, box = d['label'], float(d['confidence']), d['box']
            if not isinstance(label, str) or not 1 <= len(label) <= 100 or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detection')
            if len(box) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1 for x in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('Invalid box')
            live = d.get('live_value')
            if live is not None:
                if not isinstance(live,dict) or not isinstance(live['label'],str) or not 1<=len(live['label'])<=120:
                    raise ValueError('Invalid live label')
                UUID(live['detection_id']);UUID(live['reference_event_id'])
                for key,lo,hi in [('confidence',0,1),('match_iou',0,1),('match_hash_distance',0,64),('match_age_s',0,float('inf'))]:
                    if type(live[key]) not in (int,float) or not math.isfinite(live[key]) or not lo<=live[key]<=hi:
                        raise ValueError('Invalid live match')
                if not isinstance(live['model'],str) or len(live['model'])>120:raise ValueError('Invalid live model')
            track_id=str(UUID(d['track_id'])) if d.get('track_id') else None
            normalized.append({'label':label, 'confidence':score, 'box':box, 'class_id':int(d['class_id']),**({'track_id':track_id} if track_id else {}),**({'live_value':live} if live is not None else {})})
        detector = data['detector']
        if not isinstance(detector['model'], str) or len(detector['model']) > 120 or not re.fullmatch('[a-f0-9]{64}', detector['sha256']):
            raise ValueError('Invalid model provenance')
        encoded = data['image_b64']
        if not isinstance(encoded, str) or len(encoded) > 1_500_000:
            raise ValueError('Image too large')
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'JPEG' or image.width*image.height > 2_000_000:
                raise ValueError('Expected bounded JPEG')
            image.verify()
        provenance = {'model':detector['model'], 'sha256':detector['sha256']}
        if 'config' in detector:
            config=detector['config']
            if not isinstance(config,dict) or len(json.dumps(config,allow_nan=False))>65536:raise ValueError('Invalid detector configuration')
            provenance['config']=config
        if 'threshold' in detector:
            if type(detector['threshold']) not in (int,float) or not .05<=detector['threshold']<=.99:
                raise ValueError('Invalid detector threshold')
            provenance.update(threshold=detector['threshold'],settings_version=int(detector.get('settings_version',0)))
        event = {'event_id':event_id, 'observed_at':observed.astimezone(timezone.utc).isoformat(), 'received_at':datetime.now(timezone.utc).isoformat(), 'source_id':source_id, 'location':source.get('location') or None, 'location_kind':'configured' if source.get('location') else 'unknown', 'detector':provenance, 'detections':normalized, 'opinions':[]}
        if source.get('location_record'):
            event['location_record']=source['location_record']
            event['location_kind']=source['location_record']['method']
        return event, raw
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError, binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise InvalidEvent('Invalid observation') from exc
