import base64
import io
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from PIL import Image
from rgb.events import InvalidEvent, validate_event


@pytest.fixture
def sources():
    return {'portentauno':{'url':'http://camera:8080','location':'Teaching bench'}}


@pytest.fixture
def event():
    image = io.BytesIO()
    Image.new('RGB',(32,32),'green').save(image,'JPEG')
    return {'event_id':str(uuid4()),'source_id':'portentauno','observed_at':datetime.now(timezone.utc).isoformat(),'location':'Invented GPS','detector':{'model':'test','sha256':'a'*64},'detections':[{'label':'bottle','class_id':44,'confidence':.85,'box':[.1,.2,.8,.9]}],'image_b64':base64.b64encode(image.getvalue()).decode()}


def test_configured_location_overrides_board_claim(event,sources):
    result, image = validate_event(event,sources)
    assert result['location'] == 'Teaching bench'
    assert result['location_kind'] == 'configured'
    assert result['sorting_status'] == 'unassessed'
    assert result['opinions'] == []
    assert image.startswith(b'\xff\xd8')


@pytest.mark.parametrize('patch',[{'source_id':'unknown'},{'image_b64':'not base64'},{'observed_at':'2026-01-01'}, {'detector':{'model':'test','sha256':'../secret'}}, {'detections':[{'label':'bottle','class_id':1,'confidence':float('nan'),'box':[0,0,1,1]}]}, {'detections':[{'label':'bottle','class_id':1,'confidence':.9,'box':[.9,0,.1,1]}]}])
def test_invalid_observations(event,sources,patch):
    event.update(patch)
    with pytest.raises(InvalidEvent):
        validate_event(event,sources)
