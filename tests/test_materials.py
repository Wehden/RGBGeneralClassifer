import json
from datetime import datetime,timezone
from common.display import value
from board.tracking import Tracker
from rgb.enrichment import validate_readings
from test_identification import reading, recorded
from test_events import event, sources
import pytest


def test_tracks_are_one_to_one_and_expire_after_detection_gap():
    tracker=Tracker()
    def boxes():return [{'class_id':1,'box':[0,0,.4,.4]},{'class_id':1,'box':[.5,.5,.9,.9]}]
    first=tracker.update(boxes(),10)
    ids=[d['track_id'] for d in first]
    moved=boxes();moved[0]['box']=[.01,.01,.41,.41]
    assert [d['track_id'] for d in tracker.update(moved,11)]==ids
    assert len(set(ids))==2
    tracker.update([],12)
    assert [d['track_id'] for d in tracker.update(boxes(),16)]!=ids


def test_material_mode_preserves_object_reading_and_filters_each_material(recorded):
    store,event=recorded
    candidate={**reading(),'materials':[{'label':'plastic','confidence':.9,'reason':'Moulded shell.'},{'label':'metal','confidence':.6,'reason':'Reflective screws.'}]}
    store.enhance(event,0,candidate,'job')
    store.update_settings({'display_mode':'materials'})
    d=store.get(event['event_id'])['detections'][0]
    assert d['display_label']=='plastic'
    assert d['enhancement']['label']=='beverage can'
    assert d['label']=='bottle'
    store.update_settings({'detector_threshold':.4,'vlm_threshold':.5})
    assert store.settings()['display_mode']=='materials'
    assert store.get(event['event_id'])['detections'][0]['display_label']=='plastic + metal'
    store.update_settings({'display_mode':'objects'})
    assert store.get(event['event_id'])['detections'][0]['display_label']=='beverage can'


@pytest.mark.parametrize('materials',[[{'label':'plastic','confidence':float('nan'),'reason':'x'}],[{'label':'metal','confidence':1.1,'reason':'x'}],'plastic',[{'label':'plastic','confidence':.9}],[]])
def test_material_schema_rejects_bad_scores_and_preserves_empty_reading(materials):
    payload={'message':{'content':json.dumps({'observations':[{**reading(),'materials':materials}]})}}
    if materials==[]:
        assert validate_readings(payload,[{'index':0}])[0]['materials']==[]
    else:
        with pytest.raises((ValueError,KeyError)):validate_readings(payload,[{'index':0}])
