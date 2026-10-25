import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rgb.export import build_export
from test_identification import recorded, reading
from test_events import event, sources


@pytest.mark.parametrize('format',['parquet','arrow'])
def test_review_tables_roundtrip_and_link_initial_enhancement_failure_evaluation(recorded,format):
    store,event=recorded
    d=event['detections'][0]
    candidate={**reading(),'model':'qwen','attempt_id':'attempt1'}
    store.enhance(event,0,candidate,'job')
    store.record_attempt(event,[0],{'attempt_id':'attempt2','started_at':event['observed_at'],'finished_at':event['observed_at'],'model':'qwen','outcome':'failed','latency_ms':42,'raw_response':'bad json','error_type':'ValueError'})
    store.evaluate(d['detection_id'],{'reference_label':'can','detector_result':'incorrect','vlm_result':'correct','notes':'Pull tab visible'})
    directory,path=build_export(store,format)
    try:
        with zipfile.ZipFile(path) as bundle:
            manifest=json.loads(bundle.read('manifest.json'))
            assert manifest['tables']['detections']['rows']==1
            def read(name):
                raw=pa.BufferReader(bundle.read(name+'.'+format))
                return pq.read_table(raw) if format=='parquet' else pa.ipc.open_file(raw).read_all()
            detections=read('detections').to_pylist()
            assert detections[0]['initial_label']=='bottle'
            assert detections[0]['display_label']=='beverage can'
            assert detections[0]['enhanced_confidence']==.9
            assert detections[0]['analysis_status']=='failed'
            assert detections[0]['observed_at'].tzinfo is not None
            attempts=read('model_attempts').to_pylist()
            assert attempts[0]['outcome']=='failed'
            assert attempts[0]['raw_response']=='bad json'
            reviews=read('evaluations').to_pylist()
            assert reviews[0]['detection_id']==detections[0]['detection_id']
            assert reviews[0]['detector_result']=='incorrect'
            assert read('settings').num_rows==0
            assert 'projection.sqlite3' not in bundle.namelist()
    finally:directory.cleanup()


def test_export_projection_obeys_transactional_watermark(recorded,monkeypatch):
    store,event=recorded
    original_stream=store.stream
    injected=False
    def stream(key,stop='+'):
        nonlocal injected
        if not injected:
            injected=True
            store.enhance(event,0,{**reading(),'model':'qwen'},'after-snapshot')
        yield from original_stream(key,stop)
    monkeypatch.setattr(store,'stream',stream)
    directory,path=build_export(store,'parquet')
    try:
        with zipfile.ZipFile(path) as bundle:
            table=pq.read_table(pa.BufferReader(bundle.read('detections.parquet')))
            assert table.to_pylist()[0]['display_label']=='bottle'
            assert table.to_pylist()[0]['enhanced_label'] is None
        assert store.get(event['event_id'])['detections'][0]['display_label']=='beverage can'
    finally:directory.cleanup()
