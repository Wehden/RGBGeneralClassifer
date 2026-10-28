import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rgb.vlm_providers import PRIMARY_MODEL, ReviewResponse
from rgb import material_agreement


spec = importlib.util.spec_from_file_location(
    'box_agreement_runner', Path(__file__).resolve().parents[1]/'scripts/benchmark-box-agreement.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def inputs(tmp_path):
    frame = tmp_path/'frame.npz'
    np.savez(frame, color_rgb=np.full((40, 60, 3), 120, dtype=np.uint8),
             metadata_json_utf8=np.frombuffer(json.dumps({'frame_id':'test-frame'}).encode(), dtype=np.uint8))
    receipts = []
    for name, label in [('small', 'private-prediction-a'), ('large', 'private-prediction-b')]:
        path = tmp_path/(name+'.json')
        row = {'source_file':str(frame), 'frame_id':'test-frame',
               'detections':[{'box':[.1, .2, .8, .9], 'label':label, 'confidence':.95, 'row_index':0}]}
        path.write_text(json.dumps({'scene_readings':[row]}))
        receipts.append(path)
    return receipts


def test_prepare_deduplicates_exact_pixels_and_blinds_all_sent_content(tmp_path):
    receipts = inputs(tmp_path)
    manifest = runner.prepare(receipts, tmp_path/'private', [0])
    assert len(manifest['cases']) == 1
    assert len(manifest['cases'][0]['predictions']) == 2
    batch = manifest['batches'][0]
    sent = json.dumps({'system':manifest['system'], 'prompt':batch['prompt'],
                       'image_labels':[image['label'] for image in batch['images']]})
    for value in ['private-prediction', 'small', 'large', '.95']:
        assert value not in sent
    assert len(batch['images']) == 2
    for image in batch['images']:
        with Image.open(image['file']) as evidence:
            assert evidence.format == 'JPEG'
    assert runner.prepare(receipts, tmp_path/'private', [0]) == manifest
    Path(batch['images'][0]['file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='Review image changed'):
        runner.prepare(receipts, tmp_path/'private', [0])


def test_mismatched_frame_identity_is_rejected(tmp_path):
    receipts = inputs(tmp_path)
    value = json.loads(receipts[0].read_text())
    value['scene_readings'][0]['frame_id'] = 'different-frame'
    receipts[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.prepare(receipts, tmp_path/'private', [0])


def test_resume_never_replays_successful_or_failed_image_calls(tmp_path, monkeypatch):
    manifest = runner.prepare(inputs(tmp_path), tmp_path/'private', [0])
    requests = []

    class Reviewer:
        def __init__(self, **kwargs):
            pass

        def review(self, request):
            requests.append(request)
            # The first pass fails schema validation; its receipt still prevents replay.
            return ReviewResponse('{}')

    monkeypatch.setattr(runner, 'BedrockProvider', Reviewer)
    output = tmp_path/'private'
    result = runner.run(manifest, output, 'default', 'us-west-2', PRIMARY_MODEL, 1)
    assert len(requests) == 1
    assert requests[0].images[0][1].startswith(b'\xff\xd8')
    assert result['calls_failed_or_incomplete'] == 1
    runner.run(manifest, output, 'default', 'us-west-2', PRIMARY_MODEL, 1)
    assert len(requests) == 2
    result = runner.run(manifest, output, 'default', 'us-west-2', PRIMARY_MODEL, 64)
    assert len(requests) == 2
    assert result['models']['small']['joint_confirmed_rate_all'] == 0
    with pytest.raises(ValueError, match='configuration changed'):
        runner.run(manifest, output, 'other-profile', 'us-west-2', PRIMARY_MODEL, 64)


def test_passes_use_identical_images_without_previous_response_and_unique_consensus(tmp_path, monkeypatch):
    receipts = inputs(tmp_path)
    for path, label in zip(receipts, ['office chair', 'chair']):
        value = json.loads(path.read_text())
        value['scene_readings'][0]['detections'][0]['label'] = label
        path.write_text(json.dumps(value))
    output = tmp_path/'private'
    manifest = runner.prepare(receipts, output, [0])
    requests = []
    response = json.dumps({'boxes':[{'box_id':0, 'object_label':'chair', 'equivalent_names':[],
        'visibility':'clear', 'box_fit':'good', 'dominant_material':'fabric',
        'material_assessment':'clear', 'reason':'The upholstery is clearly visible.'}]})

    class Reviewer:
        def __init__(self, **kwargs):
            pass

        def review(self, request):
            requests.append(request)
            return ReviewResponse(response)

    monkeypatch.setattr(runner, 'BedrockProvider', Reviewer)
    result = runner.run(manifest, output, 'default', 'us-west-2', PRIMARY_MODEL, 64)
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert response not in requests[1].prompt
    assert result['reviewer_consensus_unique_cases']['reviewed_twice'] == 1
    assert result['reviewer_consensus_unique_cases']['both_clear_same_material'] == 1
    for model in ['small', 'large']:
        assert result['models'][model]['detections'] == 1
        assert result['models'][model]['joint_confirmed_rate_all'] == 1
        assert result['per_pass'][0][model]['joint_confirmed'] == 1
        assert result['per_pass'][1][model]['joint_confirmed'] == 1
        assert result['per_scene']['0'][model]['joint_confirmed'] == 1


def test_material_mode_preserves_image_evidence_and_rejects_cross_protocol_resume(tmp_path, monkeypatch):
    receipts = inputs(tmp_path)
    objects = runner.prepare(receipts, tmp_path/'objects', [0])
    materials = runner.prepare(receipts, tmp_path/'materials', [0], material_agreement)
    assert [row['sha256'] for row in objects['batches'][0]['images']] == [
        row['sha256'] for row in materials['batches'][0]['images']]
    assert objects['batches'][0]['prompt'] != materials['batches'][0]['prompt']
    with pytest.raises(ValueError, match='protocol changed'):
        runner.prepare(receipts, tmp_path/'objects', [0], material_agreement)

    class Reviewer:
        def __init__(self, **kwargs):
            pass

        def review(self, request):
            return ReviewResponse(json.dumps({'boxes':[{
                'box_id':0, 'materials':[{'label':'fabric', 'evidence':'Visible textile weave.'}],
                'dominant_material':'fabric', 'material_assessment':'clear',
                'reason':'Fabric dominates the visible target.'}]}))

    monkeypatch.setattr(runner, 'BedrockProvider', Reviewer)
    result = runner.run(materials, tmp_path/'materials', 'default', 'us-west-2', PRIMARY_MODEL, 64,
                        material_agreement)
    assert result['review_kind'] == 'materials'
    assert result['target_dominant_consensus'] == 1
    assert result['calls_recorded'] == 2
