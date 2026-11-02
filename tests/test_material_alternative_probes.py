import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'material_probes', Path(__file__).resolve().parents[1]/'scripts/probe-material-alternatives.py')
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)


def row(identity, split, material, kind, weight=1):
    return {'id': identity, 'split': split, 'material': material, 'source_kind': kind,
            'target': probes.LABELS.index(material), 'weight': weight}


def test_domain_mass_uses_training_only_and_preserves_explicit_allocations():
    rows = [row('a', 'train', 'plastic', 'MINC'), row('b', 'train', 'plastic', 'MINC'),
            row('c', 'train', 'fabric', 'camera', .6), row('d', 'train', 'human skin', 'camera', .6),
            row('e', 'validation', 'fabric', 'camera'), row('f', 'test', 'fabric', 'camera')]
    allocation = {'MINC': .5, 'camera_material': .4, 'camera_rejection': .1}
    ids, weights = probes.training_weights(rows, allocation)
    assert ids.tolist() == [0, 1, 2, 3]
    assert weights[:2].sum() == pytest.approx(.5)
    assert weights[2:] == pytest.approx([.4, .1])
    rows[4]['target'] = 0; rows[5]['target'] = 5
    assert probes.training_weights(rows, allocation)[1] == pytest.approx(weights)


@pytest.mark.parametrize('method,parameter', [('prototype', None), ('ridge', .001), ('knn', (1, .02))])
def test_classifier_recovers_separable_reference_classes(method, parameter):
    train = np.array([[1., 0], [.99, .01], [0., 1], [.01, .99]])
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    scores = probes.classify(train, np.array([0, 0, 6, 6]), np.full(4, .25),
                             np.eye(2), method, parameter)
    assert scores.argmax(axis=1).tolist() == [0, 6]


def test_metrics_reject_historical_test_and_keep_camera_macro_separate():
    rows = [row('a', 'validation', 'fabric', 'camera'), row('b', 'validation', 'fabric', 'camera'),
            row('c', 'validation', 'plastic', 'camera')]
    scores = np.zeros((3, len(probes.LABELS))); scores[:, 6] = 1
    metrics = probes.validation_metrics(scores, rows)['camera']
    assert metrics['top1'] == pytest.approx(2/3)
    assert metrics['class_macro'] == pytest.approx(.5)
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='validation'):
        probes.validation_metrics(scores, rows)


@pytest.mark.parametrize('failure', ['width', 'split', 'duplicate', 'missing_validation'])
def test_probe_rejects_malformed_feature_manifest(failure):
    rows = [row('a', 'train', 'plastic', 'MINC'), row('b', 'validation', 'fabric', 'camera')]
    features = np.eye(2)
    if failure == 'width':
        features = np.zeros((2, 0))
    elif failure == 'split':
        rows[1]['split'] = 'holdout'
    elif failure == 'duplicate':
        rows[1]['id'] = 'a'
    else:
        rows[1]['split'] = 'train'
    with pytest.raises(ValueError):
        probes.run(features, np.array([r['id'] for r in rows]), rows, {})


def test_cli_independent_feature_override_skips_baselines_and_records_actual_sources(tmp_path, monkeypatch):
    rows = [row('a', 'train', 'plastic', 'MINC'), row('b', 'validation', 'fabric', 'camera')]
    rows_file = tmp_path/'selected-rows.json'
    rows_file.write_text(json.dumps(rows))
    feature_file = tmp_path/'dino-cls.npz'
    np.savez(feature_file, features=np.eye(2), ids=np.array(['a', 'b']))
    output = tmp_path/'receipt.json'
    captured = {}

    def fake_run(features, ids, selected_rows, baselines):
        captured.update(features=features, ids=ids.tolist(), rows=selected_rows, baselines=baselines)
        return {'best_by_validation': {}, 'baselines': {}}

    monkeypatch.setattr(probes, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['probe', '--run-directory', str(tmp_path/'absent-original-run'),
                                    '--feature-file', str(feature_file), '--rows-file', str(rows_file),
                                    '--skip-baselines', '--output', str(output)])
    probes.main()
    receipt = json.loads(output.read_text())
    assert captured['ids'] == ['a', 'b']
    assert captured['rows'] == rows
    assert captured['baselines'] == {}
    assert np.array_equal(captured['features'], np.eye(2))
    assert receipt['inputs'] == {'features.npz': probes.checksum(feature_file),
                                 'evaluation-rows.json': probes.checksum(rows_file)}
    assert receipt['input_paths']['features.npz'] == str(feature_file.resolve())
    rows[1]['split'] = 'test'
    rows_file.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='train/validation'):
        probes.main()
