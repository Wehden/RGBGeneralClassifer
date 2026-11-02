"""Portable artifact/policy validation; real inference is in the runtime lane."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rgb.materials import MaterialClassifier


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    class Compiled:
        def input(self, index):
            return SimpleNamespace(shape=[1, 3, 2, 2])
        def output(self, index):
            return 0
        def __call__(self, inputs):
            return {0: np.array([[1., 0.]], np.float32)}
    calls = []
    def compile_model(*args):
        calls.append(args)
        return Compiled()
    monkeypatch.setitem(sys.modules, 'openvino', SimpleNamespace(
        __version__='synthetic-unit', Core=lambda: SimpleNamespace(compile_model=compile_model)))
    (tmp_path / 'vision.xml').write_text('synthetic graph')
    (tmp_path / 'vision.bin').write_bytes(b'synthetic weights')
    np.save(tmp_path / 'text_embeddings.npy', np.array([[1., 0.], [0., 1.], [-1., 0.]], np.float32))
    np.savez(tmp_path / 'classifier.npz', weight=np.array([[10, 0], [0, 10], [-10, 0]], np.float32),
             bias=np.zeros(3, np.float32))
    manifest = {
        'model': 'unit-fixture', 'revision': 'fixture-1', 'input_size': 2, 'embedding_size': 2,
        'labels': ['plastic', 'metal', 'unknown'], 'prompts': ['plastic', 'metal', 'unknown'],
        'rejection_labels': ['unknown'], 'minimum_similarity': .2, 'minimum_margin': .05,
        'processor': {'do_resize': True, 'do_rescale': True, 'do_normalize': True,
                      'resample': 2, 'rescale_factor': 1 / 255,
                      'image_mean': [0, 0, 0], 'image_std': [1, 1, 1]},
        'classifier': {'kind': 'linear_softmax', 'file': 'classifier.npz', 'temperature': 1.,
                       'minimum_probability': .8, 'minimum_margin': .05,
                       'qualified_for_acceptance': True},
        'training': {'train_material_counts': {'plastic': 10, 'metal': 10}},
    }
    def write():
        manifest['files'] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                             for name in ['vision.xml', 'vision.bin', 'text_embeddings.npy', 'classifier.npz']}
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    write()
    return tmp_path, manifest, calls, write


@pytest.mark.parametrize('student', [False, True])
def test_portable_material_loader_validates_and_classifies_both_artifact_kinds(artifact, student):
    directory, manifest, calls, write = artifact
    if not student:
        del manifest['classifier']
        write()
        del manifest['files']['classifier.npz']
        (directory / 'manifest.json').write_text(json.dumps(manifest))
    model = MaterialClassifier(directory, threads=3)
    assert len(calls) == 1 and calls[0][2]['INFERENCE_NUM_THREADS'] == 3
    reading = model.classify(np.zeros((12, 18, 3), np.uint8))
    assert reading['label'] == 'plastic'
    assert reading['score_type'] == ('supervised_material_head_score' if student
                                     else 'uncalibrated_image_text_similarity')
    assert reading['input_size'] == 2 and reading['runtime_version'] == 'synthetic-unit'


@pytest.mark.parametrize(('field', 'value'), [
    ('kind', 'pickle'), ('file', 'elsewhere.npz'), ('temperature', 0.),
    ('temperature', 11.), ('minimum_probability', float('nan')),
    ('minimum_margin', -1.), ('qualified_for_acceptance', 1),
])
def test_invalid_student_policies_are_rejected_before_compilation(artifact, field, value):
    directory, manifest, calls, write = artifact
    manifest['classifier'][field] = value
    write()
    with pytest.raises(ValueError, match='classifier policy'):
        MaterialClassifier(directory)
    assert not calls


@pytest.mark.parametrize('problem', ['missing', 'unexpected', 'corrupt'])
def test_material_file_inventory_fails_closed(artifact, problem):
    directory, manifest, calls, _ = artifact
    if problem == 'missing':
        del manifest['files']['vision.bin']
    elif problem == 'unexpected':
        manifest['files']['unexpected.pkl'] = 'a' * 64
    else:
        (directory / 'vision.bin').write_bytes(b'changed')
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='(Missing|Unexpected|checksum)'):
        MaterialClassifier(directory)
    assert not calls


@pytest.mark.parametrize('problem', ['shape', 'nan', 'normalization', 'duplicate', 'empty-label', 'threshold'])
def test_invalid_material_vocabulary_and_acceptance_policy_are_rejected(artifact, problem):
    directory, manifest, calls, write = artifact
    if problem in {'shape', 'nan', 'normalization'}:
        text = np.load(directory / 'text_embeddings.npy', allow_pickle=False)
        if problem == 'shape': text = text[:, :1]
        elif problem == 'nan': text[0, 0] = float('nan')
        else: text[0, 0] = 2.
        np.save(directory / 'text_embeddings.npy', text, allow_pickle=False)
    elif problem == 'duplicate':
        manifest['labels'][1] = 'plastic'
    elif problem == 'empty-label':
        manifest['labels'][1] = ''
    else:
        manifest['minimum_similarity'] = -1
    write()
    with pytest.raises(ValueError, match='(vocabulary|labels|acceptance)'):
        MaterialClassifier(directory)
    assert not calls


@pytest.mark.parametrize('problem', ['shape', 'dtype', 'nan'])
def test_invalid_supervised_weights_are_rejected(artifact, problem):
    directory, _, calls, write = artifact
    weight = np.ones((3, 2), np.float32)
    bias = np.zeros(3, np.float32)
    if problem == 'shape': weight = weight[:, :1]
    elif problem == 'dtype': bias = bias.astype(np.float64)
    else: weight[0, 0] = float('nan')
    np.savez(directory / 'classifier.npz', weight=weight, bias=bias)
    write()
    with pytest.raises(ValueError, match='classifier weights'):
        MaterialClassifier(directory)
    assert not calls
