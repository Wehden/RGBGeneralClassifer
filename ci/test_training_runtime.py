"""CPU train -> OpenVINO export -> console classifier, with a tiny real SigLIP."""
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel

from rgb.materials import MaterialClassifier, prepare

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('actual_material_trainer', ROOT / 'training/materials/train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


@pytest.fixture
def tiny_student():
    trainer.setup()
    vision = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=16, patch_size=8)).vision_model
    head = torch.nn.Linear(16, len(trainer.LABELS))
    return vision, head


def test_real_siglip_cpu_training_updates_the_backbone_and_head(tiny_student):
    vision, head = tiny_student
    pixels = torch.randn(4, 3, 16, 16)
    targets = torch.tensor([0, 0, 1, 1])
    before = {name: value.detach().clone() for name, value in vision.named_parameters()}
    old_head = head.weight.detach().clone()
    optimizer = torch.optim.AdamW([*vision.parameters(), *head.parameters()], lr=.01)
    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        features = torch.nn.functional.normalize(vision(pixel_values=pixels).pooler_output, dim=-1)
        loss = torch.nn.functional.cross_entropy(head(features), targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in vision.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert not torch.equal(old_head, head.weight)
    assert any(not torch.equal(before[name], value) for name, value in vision.named_parameters())


@pytest.mark.parametrize('qualified', [True, False])
def test_actual_trainer_export_is_loadable_and_gated_by_console(tmp_path, tiny_student, qualified):
    vision, head = tiny_student
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[0] = 12
    text = np.random.default_rng(17).normal(size=(len(trainer.LABELS), 16)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    config = {'size': {'height': 16, 'width': 16}, 'do_resize': True,
              'do_rescale': True, 'do_normalize': True, 'do_center_crop': False,
              'resample': 3, 'rescale_factor': 1 / 255,
              'image_mean': [.5] * 3, 'image_std': [.5] * 3}
    images = [np.random.default_rng(seed).uniform(-1, 1, (3, 16, 16)).astype(np.float32)
              for seed in range(3)]
    trainer.export_artifact(vision, head, config, text, {
        'temperature': 1., 'minimum_probability': .8, 'minimum_margin': .05,
        'qualified_for_acceptance': qualified,
    }, tmp_path / 'student', {
        'prompts': trainer.LABELS, 'train_material_counts': {'plastic': 10},
    }, images)
    directory = tmp_path / 'student'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['export_parity']['images'] == 3
    assert manifest['export_parity']['maximum_embedding_absolute_error'] < 1e-4
    assert set(manifest['files']) == {'vision.xml', 'vision.bin', 'text_embeddings.npy', 'classifier.npz'}
    classifier = MaterialClassifier(directory)
    result = classifier.classify(np.full((30, 40, 3), 128, np.uint8))
    assert result['label'] == ('plastic' if qualified else None)
    assert result['score_type'] == 'supervised_material_head_score'


def test_trainer_and_console_use_identical_384_pixel_preprocessing(tmp_path):
    from PIL import Image
    rgb = np.random.default_rng(4).integers(0, 256, (137, 293, 3), dtype=np.uint8)
    path = tmp_path / 'crop.png'
    Image.fromarray(rgb).save(path)
    processor = {'do_resize': True, 'do_rescale': True, 'do_normalize': True,
                 'do_center_crop': False, 'resample': 3, 'rescale_factor': 1 / 255,
                 'image_mean': [.5] * 3, 'image_std': [.5] * 3}
    training = trainer.prepare_image(path, 384, processor)
    deployed = prepare(rgb, {'processor': processor, 'input_size': 384})[0]
    np.testing.assert_array_equal(training, deployed)
    assert deployed.shape == (3, 384, 384)


@pytest.fixture
def training_data(tmp_path, tiny_student):
    from PIL import Image
    vision, _ = tiny_student
    data, output = tmp_path / 'data', tmp_path / 'output'
    data.mkdir()
    output.mkdir()
    rows, images = [], []
    processor = {'size': {'height': 16, 'width': 16}, 'do_resize': True,
                 'do_rescale': True, 'do_normalize': True, 'do_center_crop': False,
                 'resample': 3, 'rescale_factor': 1 / 255,
                 'image_mean': [.5] * 3, 'image_std': [.5] * 3}
    for split in ['train', 'validation', 'test']:
        for material in ['plastic', 'metal']:
            for index in range(3):
                identity = f'{split}-{material}-{index}'
                pixels = np.random.default_rng(len(rows)).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                file = identity + '.png'
                Image.fromarray(pixels).save(data / file)
                rows.append({'id': identity, 'file': file, 'sha256': trainer.digest(data / file),
                             'material': material, 'weight': 1., 'quality': 'synthetic',
                             'source_kind': 'MINC', 'split': split, 'group_id': identity})
                images.append(trainer.prepare_image(data / file, 16, processor))
    trainer.write(data / 'dataset.json', {'entries': []})
    trainer.write(data / 'controls.json', {'entries': rows})
    trainer.write(data / 'teacher-labels.json', {
        'dataset_sha256': trainer.digest(data / 'dataset.json'), 'entries': []})
    features = trainer.features(vision, np.stack(images), torch.device('cpu'))
    np.savez(output / 'features.npz', ids=np.array([row['id'] for row in rows]), features=features)
    text = np.random.default_rng(24).normal(size=(len(trainer.LABELS), 16)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    np.save(output / 'text_embeddings.npy', text, allow_pickle=False)
    trainer.write(output / 'feature-manifest.json', {
        'features_sha256': trainer.digest(output / 'features.npz'),
        'dataset_sha256': trainer.digest(data / 'dataset.json'),
        'controls_sha256': trainer.digest(data / 'controls.json'),
        'model_type': 'tiny-synthetic-siglip', 'input_size': 16})
    return data, output, vision, processor


def test_production_training_pipeline_exports_both_variants_without_test_leakage(training_data, monkeypatch):
    data, output, vision, processor = training_data
    monkeypatch.setattr(trainer, 'source_model', lambda **kwargs: (
        SimpleNamespace(vision_model=deepcopy(vision)), None, processor))
    trainer.train(data, output, torch.device('cpu'), epochs=1, weighting='public-only',
                  validation_only=True)
    results = json.loads((output / 'results.json').read_text())
    assert results['training']['encoder_tensors_changed'] > 0
    assert results['training']['evaluation_scope']['historical_test_rows_omitted'] == 6
    assert all('test' not in result for result in results['evaluations'].values())
    with np.load(output / 'evaluation-logits.npz', allow_pickle=False) as logits:
        assert len(logits['ids']) == 12 and not any(value.startswith('test-') for value in logits['ids'])
    for variant in ['frozen-head', 'fine-tuned']:
        classifier = MaterialClassifier(output / 'artifacts' / variant)
        assert classifier.manifest['training']['variant'] == variant
        reading = classifier.classify(np.full((16, 16, 3), 128, np.uint8))
        assert reading['status'] in {'identified', 'uncertain'}


def test_production_head_only_experiment_omits_test_predictions(training_data):
    data, output, _, _ = training_data
    results = trainer.train(data, output, torch.device('cpu'), head_only=True,
                            validation_only=True, weighting='public-only')
    assert results['training']['head_steps'] == 401
    assert results['training']['epochs'] == 0
    assert results['training']['train_material_counts'] == {'plastic': 3, 'metal': 3}
    with np.load(output / 'head-validation-logits.npz', allow_pickle=False) as logits:
        assert logits['logits'].shape == (6, len(trainer.LABELS))
        assert all(value.startswith('validation-') for value in logits['ids'])


@pytest.mark.parametrize('corruption', ['features', 'dataset', 'pixels'])
def test_training_rejects_changed_features_dataset_or_pixels(training_data, corruption):
    data, output, _, _ = training_data
    file = (output / 'features.npz' if corruption == 'features' else
            data / 'dataset.json' if corruption == 'dataset' else data / 'train-plastic-0.png')
    with file.open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError, match='(identity mismatch|checksum changed|dataset changed)'):
        trainer.train(data, output, torch.device('cpu'), head_only=True)
