import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    'dino_features', Path(__file__).resolve().parents[1]/'scripts/cache-dinov2-material-features.py')
dino = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dino)


def test_dense_feature_pooling_excludes_cls_and_normalizes_separate_representations():
    tokens = np.array([[[5., 0.], [0., 1.], [0., 3.]]])
    cls, patches = dino.normalized_features(tokens)
    assert cls.tolist() == [[1., 0.]]
    assert patches.tolist() == [[0., 1.]]
    with pytest.raises(ValueError, match='Zero feature'):
        dino.normalized_features(np.zeros((1, 3, 2)))


def test_matched_square_keeps_edge_evidence_and_disables_processor_crop():
    image = np.zeros((20, 100, 3), np.uint8)
    image[:, :10] = (255, 0, 0)
    image[:, -10:] = (0, 255, 0)
    calls = []
    def processor(*, images, **kwargs):
        calls.append(kwargs)
        pixels = np.array(images[0])
        assert pixels.shape == (384, 384, 3)
        assert pixels[:, 0, 0].min() == 255
        assert pixels[:, -1, 1].min() == 255
        return {'pixel_values': np.zeros((1, 3, 384, 384), np.float32)}
    dino.prepare_inputs(processor, [Image.fromarray(image)], 'siglip384')
    assert calls == [{'do_resize': False, 'do_center_crop': False, 'return_tensors': 'pt'}]


def test_canonical_preprocessing_remains_unchanged():
    calls = []
    def processor(**kwargs):
        calls.append(kwargs)
        return {}
    image = Image.new('RGB', (30, 60))
    assert dino.prepare_inputs(processor, [image], 'canonical') == {}
    assert calls == [{'images': [image], 'return_tensors': 'pt'}]


def test_bad_matched_processor_shape_fails_closed():
    def processor(**kwargs):
        return {'pixel_values': np.zeros((1, 3, 224, 224), np.float32)}
    with pytest.raises(ValueError, match='input shape'):
        dino.prepare_inputs(processor, [Image.new('RGB', (30, 60))], 'siglip384')
