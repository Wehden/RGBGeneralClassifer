import importlib.util
from pathlib import Path

import numpy as np
import pytest

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
