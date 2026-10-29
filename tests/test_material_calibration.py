"""Exercise the trainer's NumPy calibration without its GPU training dependencies."""
import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def calibration():
    path = Path(__file__).resolve().parents[1]/'training/materials/train.py'
    source = ast.parse(path.read_text(), filename=str(path))
    names = {'LABELS', 'REJECTIONS', 'softmax', 'metrics', 'calibrate'}
    nodes = [node for node in source.body if (
        isinstance(node, ast.FunctionDef) and node.name in names or
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    namespace = {'np':np, 'Counter':Counter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**{name:namespace[name] for name in names})


def examples(trainer, kind, rejection):
    plastic = trainer.LABELS.index('plastic')
    reject = trainer.LABELS.index(rejection)
    rows = [{'split':'train', 'source_kind':kind, 'material':'plastic', 'target':plastic} for _ in range(3)]
    rows += [{'split':'validation', 'source_kind':kind, 'material':'plastic', 'target':plastic} for _ in range(10)]
    rows += [{'split':'validation', 'source_kind':kind, 'material':rejection, 'target':reject} for _ in range(10)]
    logits = np.zeros((len(rows), len(trainer.LABELS)), dtype=np.float32)
    logits[:, plastic] = 20
    return rows, logits


@pytest.mark.parametrize('kind', ['MINC', 'camera'])
@pytest.mark.parametrize('rejection', ['human skin', 'food', 'unknown'])
def test_accepted_rejections_lower_agreement_and_prevent_qualification(kind, rejection):
    trainer = calibration()
    rows, logits = examples(trainer, kind, rejection)
    result = trainer.metrics(logits, rows, 'validation', threshold=.5)[kind]
    assert result['top1_agreement'] == 1  # Supported-material diagnostic stays separate.
    assert result['accepted_with_reference'] == 20
    assert result['accepted_reference_correct'] == 10
    assert result['accepted_reference_agreement'] == .5
    calibrated = trainer.calibrate(logits, rows)
    assert calibrated['qualified_for_acceptance'] is False
    assert calibrated['minimum_probability'] == 1


def test_correct_rejections_and_unlabeled_crops_are_not_reference_errors():
    trainer = calibration()
    rows, logits = examples(trainer, 'camera', 'human skin')
    logits[-10:] = 0
    logits[-10:, trainer.LABELS.index('human skin')] = 20
    rows.append({'split':'validation', 'source_kind':'camera', 'material':None, 'target':-1})
    logits = np.vstack([logits, logits[0]])
    result = trainer.metrics(logits, rows, 'validation', threshold=.5)['camera']
    assert result['accepted_with_reference'] == 10
    assert result['accepted_reference_agreement'] == 1
    assert result['unreferenced_accepted'] == 1
    assert trainer.calibrate(logits, rows)['qualified_for_acceptance'] is True
