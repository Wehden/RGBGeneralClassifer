import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')
spec=importlib.util.spec_from_file_location('material_student_train',Path(__file__).with_name('train.py'))
train=importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def rows_and_logits():
    rows=[]
    for split,kind,count in [('train','MINC',20),('validation','MINC',20),
                             ('validation','camera',12),('test','camera',4)]:
        for index in range(count):
            target=index%2
            rows.append({'split':split,'source_kind':kind,'target':target,'material':train.LABELS[target]})
    logits=np.zeros((len(rows),len(train.LABELS)),np.float32)
    for index,row in enumerate(rows):logits[index,row['target']]=8
    return rows,logits


def test_calibration_uses_validation_and_reports_serializable_metrics():
    rows,logits=rows_and_logits()
    calibration=train.calibrate(logits,rows)
    assert calibration['qualified_for_acceptance']
    assert calibration['minimum_probability']<1
    first=dict(calibration)
    logits[-4:]=0  # Changing test predictions must not change validation calibration.
    assert train.calibrate(logits,rows)==first
    json.dumps(train.metrics(logits,rows,'test',calibration['temperature'],calibration['minimum_probability']))


def test_saturated_scores_cannot_accept_an_unqualified_head():
    rows,logits=rows_and_logits()
    logits*=100
    result=train.metrics(logits,rows,'validation',threshold=1)
    assert result['MINC']['accepted']==0
    assert result['camera']['accepted']==0


def test_sparse_classes_remain_unaccepted_even_when_top_choice_matches():
    rows,logits=rows_and_logits()
    for row in rows:
        if row['split']=='train' and row['target']==1:
            row.update(target=-1,material=None)
    result=train.metrics(logits,rows,'test',threshold=.5)
    assert result['camera']['correct_top1']==4
    assert result['camera']['accepted']==2
