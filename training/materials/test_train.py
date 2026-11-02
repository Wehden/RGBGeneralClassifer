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


def head_fixture(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    output=tmp_path/'output';output.mkdir()
    for file in ['dataset.json','teacher-labels.json','controls.json']:
        (data/file).write_text('{}')
    rows,logits=rows_and_logits()
    for index,row in enumerate(rows):row.update(id=str(index),weight=1,group_id=str(index))
    # Distinct test features make accidental test inference detectable.
    features=np.zeros((len(rows),2),np.float32)
    features[:,0]=1
    features[-4:]=99
    np.savez(output/'features.npz',features=features,ids=np.array([row['id'] for row in rows]))
    np.save(output/'text_embeddings.npy',np.ones((len(train.LABELS),2),np.float32))
    manifest={'features_sha256':train.digest(output/'features.npz'),
              'dataset_sha256':train.digest(data/'dataset.json'),
              'controls_sha256':train.digest(data/'controls.json'),'model_type':'siglip'}
    (output/'feature-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(train,'load_rows',lambda data,with_labels:[dict(row) for row in rows])
    monkeypatch.setattr(train,'source_model',lambda **kw:pytest.fail('Head stage must not load an encoder'))
    return data,output,rows


def test_head_stage_is_cpu_only_validation_screen_and_serializes_state(tmp_path,monkeypatch):
    import torch
    data,output,rows=head_fixture(tmp_path,monkeypatch)
    original=train.nn.Linear.forward
    seen=[]
    def forward(layer,features):
        assert float(features.max())<99  # No historical test features reach classifier.
        seen.append(len(features))
        return original(layer,features)
    monkeypatch.setattr(train.nn.Linear,'forward',forward)
    device=train.setup()
    result=train.train(data,output,torch.device('cpu'),head_only=True)
    assert seen.count(20)==401 and seen.count(32)==42
    assert result['training']['epochs']==0
    assert result['training']['encoder_tensors_changed']==0
    assert set(result['evaluation'])=={'validation','calibration','validation_objective'}
    assert train.digest(output/'head-state.pt')==result['classifier_sha256']
    state=torch.load(output/'head-state.pt',weights_only=True)
    assert state['weight'].shape==(len(train.LABELS),2)
    saved=np.load(output/'head-validation-logits.npz')
    assert len(saved['ids'])==32
    assert saved['ids'].tolist()==[row['id'] for row in rows if row['split']=='validation']
    assert not (output/'evaluation-logits.npz').exists()
    assert not (output/'artifacts').exists()


def test_review_manifest_only_excludes_original_training_ids(tmp_path,monkeypatch):
    data,output,rows=head_fixture(tmp_path,monkeypatch)
    document={'schema_version':1,'labels_sha256':train.digest(data/'teacher-labels.json'),
              'dataset_sha256':train.digest(data/'dataset.json'),'excluded_training_ids':['0','1']}
    path=tmp_path/'review.json';path.write_text(json.dumps(document))
    before=(data/'teacher-labels.json').read_bytes()
    reviewed=[dict(row) for row in rows]
    report=train.apply_review_manifest(reviewed,data,path)
    assert report['excluded_training_ids']==['0','1']
    assert reviewed[0]['split']==reviewed[1]['split']=='excluded-review'
    assert reviewed[20:]==rows[20:]
    assert (data/'teacher-labels.json').read_bytes()==before
    for excluded,pattern in [(['20'],'original training'),(['missing'],'Unknown'),(['0','0'],'Duplicate')]:
        document['excluded_training_ids']=excluded;path.write_text(json.dumps(document))
        with pytest.raises(ValueError,match=pattern):train.apply_review_manifest(rows,data,path)
    document['excluded_training_ids']=['0'];document['labels_sha256']='wrong'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='identity mismatch'):train.apply_review_manifest(rows,data,path)


def test_review_manifest_aliases_and_action_lists_cannot_disagree(tmp_path,monkeypatch):
    data,output,rows=head_fixture(tmp_path,monkeypatch)
    document={'schema_version':1,'labels_sha256':train.digest(data/'teacher-labels.json'),
              'dataset_sha256':train.digest(data/'dataset.json'),'excluded_training_ids':['0'],
              'actions':[{'id':'1','decision':'exclude'}]}
    path=tmp_path/'review.json';path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='action exclusions disagree'):train.apply_review_manifest(rows,data,path)
    document['actions']=[{'id':'0','decision':'exclude'},{'id':'20','decision':'flag'}]
    document['source_labels_sha256']='wrong';path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='aliases disagree'):train.apply_review_manifest(rows,data,path)
