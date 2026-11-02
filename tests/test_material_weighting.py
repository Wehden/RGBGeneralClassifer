from collections import Counter

import numpy as np
import pytest

from test_material_calibration import calibration, examples


def rows():
    trainer=calibration()
    result=[]
    def add(kind,label,weights):
        for weight in weights:
            result.append({'id':str(len(result)),'split':'train','source_kind':kind,
                           'material':label,'target':trainer.LABELS.index(label),'weight':weight})
    add('MINC','fabric',[1])
    add('MINC','plastic',[1]*7)
    add('camera','fabric',[.6,.3])
    add('camera','plastic',[.6])
    add('camera','human skin',[.6]*3)
    add('camera','food',[.6])
    return result


def test_explicit_domain_masses_and_class_balance_preserve_quality_ratios():
    trainer=calibration()
    ids,weights,report=trainer.loss_weights(rows(),'domain-balanced',.2,.1)
    assert report['effective_domain_mass']==pytest.approx({
        'MINC':.7,'camera_material':.2,'camera_rejection':.1})
    assert report['effective_class_mass']['MINC']==pytest.approx({'fabric':.35,'plastic':.35})
    assert report['effective_class_mass']['camera_material']==pytest.approx({'fabric':.1,'plastic':.1})
    assert report['effective_class_mass']['camera_rejection']==pytest.approx({'human skin':.05,'food':.05})
    assert weights[ids.index(8)]/weights[ids.index(9)]==pytest.approx(2)


def test_weighting_statistics_do_not_use_validation_or_test_labels():
    trainer=calibration()
    original=rows()
    expected=trainer.loss_weights(original,'domain-balanced',.4,.1)
    changed=original+[{'split':split,'source_kind':'camera','material':'unknown',
                      'target':14,'weight':float('nan')} for split in ['validation','test']]
    ids,weights,report=trainer.loss_weights(changed,'domain-balanced',.4,.1)
    assert ids==expected[0] and report==expected[2]
    np.testing.assert_array_equal(weights,expected[1])


def test_legacy_weights_reproduce_original_class_count_formula():
    trainer=calibration()
    data=rows()
    ids,weights,report=trainer.loss_weights(data)
    counts=Counter(row['target'] for row in data)
    reference=np.array([row['weight']/np.sqrt(counts[row['target']]) for row in data],np.float32)
    assert ids==list(range(len(data)))
    np.testing.assert_array_equal(weights,reference)
    assert report['requested_domain_mass'] is None


def test_public_only_excludes_camera_from_optimization_and_support_counts():
    trainer=calibration()
    ids,weights,report=trainer.loss_weights(rows(),'public-only')
    assert ids==list(range(8))
    assert report['training_examples']==8
    assert report['effective_domain_mass']=={'MINC':1,'camera_material':0,'camera_rejection':0}
    assert weights.sum()==pytest.approx(1)


def test_zero_quality_training_rows_do_not_receive_domain_mass():
    trainer=calibration()
    data=rows()+[{'split':'train','source_kind':'camera','material':'metal','target':1,'weight':0}]
    ids,weights,report=trainer.loss_weights(data,'domain-balanced')
    assert len(data)-1 not in ids
    assert 'metal' not in report['effective_class_mass']['camera_material']


@pytest.mark.parametrize('material,rejection',[(.9,.2),(-.1,.1),(.2,float('nan')),(float('inf'),0)])
def test_invalid_domain_masses_are_rejected(material,rejection):
    with pytest.raises(ValueError,match='Domain masses'):
        calibration().loss_weights(rows(),'domain-balanced',material,rejection)


def test_positive_domain_mass_requires_examples_in_that_training_domain():
    data=[row for row in rows() if row['source_kind']=='MINC']
    with pytest.raises(ValueError,match='camera_material'):
        calibration().loss_weights(data,'domain-balanced')
    calibration().loss_weights(data,'public-only')


def test_strict_camera_qualification_cannot_borrow_public_reference_count():
    trainer=calibration()
    public,public_logits=examples(trainer,'MINC','human skin')
    # Public rejections are correctly rejected; only one camera target is accepted.
    public_logits[-10:]=0
    public_logits[-10:,trainer.LABELS.index('human skin')]=20
    camera=[{'split':'validation','source_kind':'camera','material':'plastic','target':0}]
    logits=np.vstack([public_logits,public_logits[0]])
    assert trainer.calibrate(logits,public+camera)['qualified_for_acceptance']
    strict=trainer.calibrate(logits,public+camera,minimum_camera_accepted_references=10)
    assert strict['qualified_for_acceptance'] is False
    assert strict['minimum_probability']==1
    assert strict['minimum_camera_accepted_references']==10


def test_top3_class_and_group_metrics_do_not_overweight_repeated_views():
    trainer=calibration()
    data=[{'split':'validation','source_kind':'camera','material':label,
           'target':trainer.LABELS.index(label),'group_id':group}
          for label,group in [('fabric','a'),('fabric','a'),('fabric','a'),('plastic','b')]]
    logits=np.zeros((4,len(trainer.LABELS)),np.float32)
    logits[:3,trainer.LABELS.index('fabric')]=5
    logits[3,trainer.LABELS.index('metal')]=5
    logits[3,trainer.LABELS.index('plastic')]=4
    result=trainer.metrics(logits,data,'validation')['camera']
    assert result['top1_agreement']==.75
    assert result['top3_agreement']==1
    assert result['class_macro_agreement']==.5
    assert result['group_macro_agreement']==.5
    assert result['approximate_groups_evaluated']==2


def test_reviewed_validation_minimum_is_explicit_and_never_uses_test():
    trainer=calibration()
    data=[{'split':'validation','source_kind':'MINC','material':'plastic','target':0} for _ in range(20)]
    data+=[{'split':'validation','source_kind':'camera','material':'fabric','target':6} for _ in range(8)]
    logits=np.zeros((len(data),len(trainer.LABELS)),np.float32)
    logits[:,0]=5
    assert trainer.objective(logits,data)==1
    assert trainer.objective(logits,data,minimum_camera_targets=8)==pytest.approx(.7)
    extra=[{'split':'test','source_kind':'camera','material':'fabric','target':6} for _ in range(100)]
    expanded=np.vstack([logits,np.zeros((100,len(trainer.LABELS)),np.float32)])
    assert trainer.objective(expanded,data+extra,minimum_camera_targets=8)==pytest.approx(.7)
