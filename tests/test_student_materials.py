import numpy as np

from rgb.materials import rank_student


def manifest():
    return {'embedding_size':2,'labels':['plastic','metal','unknown'],
            'rejection_labels':['unknown'],
            'classifier':{'temperature':1.0,'minimum_probability':.8,'minimum_margin':.05,
                          'qualified_for_acceptance':True},
            'training':{'train_material_counts':{'plastic':10,'metal':10}}}


def test_supervised_head_preserves_cosine_evidence_but_gates_on_its_own_score():
    text=np.array([[.1,.9949874],[0,1],[1,0]],np.float32)
    weights=np.array([[8,0],[0,0],[-8,0]],np.float32)
    result=rank_student([1,0],text,weights,np.zeros(3,np.float32),manifest())
    assert result['status']=='identified' and result['label']=='plastic'
    assert result['similarity']==.1
    assert result['score']>.99
    assert result['score_type']=='supervised_material_head_score'


def test_unqualified_or_untrained_classes_never_become_material_identifications():
    m=manifest();weights=np.array([[8,0],[0,0],[-8,0]],np.float32)
    text=np.ones((3,2),np.float32)/np.sqrt(2)
    m['classifier']['qualified_for_acceptance']=False
    assert rank_student([1,0],text,weights,np.zeros(3,np.float32),m)['label'] is None
    m['classifier']['qualified_for_acceptance']=True
    m['training']['train_material_counts']['plastic']=2
    assert rank_student([1,0],text,weights,np.zeros(3,np.float32),m)['label'] is None


def test_student_preserves_unknown_rejection_and_ambiguous_abstention():
    text=np.ones((3,2),np.float32)/np.sqrt(2)
    weights=np.array([[0,0],[0,0],[8,0]],np.float32)
    result=rank_student([1,0],text,weights,np.zeros(3,np.float32),manifest())
    assert result['best_candidate']=='unknown' and result['label'] is None
    weights=np.array([[8,0],[8,0],[-8,0]],np.float32)
    assert rank_student([1,0],text,weights,np.zeros(3,np.float32),manifest())['label'] is None
