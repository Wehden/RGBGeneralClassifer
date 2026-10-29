import json

import pytest

from rgb.material_training import consensus, group_split, label_prompt, parse_labels


def label(material='fabric',status='dominant',confidence='strong'):
    return {'id':0,'material':material,'status':status,'confidence':confidence,
            'visible_materials':[material] if material else [],
            'object':'target','evidence':'Visible woven surface.'}


def test_skin_is_a_valid_rejection_label_and_bad_rows_are_isolated():
    skin=label('human skin')
    broken=dict(label(),id=1,material=None)
    parsed,errors=parse_labels(json.dumps({'labels':[skin,broken]}),2)
    assert parsed[0]['material']=='human skin'
    assert 1 in errors and 1 not in parsed


def test_ambiguous_identities_reject_whole_response():
    with pytest.raises(ValueError):
        parse_labels(json.dumps({'labels':[label(),label()]}),2)
    parsed,errors=parse_labels(json.dumps([{'labels':[label()]},{'labels':[dict(label(),id=1)]}]),2)
    assert set(parsed)=={0,1} and errors=={}


def test_uncertainty_and_disagreement_never_become_training_labels():
    for reviews in [[label(None,'uncertain')]*2,[label(),label('wood')],[label(),None],
                    [label(confidence='weak')]*2]:
        assert consensus(reviews)['material'] is None
        assert consensus(reviews)['weight']==0
    assert consensus([label()]*2)['weight']==.6
    assert consensus([label(),label(confidence='moderate')])['weight']==.3


def test_later_session_reserves_entire_approximate_object_group():
    rows=[{'group_id':'shared','session_id':'early'},{'group_id':'shared','session_id':'late'},
          {'group_id':'train-only','session_id':'early'},{'group_id':'middle-only','session_id':'middle'}]
    groups,sessions=group_split(rows,{'early':'2026-10-01','middle':'2026-10-02','late':'2026-10-03'})
    assert groups=={'shared':'test','train-only':'train','middle-only':'validation'}
    assert sessions=={'early':'train','middle':'validation','late':'test'}
    assert group_split([],{'early':'1','late':'2'})[1]=={'early':'train','late':'validation'}


def test_labeler_is_blinded_and_explicit_about_prior_failure_modes():
    text=label_prompt(4)
    assert 'A whiteboard is not paper' in text
    assert 'MUST be returned as labels' in text
    for term in ['YOLO','SigLIP','folk artist','similarity']:
        assert term not in text
