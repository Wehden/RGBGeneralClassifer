import json

import pytest

from rgb.material_agreement import compare, parse_review, prompt, reference, summarize


def review(dominant='fabric', materials=None, assessment='clear'):
    labels = materials if materials is not None else ([dominant] if dominant else [])
    return {'box_id':0, 'materials':[{'label':label, 'evidence':'Visible surface texture.'} for label in labels],
            'dominant_material':dominant, 'material_assessment':assessment,
            'reason':'Surface texture supports the assessment.'}


def case(index, reviews):
    return {'case_id':str(index), 'reviews':reviews}


def prediction(index, best='fabric', accepted=False):
    return {'case_id':str(index), 'best_candidate':best,
            'status':'identified' if accepted else 'uncertain', 'label':best if accepted else None}


def test_material_reference_never_forces_mixed_or_uncertain_into_single_class():
    uncertain = review(None, assessment='uncertain')
    assert reference([uncertain]*2)['dominant'] is None
    mixed = review(None, ['fabric', 'metal'], 'mixed')
    assert reference([mixed]*2)['dominant'] is None
    assert reference([mixed]*2)['supported'] == ['fabric', 'metal']
    assert reference([review(), review('plastic')])['dominant'] is None
    assert not reference([review(), None])['reviewed_twice']


def test_skin_is_outside_target_and_unknown_predictions_do_not_get_credit():
    cases = [case(0, [review('human skin')]*2), case(1, [review()]*2),
             case(2, [review(None, assessment='uncertain')]*2)]
    score = compare(cases, [prediction(0, 'human skin'), prediction(1, 'unknown'), prediction(2, 'unknown')])
    assert score['dominant_references'] == 1
    assert score['supported_references'] == 1
    assert score['top1_dominant_agreement'] == 0
    assert summarize(cases)['out_of_scope_consensus'] == 1


def test_accepted_labels_and_ungated_candidates_have_separate_denominators():
    cases = [case(0, [review()]*2), case(1, [review('paper')]*2),
             case(2, [review(None, assessment='uncertain')]*2)]
    score = compare(cases, [prediction(0), prediction(1, 'metal', accepted=True),
                            prediction(2, 'glass', accepted=True)])
    assert score['acceptance_coverage'] == 2/3
    assert score['top1_dominant_agreement'] == .5
    assert score['accepted_dominant_agreement'] == 0
    assert score['unreferenced_identified'] == 1
    abstaining = compare(cases, [prediction(0), prediction(1, 'paper'), prediction(2, 'unknown')])
    assert abstaining['accepted_dominant_agreement'] is None


def test_any_supported_material_is_a_separate_diagnostic():
    mixed = review(None, ['fabric', 'metal'], 'mixed')
    cases = [case(0, [mixed]*2)]
    score = compare(cases, [prediction(0, 'metal')])
    assert score['dominant_references'] == 0
    assert score['top1_dominant_agreement'] is None
    assert score['top1_supported_agreement'] == 1


def test_parser_rejects_duplicate_ids_and_unsupported_certainty():
    assert parse_review(json.dumps({'boxes':[review()]}), 1)[0]['dominant_material'] == 'fabric'
    malformed = [review(None, ['fabric'], 'mixed'), review(None, ['fabric'], 'uncertain'),
                 review('metal', ['fabric']), review('fabric', ['fabric', 'fabric']),
                 dict(review(), dominant_material=[])]
    for row in malformed:
        with pytest.raises(ValueError):
            parse_review(json.dumps({'boxes':[row]}), 1)
    with pytest.raises(ValueError):
        parse_review(json.dumps({'boxes':[review(), review()]}), 2)


def test_references_and_predictions_must_pair_one_to_one():
    cases = [case(0, [review()]*2)]
    for predictions in [[], [prediction(1)], [prediction(0), prediction(0)]]:
        with pytest.raises(ValueError):
            compare(cases, predictions)


def test_material_prompt_is_blinded_and_distinguishes_printed_depictions():
    text = prompt(4)
    assert 'printed picture of food is paper' in text
    for term in ['SigLIP', 'YOLO', 'predicted label', 'score']:
        assert term not in text
