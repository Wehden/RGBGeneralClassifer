import json

import pytest

from rgb.agreement import canonical, matches, parse_review, prompt, score_case, summarize


def review(label='chair', fit='good', visibility='clear', material=None):
    return {'box_id':0, 'object_label':label, 'equivalent_names':[],
            'visibility':visibility, 'box_fit':fit,
            'dominant_material':material,
            'material_assessment':'clear' if material else 'uncertain',
            'reason':'Visible shape supports the object label.'}


def test_conservative_aliases_do_not_reward_specialized_guesses():
    assert canonical('Office chair') == canonical('chair')
    assert matches('office chair', review())
    for predicted, observed in [('jazz artist', 'person'), ('folk artist', 'person'),
                                ('podium', 'desk'), ('television', 'monitor'),
                                ('blackboard', 'whiteboard')]:
        assert not matches(predicted, review(observed))


def test_two_passes_require_both_labels_and_both_good_fits():
    assert score_case('chair', [review(), review('office chair')])['joint_confirmed']
    assert not score_case('chair', [review(), review('chair', fit='loose')])['joint_confirmed']
    assert not score_case('chair', [review(), review('couch')])['label_confirmed']
    assert not score_case('chair', [review(), None])['reviewed_twice']


def test_uncertainty_never_counts_as_confirmation():
    for row in [review(visibility='uncertain'), review('unknown')]:
        score = score_case(row['object_label'], [row, row])
        assert not score['evaluable']
        assert not score['joint_confirmed']
        assert score['reviewer_label_agreement'] is None


def test_primary_denominator_includes_uncertain_and_missing_reviews():
    cases = [{'predictions':[{'model':'test', 'label':'chair'}], 'reviews':rows}
             for rows in [[review(), review()], [review(visibility='uncertain')]*2, [review(), None]]]
    result = summarize(cases)['models']['test']
    assert result['detections'] == 3
    assert result['evaluable'] == 1
    assert result['joint_confirmed_rate_all'] == 1/3
    assert result['joint_confirmed_rate_evaluable'] == 1


def test_parse_exact_box_identities_and_material_certainty():
    assert parse_review(json.dumps({'boxes':[review()]}), 1)[0]['object_label'] == 'chair'
    for rows, count in [([review()], 2), ([review(), review()], 2),
                        ([dict(review(), box_id=True)], 1),
                        ([dict(review(), dominant_material='plastic')], 1),
                        ([dict(review(), box_fit='perfect')], 1)]:
        with pytest.raises(ValueError):
            parse_review(json.dumps({'boxes':rows}), count)


def test_material_requires_two_clear_identical_judgments():
    assert score_case('chair', [review(material='fabric')]*2)['material_consensus'] == 'fabric'
    assert score_case('chair', [review(material='fabric'), review(material='plastic')])['material_consensus'] is None
    assert score_case('chair', [review(material='fabric'), review()])['material_consensus'] is None


def test_review_prompt_hides_detector_predictions_and_prior_reviews():
    text = prompt(4)
    assert 'exact unannotated RGB crop' in text
    assert '0 through 3' in text
    for token in ['YOLO', 'yoloe', 'confidence', 'jazz artist', 'folk artist']:
        assert token not in text
