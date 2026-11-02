import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location(
    'material_crop_study', Path(__file__).resolve().parents[1]/'scripts/evaluate-material-crop-study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def record(group, strategy, reference, candidate, accepted=None):
    return {
        'group_id': group, 'strategy': strategy, 'reference': reference,
        'reference_kind': 'unresolved' if reference is None else
                          'rejection' if reference in study.REJECTIONS else 'material',
        'teacher_consensus': {'material': reference, 'quality': 'moderate_agreement'},
        'prediction': {'best_candidate': candidate, 'label': accepted,
                       'candidates': [{'label': label} for label in [candidate, 'plastic', 'fabric']]},
    }


def test_paired_comparisons_use_identical_resolved_material_references_only():
    records = [
        record('a', 'stretch', 'plastic', 'metal'),
        record('a', 'letterbox', 'plastic', 'plastic', 'plastic'),
        record('a', 'foreground', 'plastic', 'plastic'),
        record('a', 'interior', 'plastic', 'fabric'),
        record('b', 'stretch', 'plastic', 'plastic', 'plastic'),
        record('b', 'letterbox', 'fabric', 'fabric', 'fabric'),  # Changed teacher target.
        record('b', 'foreground', 'plastic', 'metal'),
        record('b', 'interior', None, 'plastic'),
    ]
    result = study.comparisons(records)
    letterbox = result['paired_against_stretch']['letterbox']
    assert letterbox['group_ids'] == ['a']
    assert letterbox['excluded_different_material_groups'] == 1
    assert letterbox['top1_gained'] == 1 and letterbox['top1_lost'] == 0
    assert letterbox['top1_agreement_delta'] == 1
    foreground = result['paired_against_stretch']['foreground']
    assert foreground['eligible_same_material_groups'] == 2
    assert foreground['top1_gained'] == foreground['top1_lost'] == 1
    assert foreground['top1_agreement_delta'] == 0
    assert result['all_four_resolved_same_material']['group_ids'] == ['a']
    # Independent strategy summaries may look perfect despite an incomparable target.
    assert study.summarize([row for row in records if row['strategy'] == 'letterbox'])['top1_teacher_agreement'] == 1
    interior = result['paired_against_stretch']['interior']
    assert interior['excluded_unresolved_or_rejection_groups'] == 1


def test_unresolved_and_rejection_acceptance_do_not_inflate_reference_precision():
    result = study.summarize([
        record('a', 'stretch', 'plastic', 'plastic', 'plastic'),
        record('b', 'stretch', 'human skin', 'plastic', 'plastic'),
        record('c', 'stretch', None, 'plastic', 'plastic'),
    ])
    assert result['material_references'] == result['rejection_references'] == result['unresolved_references'] == 1
    assert result['top1_teacher_agreement'] == result['top3_teacher_agreement'] == 1
    assert result['coverage'] == 1
    assert result['accepted_with_reference'] == 2
    assert result['accepted_reference_teacher_agreement'] == .5
    assert result['rejection_accepted'] == result['unresolved_accepted'] == 1
    assert study.summarize([])['top1_teacher_agreement'] is None


def test_no_pair_eligible_returns_explicit_empty_denominators():
    records = [record('a', strategy, None, 'plastic') for strategy in study.STRATEGIES]
    result = study.comparisons(records)
    assert result['all_four_resolved_same_material']['groups'] == 0
    for paired in result['paired_against_stretch'].values():
        assert paired['eligible_same_material_groups'] == 0
        assert paired['top1_agreement_delta'] is None


def fixture(tmp_path):
    Image.new('RGB', (24, 24), 'white').save(tmp_path/'scene.jpg')
    rows = []
    for strategy in study.STRATEGIES:
        path = tmp_path/f'{strategy}.jpg'
        Image.new('RGB', (384, 384), 'gray').save(path)
        rows.append({'id': strategy, 'group_id': 'a', 'strategy': strategy,
                     'split': 'evaluation-candidate', 'file': path.name, 'sha256': study.digest(path),
                     'scene_file': 'scene.jpg', 'scene_sha256': study.digest(tmp_path/'scene.jpg')})
    dataset = {'entries': rows}
    labels = {'entries': [{**row, 'teacher_consensus': {'material': 'plastic', 'quality': 'strong_agreement'}}
                          for row in rows]}
    save(tmp_path, dataset, labels)
    return dataset, labels


def save(directory, dataset, labels):
    (directory/'dataset.json').write_text(json.dumps(dataset))
    labels['dataset_sha256'] = study.digest(directory/'dataset.json')
    (directory/'teacher-labels.json').write_text(json.dumps(labels))


def test_study_validates_alignment_evidence_and_complete_group_strategies(tmp_path):
    dataset, labels = fixture(tmp_path)
    loaded = study.load_study(tmp_path)[2]
    assert len(loaded) == 4
    labels['entries'].reverse()
    save(tmp_path, dataset, labels)
    with pytest.raises(ValueError, match='alignment'):
        study.load_study(tmp_path)
    labels['entries'].reverse()
    dataset['entries'][1]['strategy'] = labels['entries'][1]['strategy'] = 'stretch'
    save(tmp_path, dataset, labels)
    with pytest.raises(ValueError, match='Duplicate group/strategy'):
        study.load_study(tmp_path)


def test_study_rejects_label_hash_mismatch_and_pixel_mutation(tmp_path):
    dataset, labels = fixture(tmp_path)
    (tmp_path/'dataset.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        study.load_study(tmp_path)
    save(tmp_path, dataset, labels)
    Image.new('RGB', (384, 384), 'red').save(tmp_path/'stretch.jpg')
    with pytest.raises(ValueError, match='checksum'):
        study.load_study(tmp_path)


def test_study_rejects_symlink_escape_even_with_matching_pixel_hash(tmp_path):
    dataset, labels = fixture(tmp_path)
    external = tmp_path.parent/'outside.jpg'
    Image.new('RGB', (384, 384), 'gray').save(external)
    (tmp_path/'stretch.jpg').unlink()
    (tmp_path/'stretch.jpg').symlink_to(external)
    with pytest.raises(ValueError, match='path or checksum'):
        study.load_study(tmp_path)


def test_report_write_is_private_and_exclusive(tmp_path):
    path = tmp_path/'report.json'
    study.write_exclusive(path, {'value': 1})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        study.write_exclusive(path, {'value': 2})
    assert json.loads(path.read_text()) == {'value': 1}
