import copy
import hashlib
import json

import pytest

from rgb.material_review import document_hash, reviewed_labels, write_reviewed_labels


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def documents(tmp_path):
    dataset = {'entries': [
        {'id': split, 'split': split, 'sha256': split*8, 'group_id': split+'-group'}
        for split in ['train', 'validation', 'test']]}
    dataset_path = tmp_path/'dataset.json'
    dataset_sha = write(dataset_path, dataset)
    labels = {'dataset_sha256': dataset_sha, 'entries': [
        {**row, 'teacher_reviews': [{'material': 'fabric'}]*2,
         'teacher_consensus': {'material': 'fabric', 'weight': .6,
                               'quality': 'strong_agreement', 'supported': ['fabric']}}
        for row in dataset['entries']]}
    labels_path = tmp_path/'teacher-labels.json'
    labels_sha = write(labels_path, labels)
    action = {**dataset['entries'][0], 'source_consensus_sha256':
              document_hash(labels['entries'][0]['teacher_consensus']),
              'decision': 'exclude', 'flags': ['conflicting_weak_references'],
              'note': 'No hard target until ambiguous group is independently reviewed.'}
    review = {'schema_version': 1, 'source_dataset_sha256': dataset_sha,
              'source_labels_sha256': labels_sha, 'actions': [action]}
    review_path = tmp_path/'review.json'
    write(review_path, review)
    return dataset_path, labels_path, review_path, dataset, labels, review


def test_exclusions_preserve_source_labels_pixels_splits_and_complete_test_evidence(documents):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    original_dataset, original_labels = dataset_path.read_bytes(), labels_path.read_bytes()
    result = reviewed_labels(dataset_path, labels_path, review_path)
    assert dataset_path.read_bytes() == original_dataset
    assert labels_path.read_bytes() == original_labels
    train, validation, test = result['entries']
    assert train['teacher_reviews'] == labels['entries'][0]['teacher_reviews']
    assert train['teacher_consensus']['material'] is None
    assert train['teacher_consensus']['weight'] == 0
    assert train['teacher_consensus']['quality'] == 'review_excluded'
    assert validation == labels['entries'][1] and test == labels['entries'][2]
    assert [row['split'] for row in result['entries']] == ['train', 'validation', 'test']
    assert result['label_review']['test_evidence_unchanged']


def test_flags_leave_original_targets_and_weights_intact(documents):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    review['actions'][0]['decision'] = 'flag'
    write(review_path, review)
    result = reviewed_labels(dataset_path, labels_path, review_path)
    assert result['entries'][0]['teacher_consensus'] == labels['entries'][0]['teacher_consensus']
    assert result['entries'][0]['label_review']['decision'] == 'flag'


def test_validation_can_only_be_flagged_with_unchanged_target(documents):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    review['actions'][0].update(**dataset['entries'][1], source_consensus_sha256=
                              document_hash(labels['entries'][1]['teacher_consensus']))
    write(review_path, review)
    with pytest.raises(ValueError, match='Validation targets'):
        reviewed_labels(dataset_path, labels_path, review_path)
    review['actions'][0]['decision'] = 'flag'
    write(review_path, review)
    result = reviewed_labels(dataset_path, labels_path, review_path)
    assert result['entries'][1]['teacher_consensus'] == labels['entries'][1]['teacher_consensus']


@pytest.mark.parametrize('field,value', [
    ('dataset_sha256', 'wrong'), ('labels_sha256', 'wrong'),
    ('excluded_training_ids', []), ('excluded_training_ids', ['train', 'train'])])
def test_optional_source_and_exclusion_aliases_cannot_disagree(documents, field, value):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    review[field] = value
    write(review_path, review)
    with pytest.raises(ValueError, match='alias mismatch'):
        reviewed_labels(dataset_path, labels_path, review_path)


@pytest.mark.parametrize('decision', ['exclude', 'flag'])
def test_original_test_samples_cannot_be_excluded_flagged_or_reused_for_training(documents, decision):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    action = review['actions'][0]
    action.update(**dataset['entries'][2], decision=decision,
                  source_consensus_sha256=document_hash(labels['entries'][2]['teacher_consensus']))
    write(review_path, review)
    with pytest.raises(ValueError, match='holdouts'):
        reviewed_labels(dataset_path, labels_path, review_path)


@pytest.mark.parametrize('field,value', [
    ('material', 'plastic'), ('weight', 1.), ('decision', 'relabel'),
    ('split', 'test'), ('sha256', 'changed'), ('source_consensus_sha256', 'changed')])
def test_relabels_and_stale_sample_provenance_are_rejected(documents, field, value):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    review['actions'][0][field] = value
    write(review_path, review)
    with pytest.raises(ValueError):
        reviewed_labels(dataset_path, labels_path, review_path)


def test_source_hashes_bind_review_and_duplicate_actions_are_rejected(documents):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    labels_path.write_bytes(labels_path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='source identity'):
        reviewed_labels(dataset_path, labels_path, review_path)
    review['source_labels_sha256'] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    review['actions'].append(copy.deepcopy(review['actions'][0]))
    write(review_path, review)
    with pytest.raises(ValueError, match='Duplicate'):
        reviewed_labels(dataset_path, labels_path, review_path)


def test_original_split_metadata_must_match_dataset(documents):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    labels['entries'][0]['split'] = 'test'
    review['source_labels_sha256'] = write(labels_path, labels)
    write(review_path, review)
    with pytest.raises(ValueError, match='provenance'):
        reviewed_labels(dataset_path, labels_path, review_path)


def test_new_label_documents_cannot_overwrite_existing_receipts(documents, tmp_path):
    dataset_path, labels_path, review_path, dataset, labels, review = documents
    output = tmp_path/'reviewed.json'
    write_reviewed_labels(dataset_path, labels_path, review_path, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_reviewed_labels(dataset_path, labels_path, review_path, output)
    assert output.read_bytes() == before
