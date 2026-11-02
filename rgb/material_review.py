"""Exclude uncertain weak references without inventing labels or moving holdouts."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def document_hash(value):
    """Canonical hash for one original consensus record."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _read(path):
    content = Path(path).read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _identities(entries, name):
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise ValueError(f'Invalid {name} entries')
    result = {row['id']: row for row in entries}
    if len(result) != len(entries):
        raise ValueError(f'Duplicate {name} identity')
    return result


def reviewed_labels(dataset_path, labels_path, review_path):
    """Return a derived label document; source dataset, labels and pixels stay untouched.

    A review can exclude original training rows or flag train/validation rows.
    It cannot assign a material, alter validation targets, change a split, or
    touch an original test row.
    """
    dataset, dataset_sha = _read(dataset_path)
    labels, labels_sha = _read(labels_path)
    review, review_sha = _read(review_path)
    if review.get('schema_version') != 1:
        raise ValueError('Unsupported material review schema')
    if (review.get('source_dataset_sha256') != dataset_sha
            or review.get('source_labels_sha256') != labels_sha
            or labels.get('dataset_sha256') != dataset_sha):
        raise ValueError('Material review source identity mismatch')
    for alias, canonical in [('dataset_sha256', 'source_dataset_sha256'),
                             ('labels_sha256', 'source_labels_sha256')]:
        if alias in review and review[alias] != review[canonical]:
            raise ValueError('Material review source alias mismatch')
    samples = _identities(dataset['entries'], 'dataset')
    annotated = _identities(labels['entries'], 'label')
    if samples.keys() != annotated.keys():
        raise ValueError('Label/sample identity mismatch')
    for identity, sample in samples.items():
        label = annotated[identity]
        if any(sample.get(key) != label.get(key) for key in ['split', 'sha256', 'group_id']):
            raise ValueError('Label/sample provenance mismatch')
    actions = review.get('actions')
    if not isinstance(actions, list):
        raise ValueError('Expected material review actions')
    action_ids = [row.get('id') for row in actions if isinstance(row, dict)]
    if len(action_ids) != len(actions) or len(set(action_ids)) != len(action_ids):
        raise ValueError('Duplicate or invalid material review actions')
    if 'excluded_training_ids' in review:
        excluded = review['excluded_training_ids']
        if (not isinstance(excluded, list) or any(not isinstance(key, str) for key in excluded)
                or len(set(excluded)) != len(excluded)
                or set(excluded) != {row['id'] for row in actions if row.get('decision') == 'exclude'}):
            raise ValueError('Material review exclusion alias mismatch')
    updated = copy.deepcopy(labels)
    derived = {row['id']: row for row in updated['entries']}
    decisions = Counter()
    allowed_keys = {'id', 'split', 'sha256', 'group_id', 'source_consensus_sha256',
                    'decision', 'flags', 'note'}
    for action in actions:
        if set(action)-allowed_keys:
            raise ValueError('Review actions cannot introduce label or split overrides')
        original = annotated.get(action['id'])
        if original is None:
            raise ValueError('Unknown material review sample')
        if original['split'] not in {'train', 'validation'}:
            raise ValueError('Original test/other holdouts cannot be reviewed for training')
        if any(action.get(key) != original.get(key) for key in ['split', 'sha256', 'group_id']):
            raise ValueError('Material review sample provenance mismatch')
        if action.get('source_consensus_sha256') != document_hash(original['teacher_consensus']):
            raise ValueError('Original consensus changed')
        if action.get('decision') not in {'exclude', 'flag'}:
            raise ValueError('Only exclusions or review flags are allowed')
        if action['decision'] == 'exclude' and original['split'] != 'train':
            raise ValueError('Validation targets cannot be excluded or changed')
        flags = action.get('flags')
        if (not isinstance(flags, list) or not flags
                or any(not isinstance(flag, str) or not 0 < len(flag) <= 120 for flag in flags)
                or len(set(flags)) != len(flags)):
            raise ValueError('Expected distinct nonempty material review flags')
        if not isinstance(action.get('note'), str) or not 0 < len(action['note']) <= 2000:
            raise ValueError('Expected material review rationale')
        target = derived[action['id']]
        target['label_review'] = {key: copy.deepcopy(action[key])
                                  for key in ['decision', 'flags', 'note', 'source_consensus_sha256']}
        if action['decision'] == 'exclude':
            target['teacher_consensus'] = {
                **target['teacher_consensus'], 'material': None, 'weight': 0.,
                'quality': 'review_excluded'}
        decisions[f"{original['split']}:{action['decision']}"] += 1
    # Protect complete original test records, not just their labels.
    if any(derived[key] != row for key, row in annotated.items() if row['split'] == 'test'):
        raise ValueError('Original test evidence changed')
    if 'summary' in updated:
        updated['source_summary'] = copy.deepcopy(labels['summary'])
        updated['summary'].update(
            consensus_counts=dict(Counter(row['teacher_consensus']['quality'] for row in updated['entries'])),
            material_counts=dict(Counter(row['teacher_consensus']['material'] for row in updated['entries']
                                         if row['teacher_consensus']['material'] is not None)))
    updated['label_review'] = {
        'schema_version': 1, 'source_dataset_sha256': dataset_sha, 'source_labels_sha256': labels_sha,
        'review_sha256': review_sha, 'actions': len(actions), 'decisions_by_split': dict(decisions),
        'policy': 'Exclude original training weak references or flag train/validation; no validation target changes, relabeling, or split changes',
        'test_evidence_unchanged': True}
    return updated


def write_reviewed_labels(dataset_path, labels_path, review_path, output_path):
    """Write a new file after validating the complete proposed review."""
    result = reviewed_labels(dataset_path, labels_path, review_path)
    target = Path(output_path)
    # Exclusive creation protects immutable prior label documents and receipts.
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result['label_review']
