"""Verified, commit-pinned transport for exported material student weights."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

REPOSITORY = 'kwehden/rgb-material-student'
FILES = {'vision.xml', 'vision.bin', 'text_embeddings.npy', 'classifier.npz'}
COMMIT = re.compile(r'[0-9a-f]{40}')
SHA256 = re.compile(r'[0-9a-f]{64}')


class ArtifactError(ValueError):
    pass


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactError('Duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError('Invalid artifact JSON') from exc


def validate(directory):
    """Verify the export contract and bytes, without importing inference libraries."""
    directory = Path(directory)
    names = FILES | {'manifest.json'}
    if any((directory/name).is_symlink() or not (directory/name).is_file() for name in names):
        raise ArtifactError('Missing artifact file or unsupported file symlink')
    manifest = read_json(directory/'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ArtifactError('Unsupported manifest schema')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != FILES:
        raise ArtifactError('Expected exactly the four material student artifact checksums')
    for name, digest in files.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest) or checksum(directory/name) != digest:
            raise ArtifactError('Artifact checksum mismatch: '+name)
    policy = manifest.get('classifier')
    if not isinstance(policy, dict) or policy.get('kind') != 'linear_softmax' or policy.get('file') != 'classifier.npz':
        raise ArtifactError('Expected a learned material student classifier')
    for key, low, high in [('temperature', .1, 10), ('minimum_probability', 0, 1), ('minimum_margin', 0, 1)]:
        value = policy.get(key)
        if type(value) not in (int, float) or not low <= value <= high:
            raise ArtifactError('Invalid classifier policy: '+key)
    if type(policy.get('qualified_for_acceptance')) is not bool:
        raise ArtifactError('Missing classifier qualification policy')
    if not all(isinstance(manifest.get(key), str) and manifest[key] for key in ['model', 'revision', 'license']):
        raise ArtifactError('Missing model provenance or license')
    return manifest, {name:checksum(directory/name) for name in sorted(names)}


def publish(directory, api, operation, repo=REPOSITORY):
    """Upload a stable, allowlisted snapshot in one optimistic commit."""
    validate(directory)
    info = api.model_info(repo)
    if not info.private:
        raise ArtifactError('Publishing requires an existing private model repository')
    with tempfile.TemporaryDirectory(prefix='rgb-publish-') as temporary:
        stage = Path(temporary)
        for name in FILES | {'manifest.json'}:
            shutil.copyfile(Path(directory)/name, stage/name)
        manifest, hashes = validate(stage)
        release = {'schema_version':1, 'files':hashes, 'model':manifest['model'],
                   'source_revision':manifest['revision']}
        (stage/'release.json').write_text(json.dumps(release, indent=2)+'\n')
        # Generate a model card from artifact provenance, not the repo's placeholder license.
        card = ('---\nlicense: '+json.dumps(manifest['license'].lower())+'\n---\n\n'
                '# RGB material student\n\n'
                'Exported material classifier for RGB Console. Download by full Hub commit SHA.\n\n'
                'The manifest records upstream model revision, training provenance, export checksums, '
                'and the classifier acceptance policy. Qualification is reported by the artifact; '
                'publishing verifies file integrity and does not rerun model evaluation.\n\n'
                'This repository contains inference artifacts only. Camera images, training datasets, '
                'credentials, and checkpoints are not uploaded.\n')
        (stage/'README.md').write_text(card)
        names = sorted(hashes) + ['release.json', 'README.md']
        commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=info.sha,
                                   commit_message='Publish verified material student artifacts',
                                   operations=[operation(path_in_repo=name, path_or_fileobj=str(stage/name)) for name in names])
        revision = commit.oid
        if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
            raise ArtifactError('Hub did not return a full commit SHA')
        return {'repo':repo, 'revision':revision, 'manifest_sha256':hashes['manifest.json']}


def verify_release(directory, release):
    _, hashes = validate(directory)
    if not isinstance(release, dict) or release.get('schema_version') != 1 or release.get('files') != hashes:
        raise ArtifactError('Release checksums do not match exported artifacts')
    return hashes


def fetch(revision, cache, download, repo=REPOSITORY, activate=False):
    if not COMMIT.fullmatch(revision):
        raise ArtifactError('Download requires a full 40-character lowercase Hub commit SHA')
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache/('rgb-material-student-'+revision)
    if destination.is_symlink():
        raise ArtifactError('Version directory must not be a symlink')
    if destination.exists():
        origin = read_json(destination/'origin.json')
        if origin.get('repo') != repo or origin.get('revision') != revision:
            raise ArtifactError('Existing cache belongs to another repository or revision')
        hashes = verify_release(destination, read_json(destination/'release.json'))
        if origin.get('manifest_sha256') != hashes['manifest.json']:
            raise ArtifactError('Existing cache manifest differs from its receipt')
    else:
        with tempfile.TemporaryDirectory(prefix='.rgb-fetch-', dir=cache) as temporary:
            stage = Path(temporary)/'model'
            stage.mkdir()
            # Download known names only; never trust remote paths or copy arbitrary repo files.
            for name in sorted(FILES | {'manifest.json', 'release.json'}):
                source = download(repo_id=repo, filename=name, revision=revision, repo_type='model')
                shutil.copyfile(source, stage/name)
            hashes = verify_release(stage, read_json(stage/'release.json'))
            origin = {'repo':repo, 'revision':revision, 'manifest_sha256':hashes['manifest.json']}
            (stage/'origin.json').write_text(json.dumps(origin, indent=2)+'\n')
            # Atomic install on the same filesystem; a failed transfer never touches active weights.
            stage.rename(destination)
    if activate:
        link = cache/'siglip2-material-student-384'
        if link.exists() and not link.is_symlink():
            raise ArtifactError('Activation will not replace an existing real model directory')
        temporary = cache/('.rgb-activate-'+os.urandom(8).hex())
        try:
            temporary.symlink_to(destination.resolve(), target_is_directory=True)
            temporary.replace(link)
        finally:
            temporary.unlink(missing_ok=True)
    return {**origin, 'directory':str(destination.resolve()), 'activated':activate}
