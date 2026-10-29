"""Registry transport checks without uploading fixtures or contacting the Hub."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rgb.model_artifacts import ArtifactError, FILES, checksum, fetch, publish, validate

SHA = 'a'*40


@pytest.fixture
def artifact(tmp_path):
    directory = tmp_path/'export'
    directory.mkdir()
    for name in FILES:
        (directory/name).write_bytes(('test-'+name).encode())
    manifest = {'schema_version':1, 'model':'test-student', 'revision':'upstream-revision',
                'license':'Apache-2.0', 'files':{name:checksum(directory/name) for name in FILES},
                'classifier':{'kind':'linear_softmax', 'file':'classifier.npz', 'temperature':1,
                              'minimum_probability':.8, 'minimum_margin':.05,
                              'qualified_for_acceptance':False}}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    return directory


class Hub:
    def __init__(self, private=True):
        self.private = private
        self.uploads = {}
        self.kwargs = None

    def model_info(self, repo):
        return SimpleNamespace(private=self.private, sha='b'*40)

    def create_commit(self, **kwargs):
        self.kwargs = kwargs
        self.uploads = {op.path_in_repo:Path(op.path_or_fileobj).read_bytes() for op in kwargs['operations']}
        return SimpleNamespace(oid=SHA)


def served_release(artifact, tmp_path):
    hub = Hub()
    receipt = publish(artifact, hub, SimpleNamespace)
    remote = tmp_path/'remote'
    remote.mkdir()
    for name, data in hub.uploads.items():
        (remote/name).write_bytes(data)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs['revision'] == receipt['revision']
        return remote/kwargs['filename']
    return remote, calls, download


def test_publish_is_atomic_allowlisted_private_and_retains_qualification(artifact):
    (artifact/'HFToken.key').write_text('must not upload')
    (artifact/'private-camera.jpg').write_bytes(b'private image')
    hub = Hub()
    receipt = publish(artifact, hub, SimpleNamespace)
    assert receipt['revision'] == SHA
    assert hub.kwargs['parent_commit'] == 'b'*40
    assert set(hub.uploads) == FILES | {'manifest.json', 'release.json', 'README.md'}
    assert json.loads(hub.uploads['manifest.json'])['classifier']['qualified_for_acceptance'] is False
    assert 'apache-2.0' in hub.uploads['README.md'].decode()
    assert 'must not upload' not in str(hub.uploads)
    public = Hub(private=False)
    with pytest.raises(ArtifactError, match='private'):
        publish(artifact, public, SimpleNamespace)
    assert public.kwargs is None


@pytest.mark.parametrize('damage', ['corrupt', 'extra-path', 'symlink', 'missing', 'policy'])
def test_invalid_exports_never_reach_upload(artifact, damage):
    manifest = json.loads((artifact/'manifest.json').read_text())
    if damage == 'corrupt':
        (artifact/'vision.bin').write_bytes(b'corrupt')
    elif damage == 'extra-path':
        manifest['files']['../secret'] = 'f'*64
    elif damage == 'symlink':
        (artifact/'vision.bin').rename(artifact/'real.bin')
        (artifact/'vision.bin').symlink_to(artifact/'real.bin')
    elif damage == 'missing':
        (artifact/'classifier.npz').unlink()
    else:
        manifest['classifier']['temperature'] = float('nan')
    (artifact/'manifest.json').write_text(json.dumps(manifest))
    hub = Hub()
    with pytest.raises(ArtifactError):
        publish(artifact, hub, SimpleNamespace)
    assert hub.kwargs is None


def test_pinned_fetch_installs_verified_cache_and_opt_in_activation(artifact, tmp_path):
    remote, calls, download = served_release(artifact, tmp_path)
    cache = tmp_path/'cache'
    result = fetch(SHA, cache, download)
    target = Path(result['directory'])
    assert not (cache/'siglip2-material-student-384').exists()
    assert set(c['filename'] for c in calls) == FILES | {'manifest.json', 'release.json'}
    assert all(c['repo_type'] == 'model' for c in calls)
    assert result['manifest_sha256'] == checksum(remote/'manifest.json')
    def offline(**kwargs):
        raise AssertionError('Verified cached revision should not need a network call')
    again = fetch(SHA, cache, offline, activate=True)
    assert again['revision'] == SHA
    assert (cache/'siglip2-material-student-384').resolve() == target


@pytest.mark.parametrize('revision', ['main', 'latest', 'v1', SHA[:12], '../bad', 'A'*40])
def test_fetch_refuses_unpinned_revisions_before_downloading(tmp_path, revision):
    with pytest.raises(ArtifactError, match='full 40-character'):
        fetch(revision, tmp_path/'cache', lambda **kwargs:pytest.fail('network accessed'))


@pytest.mark.parametrize('failure', ['transfer', 'bytes', 'release'])
def test_failed_fetch_preserves_previous_active_model(artifact, tmp_path, failure):
    remote, _, original = served_release(artifact, tmp_path)
    cache = tmp_path/'cache'
    cache.mkdir()
    previous = cache/'previous'
    previous.mkdir()
    link = cache/'siglip2-material-student-384'
    link.symlink_to(previous)
    if failure == 'bytes':
        (remote/'vision.bin').write_bytes(b'changed')
    elif failure == 'release':
        release = json.loads((remote/'release.json').read_text())
        release['files']['manifest.json'] = 'f'*64
        (remote/'release.json').write_text(json.dumps(release))
    def download(**kwargs):
        if failure == 'transfer' and kwargs['filename'] == 'vision.bin':
            raise OSError('interrupted download')
        return original(**kwargs)
    with pytest.raises((ArtifactError, OSError)):
        fetch(SHA, cache, download, activate=True)
    assert link.resolve() == previous
    assert not (cache/('rgb-material-student-'+SHA)).exists()
    assert not list(cache.glob('.rgb-fetch-*'))


def test_cached_artifact_tampering_and_real_directory_activation_are_rejected(artifact, tmp_path):
    _, _, download = served_release(artifact, tmp_path)
    cache = tmp_path/'cache'
    result = fetch(SHA, cache, download)
    link = cache/'siglip2-material-student-384'
    link.mkdir()
    with pytest.raises(ArtifactError, match='real model directory'):
        fetch(SHA, cache, download, activate=True)
    assert link.is_dir() and not link.is_symlink()
    (Path(result['directory'])/'classifier.npz').write_bytes(b'modified')
    with pytest.raises(ArtifactError, match='checksum mismatch'):
        fetch(SHA, cache, download)


def test_duplicate_manifest_keys_are_rejected(artifact):
    (artifact/'manifest.json').write_text('{"files":{},"files":{}}')
    with pytest.raises(ArtifactError, match='Duplicate JSON'):
        validate(artifact)
