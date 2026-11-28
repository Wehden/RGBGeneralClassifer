"""Exclusive source selection is shared by dashboard, policies and admission."""
import json
from uuid import uuid4

import fakeredis
import pytest

from rgb.events import validate_event
from rgb.store import Store, InactiveSource
from rgb.web import create_app
from test_events import event

AUTH = {'Authorization': 'Bearer ' + 'x' * 40}
DEVICE = {'Authorization': 'Bearer ' + 'd' * 40}
CONTROL = {'X-RGB-Request': 'settings', 'X-Laminarauth-Identity': 'operator'}
SOURCES = {'portentauno': {'label': 'Uno Q'}, 'zcam': {'label': 'Z CAM'}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('DEVICE_TOKENS_JSON', json.dumps({'portentauno': 'd' * 40}))
    store = Store(fakeredis.FakeRedis(decode_responses=True), tmp_path)
    return create_app(store, SOURCES, 'x' * 40).test_client(), store


def select(client, source):
    return client.post('/api/settings', json={'active_source': source}, headers=CONTROL)


def test_selection_masks_policy_preserves_settings_and_survives_restart(setup):
    client, store = setup
    store.update_settings({'detector_threshold': .31, 'display_mode': 'materials', 'vlm_paused': True})
    before = store.settings()
    result = select(client, 'zcam')
    assert result.status_code == 200
    for key, value in before.items():
        if key not in {'version', 'changed_at', 'actor'}:
            assert result.json[key] == value
    restarted = create_app(Store(store.redis, store.root), SOURCES, 'x' * 40).test_client()
    assert restarted.get('/api/settings').json['active_source'] == 'zcam'
    for path, headers in [('/api/board-policy/portentauno', AUTH), ('/api/device/portentauno/policy', DEVICE)]:
        policy = restarted.get(path, headers=headers).json['settings']
        assert policy['classification_running'] is False
        assert policy['record_observations'] is False
        assert policy['detector_threshold'] == .31
        assert policy['version'] == result.json['version']
    store.update_settings({'classification_running': True, 'record_observations': True})
    assert store.source_settings('portentauno')['classification_running'] is False
    assert restarted.get('/api/board-policy/zcam', headers=AUTH).json['settings']['classification_running'] is True
    select(restarted, 'portentauno')
    assert restarted.get('/api/device/portentauno/policy', headers=DEVICE).json['settings']['record_observations'] is True
    store.update_settings({'classification_running': False, 'record_observations': False})
    select(client, 'zcam')
    select(client, 'portentauno')
    assert store.source_settings('portentauno')['classification_running'] is False


def test_inactive_ingest_preserves_archives_and_blocks_backdated_events(setup, event):
    client, store = setup
    assert client.post('/api/ingest', json=event, headers=AUTH).status_code == 201
    original = store.raw(event['event_id'])
    files = set(store.root.iterdir())
    select(client, 'zcam')
    blocked = {**event, 'event_id': str(uuid4())}
    for path, headers in [('/api/ingest', AUTH), ('/api/device/portentauno/observations', DEVICE)]:
        assert client.post(path, json=blocked, headers=headers).status_code == 409
    assert store.redis.xlen('rgb:events') == 1
    assert store.raw(event['event_id']) == original
    assert set(store.root.iterdir()) == files
    assert client.get('/api/events/' + event['event_id']).status_code == 200
    assert client.post('/api/ingest', json={**blocked, 'event_id': str(uuid4()), 'source_id': 'zcam'}, headers=AUTH).status_code == 201
    select(client, 'portentauno')
    assert client.post('/api/ingest', json=blocked, headers=AUTH).status_code == 201


def test_switch_during_save_is_checked_atomically(setup, event, monkeypatch):
    _, store = setup
    parsed, image = validate_event(event, SOURCES)
    original_eval = store.redis.eval
    def racing_eval(script, *args):
        if script.startswith('\nlocal settings'):
            store.update_settings({'active_source': 'zcam'})
        return original_eval(script, *args)
    monkeypatch.setattr(store.redis, 'eval', racing_eval)
    with pytest.raises(InactiveSource):
        store.save(parsed, image)
    assert store.redis.xlen('rgb:events') == 0
    assert store.redis.hlen('rgb:latest') == 0
    assert store.redis.hlen('rgb:detections') == 0


def test_selection_validation_and_auth_boundaries(setup, monkeypatch):
    client, store = setup
    for source in [None, [], {}, 'unknown']:
        assert select(client, source).status_code == 400
    assert client.post('/api/settings', json={'active_source': 'zcam'}).status_code == 400
    assert client.post('/api/settings', json={'active_source': 'zcam'}, headers={**CONTROL, 'Origin': 'https://elsewhere'}).status_code == 403
    assert client.get('/api/device/portentauno/policy', headers=AUTH).status_code == 401
    assert client.get('/api/board-policy/portentauno').status_code == 401
    monkeypatch.setenv('ROLE', 'ingest')
    assert select(client, 'zcam').status_code == 404
    assert client.get('/api/settings').status_code == 404
    assert 'active_source' not in store.settings()
