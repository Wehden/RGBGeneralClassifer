"""Lifecycle contracts work without importing Qt or opening a display."""
import json
import subprocess
import threading
from argparse import Namespace

import pytest

from rgb.desktop import controller_for
from rgb.desktop_service import (
    ServiceController, ServiceMismatch, ServiceUnavailable, local_url, probe,
)


@pytest.mark.parametrize('value', [
    'http://127.0.0.1:8080/', 'https://localhost:8443', 'http://[::1]:8080',
])
def test_console_accepts_loopback_origins(value):
    assert local_url(value) == value.rstrip('/')


@pytest.mark.parametrize('value', [
    'http://example.com:8080', 'file:///tmp/test', 'http://127.0.0.1:99999',
    'http://user@localhost:8080', 'http://localhost:8080/records',
    'http://localhost:8080?key=test', 'http://localhost:8080#fragment',
])
def test_console_rejects_nonlocal_or_ambiguous_origins(value):
    with pytest.raises(ValueError):
        local_url(value)


def test_console_attaches_without_spawning_or_stopping_existing_service(tmp_path, monkeypatch):
    status = {'health': {'status': 'ok'}, 'vlm': {}}
    controller = ServiceController(
        'http://127.0.0.1:8080', command=['unused'], state_dir=tmp_path,
        probe_fn=lambda _: status)
    def forbidden(*args, **kwargs):
        raise AssertionError('Attaching must not create a child process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    assert controller.connect() == status
    assert not controller.owned
    controller.close()
    assert not (tmp_path / 'desktop-service.log').exists()


def test_console_wrong_listener_is_not_replaced(tmp_path, monkeypatch):
    def wrong_listener(_):
        raise ServiceMismatch('Not a Bolt RGB service')
    controller = ServiceController('http://localhost:8080', command=['unused'],
                                   state_dir=tmp_path, probe_fn=wrong_listener)
    with pytest.raises(ServiceMismatch):
        controller.connect()
    assert not controller.owned


def test_console_launches_once_and_reaps_only_owned_child(tmp_path):
    def unavailable(_):
        raise ServiceUnavailable()
    controller = ServiceController(
        'http://127.0.0.1:8080',
        command=['python3', '-c', 'import time; time.sleep(60)'],
        state_dir=tmp_path, probe_fn=unavailable)
    try:
        with pytest.raises(ServiceUnavailable):
            controller.connect()
        child = controller.process
        with pytest.raises(ServiceUnavailable):
            controller.connect()
        assert controller.process is child and child.poll() is None
        assert controller.log_path.stat().st_mode & 0o777 == 0o600
        assert controller.owned
    finally:
        controller.close()
    assert child.poll() is not None


def test_console_closing_during_probe_cannot_start_orphan(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def delayed(_):
        entered.set()
        assert release.wait(5)
        raise ServiceUnavailable()
    controller = ServiceController('http://localhost:8080', command=['unused'],
                                   state_dir=tmp_path, probe_fn=delayed)
    statuses = []
    worker = threading.Thread(target=controller.watch, args=(statuses.append,))
    worker.start()
    assert entered.wait(5)
    controller.close()
    release.set()
    worker.join(5)
    assert not worker.is_alive() and not controller.owned
    assert not controller.log_path.exists()


def test_console_reports_exited_runner_without_respawning(tmp_path):
    def unavailable(_):
        raise ServiceUnavailable()
    controller = ServiceController(
        'http://localhost:8080', command=['python3', '-c', 'raise SystemExit(4)'],
        state_dir=tmp_path, probe_fn=unavailable)
    with pytest.raises(ServiceUnavailable):
        controller.connect()
    controller.process.wait(5)
    statuses = []
    controller.watch(statuses.append)
    assert statuses[0]['state'] == 'failed'
    assert 'exited (4)' in statuses[0]['message']
    controller.close()


def test_console_uses_runner_defaults_and_forces_loopback(tmp_path, monkeypatch):
    monkeypatch.setenv('RGB_STATE_DIR', str(tmp_path))
    monkeypatch.setenv('RGB_PORT', '8088')
    monkeypatch.delenv('RGB_HOST', raising=False)
    args = Namespace(url=None, attach=False,
                     runner_options=['--', '--no-vlm', '--aws-profile', 'default'])
    controller = controller_for(args)
    assert controller.url == 'http://127.0.0.1:8088'
    assert controller.state_dir == tmp_path
    assert controller.command[-4:] == ['--host', '127.0.0.1', '--port', '8088']
    assert '--no-vlm' in controller.command
    monkeypatch.setenv('RGB_HOST', '0.0.0.0')
    with pytest.raises(ValueError, match='127.0.0.1'):
        controller_for(args)


def test_console_attach_url_cannot_silently_apply_runner_options():
    args = Namespace(url='http://localhost:8088', attach=False, runner_options=[])
    assert controller_for(args).command is None
    args.runner_options = ['--', '--no-vlm']
    with pytest.raises(ValueError, match='cannot be applied'):
        controller_for(args)


def test_console_probes_validate_service_and_bypass_proxy(monkeypatch):
    from rgb import desktop_service
    from io import BytesIO
    seen = []
    class Opener:
        def open(self, request, timeout):
            seen.append(request.full_url)
            return BytesIO(json.dumps(
                {'status': 'ok', 'camera': {}} if request.full_url.endswith('/healthz')
                else {'configuration': {}, 'enabled': True}).encode())
    def opener(handler):
        assert handler.proxies == {}
        return Opener()
    monkeypatch.setattr(desktop_service, 'build_opener', opener)
    assert probe('http://localhost:8080')['vlm']['enabled']
    assert seen == ['http://localhost:8080/healthz', 'http://localhost:8080/api/vlm/status']
    class WrongOpener:
        def open(self, *args, **kwargs):
            return BytesIO(b'{"status":"ok"}')
    monkeypatch.setattr(desktop_service, 'build_opener', lambda *_: WrongOpener())
    with pytest.raises(ServiceMismatch):
        probe('http://localhost:8080')
