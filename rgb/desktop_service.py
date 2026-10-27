"""Desktop service lifecycle, independent of Qt and sensor ownership."""
import json
import os
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class ServiceUnavailable(Exception):
    pass


class ServiceMismatch(Exception):
    pass


def local_url(value):
    """Accept a loopback origin, with no credentials, path or query."""
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValueError('Expected a loopback HTTP(S) URL with a valid port') from exc
    if (parts.scheme not in {'http', 'https'}
            or parts.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parts.username is not None or parts.password is not None
            or parts.path not in {'', '/'} or parts.query or parts.fragment
            or port is not None and not 1 <= port <= 65535):
        raise ValueError('Expected a loopback origin, for example http://127.0.0.1:8080')
    return value.rstrip('/')


def probe(url, timeout=2):
    # Local requests must not take an ambient HTTP proxy route.
    opener = build_opener(ProxyHandler({}))
    result = {}
    for name, path in [('health', '/healthz'), ('vlm', '/api/vlm/status')]:
        try:
            with opener.open(Request(url + path, headers={'Accept': 'application/json'}),
                             timeout=timeout) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000:
                    raise ServiceMismatch('Local service response exceeds the console limit')
                result[name] = json.loads(payload)
        except HTTPError as exc:
            # An answering listener is not permission to start a second service.
            raise ServiceMismatch(f'Local service returned HTTP {exc.code} for {path}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ServiceUnavailable('Local service is not responding') from exc
        except (ValueError, UnicodeError) as exc:
            raise ServiceMismatch('The listener is not a local RGB service') from exc
    health, vlm = result['health'], result['vlm']
    if (not isinstance(health, dict) or health.get('status') != 'ok'
            or not isinstance(health.get('camera'), dict)
            or not isinstance(vlm, dict) or not isinstance(vlm.get('configuration'), dict)
            or not isinstance(vlm.get('enabled'), bool)):
        raise ServiceMismatch('The listener is not a local RGB service')
    return result


class ServiceController:
    """Attach first; own and stop only a child launched by this controller."""

    def __init__(self, url, *, command=None, state_dir=None, cwd=None, probe_fn=probe):
        self.url = local_url(url)
        self.command = command
        self.cwd = cwd
        self.state_dir = Path(state_dir).expanduser() if state_dir else None
        self.probe = probe_fn
        self.process = None
        self.log_path = self.state_dir / 'desktop-service.log' if self.state_dir else None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    @property
    def owned(self):
        return self.process is not None

    def connect(self):
        try:
            return self.probe(self.url)
        except ServiceUnavailable:
            if not self.command:
                raise
        # Closing during an initial probe must never launch an orphaned runner.
        with self.lock:
            if self.stop_event.is_set():
                raise ServiceUnavailable('Console is closing')
            if self.process is None:
                self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(self.log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                with os.fdopen(fd, 'ab', buffering=0) as output:
                    self.process = subprocess.Popen(
                        self.command, cwd=self.cwd, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        raise ServiceUnavailable('Starting the local camera service')

    def watch(self, notify):
        while not self.stop_event.is_set():
            try:
                if self.process is not None:
                    code = self.process.poll()
                    if code is not None:
                        notify({'state': 'failed', 'message': f'Runner exited ({code}). '
                                f'See {self.log_path}', 'owned': True})
                        return
                data = self.connect()
                notify({'state': 'ready', 'data': data, 'owned': self.owned})
            except ServiceMismatch as exc:
                notify({'state': 'failed', 'message': str(exc), 'owned': self.owned})
                return
            except (ServiceUnavailable, OSError) as exc:
                notify({'state': 'waiting', 'message': str(exc), 'owned': self.owned})
            if self.stop_event.wait(2):
                return

    def request_stop(self):
        self.stop_event.set()

    def close(self):
        self.request_stop()
        with self.lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
