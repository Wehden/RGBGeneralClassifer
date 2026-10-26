"""Outbound HTTPS only. Network identity is the source ID, never an IP address."""
import os
import json
import ipaddress
import threading
from urllib.parse import urlsplit
import requests


class Transport:
    def __init__(self, source, environ=None):
        env = os.environ if environ is None else environ
        self.origin = env['ARCHIVE_URL'].rstrip('/')
        parsed = urlsplit(self.origin)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or
                parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError('ARCHIVE_URL must be an HTTPS origin')
        try:ipaddress.ip_address(parsed.hostname)
        except ValueError:pass
        else:raise ValueError('ARCHIVE_URL must use public DNS')
        if '.' not in parsed.hostname or parsed.hostname.endswith(('.local','.internal','.cluster.local')):
            raise ValueError('ARCHIVE_URL must use public DNS')
        self.base = self.origin + '/api/device/' + source
        if not source or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in source):
            raise ValueError('Invalid source ID')
        self.headers = {'Authorization': 'Bearer ' + env['INGEST_TOKEN']}
        if bool(env.get('CF_ACCESS_CLIENT_ID')) != bool(env.get('CF_ACCESS_CLIENT_SECRET')):
            raise ValueError('Both Access service credentials are required')
        if env.get('CF_ACCESS_CLIENT_ID') and env.get('CF_ACCESS_CLIENT_SECRET'):
            self.headers.update({'CF-Access-Client-Id':env['CF_ACCESS_CLIENT_ID'],
                                 'CF-Access-Client-Secret':env['CF_ACCESS_CLIENT_SECRET']})
        self.local = threading.local()

    @property
    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.base + path, headers=self.headers,
                                        timeout=(5, 10), allow_redirects=False, **kwargs)
        # A login redirect or HTML success must never acknowledge an outbox row.
        if not 200 <= response.status_code < 300:
            raise requests.HTTPError('Device endpoint HTTP ' + str(response.status_code))
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('Expected device response')
        return data


def observation_payload(payload):
    data=json.loads(payload)
    # Older queued payloads may contain overlay/tracking keys on shared proposals.
    # Keep the original outbox row until the corrected raw proposals are acknowledged.
    if 'detector_proposals' in data:
        data['detector_proposals']=[{key:proposal[key] for key in ['class_id','label','confidence','box']}
                                    for proposal in data['detector_proposals']]
    return data
