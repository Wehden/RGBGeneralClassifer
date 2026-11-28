"""Short-lived authority for the outbound, remotely controlled board runner."""
from copy import deepcopy
import math
import threading
import time

POLICY_TTL_S = 15


class RemotePolicy:
    def __init__(self, source, clock=time.monotonic):
        self.source = source
        self.clock = clock
        self.lock = threading.Lock()
        self.accepted_at = None
        self.value = {'settings': {'detector_threshold': .45, 'vlm_threshold': .65,
            'display_mode': 'objects', 'classification_running': False,
            'record_observations': False, 'version': 0}, 'refinements': None}

    def accept(self, received, requested_at):
        # Invalid or delayed responses never renew authority. Start the lease at
        # request dispatch so a slow reply cannot extend an old enabled policy.
        settings = received['settings']
        for key in ('detector_threshold', 'vlm_threshold'):
            if type(settings[key]) not in (float, int) or not .05 <= settings[key] <= .99:
                raise ValueError('Invalid threshold')
        for key in ('classification_running', 'record_observations'):
            if type(settings[key]) is not bool:
                raise ValueError('Invalid classification controls')
        if type(settings['version']) is not int or settings['version'] < 0:
            raise ValueError('Invalid policy version')
        if not math.isfinite(requested_at) or not 0 <= self.clock()-requested_at < POLICY_TTL_S:
            raise ValueError('Expired policy response')
        candidate = deepcopy(received)
        candidate.setdefault('refinements', None)
        with self.lock:
            if self.accepted_at is not None and requested_at < self.accepted_at:
                raise ValueError('Superseded policy response')
            self.value, self.accepted_at = candidate, requested_at

    def snapshot(self):
        with self.lock:
            value = deepcopy(self.value)
            age = None if self.accepted_at is None else self.clock()-self.accepted_at
        fresh = age is not None and 0 <= age < POLICY_TTL_S
        settings = value['settings']
        if not fresh or settings.get('active_source', self.source) != self.source:
            settings.update(classification_running=False, record_observations=False)
            value['refinements'] = None
        value.update(policy_fresh=fresh, policy_max_age_s=POLICY_TTL_S)
        return value

    def refresh(self, transport):
        requested_at = self.clock()
        self.accept(transport.request('GET', '/policy'), requested_at)
