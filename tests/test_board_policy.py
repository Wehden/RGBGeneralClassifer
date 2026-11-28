import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from board.policy import RemotePolicy, POLICY_TTL_S


def enabled(version=1, **updates):
    return {'settings': {'detector_threshold': .45, 'vlm_threshold': .65,
        'classification_running': True, 'record_observations': True,
        'version': version, **updates}, 'refinements': None}


def test_remote_policy_startup_expiry_and_reconnect():
    clock = [100.]
    policy = RemotePolicy('unoq', clock=lambda: clock[0])
    assert policy.snapshot()['settings']['classification_running'] is False
    policy.accept(enabled(), 100.)
    clock[0] += POLICY_TTL_S - .01
    assert policy.snapshot()['settings']['classification_running'] is True
    clock[0] += .01
    expired = policy.snapshot()
    assert expired['policy_fresh'] is False
    assert expired['settings']['classification_running'] is False
    assert expired['settings']['record_observations'] is False
    policy.accept(enabled(2, active_source='zcam'), clock[0])
    assert policy.snapshot()['settings']['classification_running'] is False
    policy.accept(enabled(3, active_source='unoq', record_observations=False), clock[0])
    assert policy.snapshot()['settings']['classification_running'] is True
    assert policy.snapshot()['settings']['record_observations'] is False


def test_network_errors_and_invalid_or_delayed_responses_never_renew_lease():
    clock = [0.]
    policy = RemotePolicy('unoq', clock=lambda: clock[0])
    policy.accept(enabled(), 0.)
    clock[0] = 14.
    transport = SimpleNamespace(request=Mock(side_effect=requests.Timeout()))
    with pytest.raises(requests.Timeout):policy.refresh(transport)
    with pytest.raises(ValueError):policy.accept(enabled(classification_running='true'), 14.)
    clock[0] = 15.
    assert policy.snapshot()['settings']['classification_running'] is False
    with pytest.raises(ValueError):policy.accept(enabled(), 0.)
    assert policy.snapshot()['policy_fresh'] is False
    policy.accept(enabled(2, classification_running=False), 15.)
    with pytest.raises(ValueError):policy.accept(enabled(), 14.)
    assert policy.snapshot()['settings']['classification_running'] is False


def test_refresh_lease_starts_before_network_request():
    clock = [0.]
    policy = RemotePolicy('unoq', clock=lambda: clock[0])
    def request(*args):
        clock[0] = 14.
        return enabled()
    policy.refresh(SimpleNamespace(request=request))
    assert policy.snapshot()['policy_fresh'] is True
    clock[0] = 15.
    assert policy.snapshot()['policy_fresh'] is False


@pytest.mark.parametrize('expire_during_inference', [False, True])
def test_actual_capture_loop_skips_expired_inference_and_recording(expire_during_inference):
    # Execute the runner loop without importing model-loading/device startup.
    from datetime import datetime, timezone
    clock = [0.]
    policy = RemotePolicy('unoq', clock=lambda: clock[0])
    policy.accept(enabled(), 0.)
    if not expire_during_inference:clock[0] = 15.
    class Finished(BaseException):pass
    def finish(*args):raise Finished()
    def detect(frame):
        clock[0] = 15.
        return [], 10
    detector = SimpleNamespace(detect=Mock(side_effect=detect))
    class Condition:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def notify_all(self):pass
    state = {'sequence': 0}
    database = Mock(side_effect=AssertionError('Expired policy must not record'))
    scope = dict(policy=policy, Tracker=lambda:SimpleNamespace(update=lambda *args:None),
        camera=lambda:(SimpleNamespace(read=lambda:(True, object())), 'fake'),
        state=state, detectors={'objects':detector}, time=SimpleNamespace(monotonic=lambda:clock[0],time=lambda:100.,sleep=finish),
        datetime=datetime, timezone=timezone, identity=lambda _: {}, condition=Condition(),
        cv2=SimpleNamespace(imencode=lambda *args:(True,SimpleNamespace(tobytes=lambda:b'jpeg')),IMWRITE_JPEG_QUALITY=1),
        overlay=lambda frame,detections:frame, MAX_FPS=2, database=database,
        log=SimpleNamespace(warning=Mock(side_effect=AssertionError('Unexpected capture failure'))))
    tree=ast.parse(Path('board/main.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'board/main.py','exec'),scope)
    with pytest.raises(Finished):scope['capture']()
    assert detector.detect.call_count == int(expire_during_inference)
    assert state['classification_running'] is False
    assert state['record_observations'] is False
    assert state['policy_fresh'] is False
    assert state['detections'] == []
    database.assert_not_called()
