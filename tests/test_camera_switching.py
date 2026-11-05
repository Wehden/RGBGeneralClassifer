"""Discovery, ownership transfer, session isolation, and camera selection API."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from rgb import camera_devices
from rgb.camera import Camera, CameraBusy, CameraLock, CameraUnavailable, FrameExpired, make_frame, v4l2_serial
from test_local import local


def device(backend, physical):
    return {'id': camera_devices.identifier(backend, physical), 'backend': backend,
            'name': {'orbbec': 'Femto Bolt', 'realsense': 'RealSense D415', 'v4l2': 'USB webcam'}[backend],
            'serial': None if backend == 'realsense' else physical,
            'usb_serial': physical, 'device': '/dev/video12' if backend == 'v4l2' else None,
            'connected': True}


@pytest.fixture
def switching(local, monkeypatch, tmp_path):
    store, camera, frame, result, runtime, app = local
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    inventory = [device('orbbec', 'bolt-1'), device('realsense', 'rs-asic-1'), device('v4l2', 'webcam-1')]
    monkeypatch.setattr(camera_devices, 'discover', lambda: deepcopy(inventory))
    active, transitions = set(), []

    def capture():
        backend = camera.backend
        assert not active, 'Only one camera may be capturing'
        active.add(backend)
        transitions.append(('start', backend))
        try:
            camera.session_id = str(uuid4())
            camera.info = {'name': next(d['name'] for d in inventory if d['backend'] == backend)}
            metadata = {**frame.metadata, 'frame_id': str(uuid4()), 'session_id': camera.session_id,
                        'backend': backend, 'device': camera.info}
            camera.publish(make_frame(frame.color_rgb.copy(),
                frame.depth_native.copy() if backend != 'v4l2' else None,
                frame.ir_native.astype(np.uint8 if backend == 'realsense' else np.uint16) if backend != 'v4l2' else None,
                frame.depth_registered.copy() if backend != 'v4l2' else None, metadata))
            camera.stop_event.wait()
        finally:
            active.remove(backend)
            transitions.append(('stop', backend))

    monkeypatch.setattr(camera, '_orbbec', capture)
    monkeypatch.setattr(camera, '_v4l2', capture)
    monkeypatch.setattr('rgb.realsense.capture', lambda selected: capture())
    camera.select(inventory[0])
    with camera.condition:
        assert camera.condition.wait_for(lambda: bool(camera.frames), 2)
    yield store, camera, frame, result, runtime, app, inventory, transitions
    assert camera.stop()
    assert not active


def select(client, choice, **kwargs):
    return client.post('/api/camera/select', json={'camera_id': choice['id']},
                       headers={'X-RGB-Request': 'camera-select', **kwargs})


def test_switch_releases_old_camera_and_clears_session_results(switching):
    store, camera, old_fixture, result, runtime, app, inventory, transitions = switching
    old = camera.get()
    result = {**result, 'frame_id': old.id, 'session_id': old.metadata['session_id']}
    camera.latest_detection = (old, result)
    archived = runtime.archive(old, result)
    before = store.settings()
    runtime.live_readings['old'] = {}
    runtime.material_cache['old'] = {}
    runtime.material_comparison = {'old': True}
    runtime.material_inputs['old'] = b'pixels'
    runtime.material_pending = (old, result)
    runtime.pending = (old, result)
    runtime.last_auto_capture = 1
    response = select(app.test_client(), inventory[1])
    assert response.status_code == 200
    assert response.json['active_id'] == inventory[1]['id']
    with camera.condition:
        assert camera.condition.wait_for(lambda: bool(camera.frames), 2)
    new = camera.get()
    assert new.metadata['session_id'] != old.metadata['session_id']
    assert camera.backend == 'realsense' and new.ir_native.dtype == np.uint8
    assert transitions == [('start', 'orbbec'), ('stop', 'orbbec'), ('start', 'realsense')]
    assert not runtime.live_readings and not runtime.material_cache and not runtime.material_inputs
    assert runtime.material_pending is runtime.pending is runtime.material_comparison is None
    assert runtime.last_auto_capture == 1 and camera.latest_detection is None
    assert runtime.configuration['camera_backend'] == 'realsense'
    assert store.settings() == before and store.get(archived['event_id'])
    assert runtime.comparison()['rows'] == []
    with pytest.raises(FrameExpired): camera.get(old.id)
    with pytest.raises(CameraUnavailable, match='Active camera changed'): runtime.archive(old_fixture, result)
    # The old physical device is available again; the selected device is owned.
    lock = CameraLock('bolt-1')
    lock.close()
    with pytest.raises(CameraBusy): CameraLock('rs-asic-1')
    status = app.test_client().get('/api/camera/status').json
    assert status['views'] == ['color', 'depth', 'ir', 'pointcloud']


def test_switch_to_webcam_removes_depth_capabilities(switching):
    _, camera, _, _, _, app, inventory, _ = switching
    client = app.test_client()
    assert select(client, inventory[2]).status_code == 200
    with camera.condition:
        assert camera.condition.wait_for(lambda: bool(camera.frames), 2)
    assert client.get('/api/camera/status').json['views'] == ['color']
    assert client.get('/api/camera/depth/raw.png').status_code == 404
    assert client.get('/api/camera/pointcloud').status_code == 404
    assert client.get('/api/camera/color/snapshot.jpg').status_code == 200


def test_busy_target_restores_previous_owner_and_service_remains_available(switching):
    _, camera, _, _, _, app, inventory, transitions = switching
    lock = CameraLock('rs-asic-1')
    try:
        response = select(app.test_client(), inventory[1])
        assert response.status_code == 409 and 'another runner' in response.json['error']
        with camera.condition:
            assert camera.condition.wait_for(lambda: bool(camera.frames), 2)
        assert camera.backend == 'orbbec' and camera.fatal_error is None
        assert app.test_client().get('/healthz').status_code == 200
        assert transitions == [('start', 'orbbec'), ('stop', 'orbbec'), ('start', 'orbbec')]
    finally:
        lock.close()


def test_same_camera_is_noop_and_disconnected_target_does_not_stop_capture(switching):
    _, camera, _, _, runtime, app, inventory, transitions = switching
    previous = camera.get()
    client = app.test_client()
    assert select(client, inventory[0]).status_code == 200
    assert camera.get().id == previous.id and transitions == [('start', 'orbbec')]
    disconnected = inventory.pop()
    assert select(client, disconnected).status_code == 404
    assert camera.get().id == previous.id
    runtime.stop_event.set()
    assert select(client, inventory[1]).status_code == 503


@pytest.mark.parametrize('body', [None, [], {}, {'camera_id': 12}, {'camera_id': ''},
                                      {'camera_id': 'x' * 129}, {'camera_id': 'x', 'backend': 'v4l2'}])
def test_selection_rejects_invalid_payload(local, body):
    app = local[-1]
    response = app.test_client().post('/api/camera/select', json=body,
                                     headers={'X-RGB-Request': 'camera-select'})
    assert response.status_code == 400


def test_selection_checks_origin_and_request_marker(switching):
    *_, app, inventory, _ = switching
    client = app.test_client()
    assert client.post('/api/camera/select', json={'camera_id': inventory[1]['id']}).status_code == 400
    assert select(client, inventory[1], Origin='https://unrelated.example').status_code == 403
    assert select(client, inventory[0], Origin='http://localhost').status_code == 200


def test_existing_stream_ends_at_camera_change_and_releases_viewer_slot(switching):
    _, camera, _, _, _, app, inventory, _ = switching
    with app.test_request_context('/api/camera/ir/stream'):
        response = app.view_functions['camera_stream']('ir')
        iterator = iter(response.response)
        assert b'X-RGB-Session-ID' in next(iterator)
        old_stop = camera.stop_event
        camera.select(inventory[2])
        assert old_stop.is_set() and camera.stop_event is not old_stop
        with pytest.raises(StopIteration): next(iterator)
        response.close()
    assert app.test_client().get('/api/camera/status').json['stream_viewers'] == 0


def test_camera_will_not_open_target_until_old_worker_has_finished(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    camera = Camera(serial='bolt-1')
    camera.lock = CameraLock('bolt-1')
    camera.thread = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True)
    opened = []
    monkeypatch.setattr(camera, 'start', lambda: opened.append(True))
    with pytest.raises(CameraUnavailable, match='still stopping'):
        camera.select(device('realsense', 'rs-asic-1'))
    assert not opened and camera.backend == 'orbbec'
    with pytest.raises(CameraBusy): CameraLock('bolt-1')
    camera.thread = None
    assert camera.stop()


@pytest.mark.parametrize('selection_id', [None, 'selected-from-ui'])
def test_camera_change_is_serialized_and_native_busy_is_not_fatal(tmp_path, monkeypatch, selection_id):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    camera = Camera()
    camera.lifecycle_lock.acquire()
    try:
        with pytest.raises(CameraBusy, match='already in progress'):
            camera.select(device('realsense', 'rs-asic-1'))
    finally: camera.lifecycle_lock.release()
    camera.selection_id = selection_id
    def busy(): raise CameraBusy('Native capture busy')
    monkeypatch.setattr(camera, '_orbbec', busy)
    camera._run()
    assert camera.state == 'busy' and camera.fatal_error is None


def usb(root, name, vendor, product, title, serial=None):
    directory = root / name
    directory.mkdir(parents=True)
    for key, value in {'idVendor': vendor, 'idProduct': product, 'product': title}.items():
        (directory / key).write_text(value)
    if serial is not None: (directory / 'serial').write_text(serial)
    return directory


def video(root, name, parent, index=0, title='Webcam'):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / 'index').write_text(str(index))
    (directory / 'name').write_text(title)
    (directory / 'device').symlink_to(parent)
    return directory


def test_discovery_deduplicates_native_planes_and_groups_serialless_webcam(tmp_path, monkeypatch):
    usb_root, video_root = tmp_path / 'usb', tmp_path / 'video'
    monkeypatch.setattr(camera_devices, 'USB_ROOT', usb_root)
    monkeypatch.setattr(camera_devices, 'VIDEO_ROOT', video_root)
    bolt = usb(usb_root, '4-2', '2bc5', '066b', 'Orbbec Femto Bolt', 'bolt')
    real = usb(usb_root, '2-3', '8086', '0ad3', 'Intel RealSense Depth Camera 415', 'asic')
    webcam = usb(usb_root, '2-4', '1234', '5678', 'Webcam')
    usb(usb_root, '2-5', '8086', 'ffff', 'Bluetooth', 'other')
    usb(usb_root, '2-6', '8086', '0ad3', 'RealSense', '')  # Unverified native identity.
    for name in ['video0', 'video2', 'video4']: video(video_root, name, real)
    video(video_root, 'video6', bolt)
    video(video_root, 'video8', webcam)
    video(video_root, 'video9', webcam, index=1)  # Metadata node.
    video(video_root, 'video10', webcam)  # Another interface on the same webcam.
    video(video_root, 'video11', tmp_path / 'missing')
    cameras = camera_devices.discover()
    assert len(cameras) == 3
    assert [item['backend'] for item in cameras] == ['realsense', 'orbbec', 'v4l2']
    generic = cameras[-1]
    assert generic['usb_serial'] == 'usb:2-4'
    assert generic['device'] in {'/dev/video8', '/dev/video10'}
    assert generic['id'] == camera_devices.identifier('v4l2', 'usb:2-4')
    with pytest.raises(OSError): camera_devices.usb_identity(tmp_path / 'missing')


def test_catalog_preserves_active_disconnected_camera_and_distinct_same_model_labels(monkeypatch):
    inventory = [device('realsense', 'asic-one'), device('realsense', 'asic-two')]
    monkeypatch.setattr(camera_devices, 'discover', lambda: deepcopy(inventory))
    camera = Camera(backend='realsense', serial='sdk-two')
    camera.usb_serial = 'asic-two'
    catalog = camera_devices.catalog(camera)
    assert catalog['active_id'] == inventory[1]['id']
    assert len({entry['id'] for entry in catalog['cameras']}) == 2
    camera.selection_id = inventory[1]['id']
    inventory.pop()
    catalog = camera_devices.catalog(camera)
    assert not catalog['cameras'][-1]['connected']
    assert catalog['active_id'] == camera.selection_id
    camera = Camera(backend='v4l2', device='/dev/video12')
    inventory[:] = [device('v4l2', 'webcam')]
    assert camera_devices.catalog(camera)['active_id'] == inventory[0]['id']


def test_detector_and_material_completion_from_previous_session_cannot_publish(local, monkeypatch):
    _, camera, frame, result, runtime, _ = local
    model = SimpleNamespace(manifest={'model': 'fixture'}, sha256='a' * 64, config={},
                            instance_masks=[None], mask_error=None)
    def detect(*args, **kwargs):
        with camera.condition:
            camera.session_id = str(uuid4())
            camera.latest_detection = None
        runtime.stop_event.set()
        return deepcopy(result['detections']), 1.
    model.detect = detect
    monkeypatch.setattr('board.yoloe.Detector', lambda *args, **kwargs: model)
    runtime._detector()
    assert camera.latest_detection is None and runtime.material_pending is None
    runtime.stop_event.clear()
    camera.session_id = frame.metadata['session_id']
    detection = deepcopy(result['detections'][0])
    detection['box'] = [0, 0, 1, 1]
    class Materials:
        def classify(self, image):
            camera.session_id = str(uuid4())
            return {'label': 'plastic', 'inference_ms': 1., 'status': 'identified'}
    runtime._material_pass(Materials(), frame, [detection])
    assert not runtime.material_cache and not runtime.material_inputs
    assert runtime.material_comparison is None


def test_serialless_v4l2_identity_uses_physical_usb_port(tmp_path, monkeypatch):
    parent = usb(tmp_path, '2-7', '1234', '5678', 'Webcam', '')
    video(tmp_path / 'video', 'video0', parent)
    original = Path.resolve
    def resolve(path, *args, **kwargs):
        if str(path).startswith('/sys/class/video4linux/video0/device'):
            return parent
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', resolve)
    assert v4l2_serial('/dev/video0') == 'usb:2-7'


def test_startup_chooses_one_native_camera_without_overriding_explicit_cli_selection(monkeypatch):
    inventory = [device('orbbec', 'bolt-one'), device('orbbec', 'bolt-two'),
                 device('realsense', 'rs-one'), device('realsense', 'rs-two')]
    monkeypatch.setattr(camera_devices, 'discover', lambda: deepcopy(inventory))
    camera = Camera()
    camera_devices.choose_default(camera)
    assert camera.serial == 'bolt-one' and camera.selection_id == inventory[0]['id']
    camera = Camera(backend='realsense')
    camera_devices.choose_default(camera)
    assert camera.serial is None and camera.usb_serial == 'rs-one'
    for camera in (Camera(serial='explicit'), Camera(backend='v4l2', device='/dev/video20')):
        camera_devices.choose_default(camera)
        assert camera.selection_id is None
    inventory.clear()
    camera = Camera()
    camera_devices.choose_default(camera)
    assert camera.selection_id is None


def test_local_service_rejects_non_loopback_bind_before_initializing_hardware(monkeypatch):
    from rgb.local import main
    monkeypatch.setattr('sys.argv', ['rgb.local', '--host', '0.0.0.0'])
    with pytest.raises(SystemExit, match='requires --host 127.0.0.1'): main()


@pytest.mark.parametrize('error', [CameraBusy('Camera already owned'), CameraUnavailable('Device unavailable')])
def test_busy_or_unavailable_startup_keeps_workers_and_selection_api_available(local, monkeypatch, error):
    _, camera, _, _, runtime, app = local
    camera.frames.clear();camera.latest_detection=None
    def blocked(): raise error
    monkeypatch.setattr(camera, 'start', blocked)
    started=[]
    class Thread:
        def __init__(self, **kwargs): self.name=kwargs['name']
        def start(self): started.append(self.name)
    monkeypatch.setattr('rgb.local_runtime.threading.Thread', Thread)
    runtime.start()
    assert camera.state == ('busy' if isinstance(error,CameraBusy) else 'unavailable')
    assert camera.error == str(error) and camera.fatal_error is None
    assert set(started) == {'detector','archive','materials','vlm'}
    assert app.test_client().get('/api/camera/devices').status_code == 200


def test_switch_waits_for_manual_capture_and_preserves_its_camera_identity(switching, monkeypatch):
    store, camera, _, result, runtime, _, inventory, _ = switching
    frame=camera.get()
    camera.latest_detection=(frame,{**result,'frame_id':frame.id,'session_id':camera.session_id})
    started, release, selecting = threading.Event(), threading.Event(), threading.Event()
    original=store.save
    def save(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original(*args,**kwargs)
    monkeypatch.setattr(store,'save',save)
    def switch():
        selecting.set()
        runtime.select_camera(inventory[1]['id'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        saved=pool.submit(runtime.archive)
        try:
            assert started.wait(1)
            changed=pool.submit(switch)
            assert selecting.wait(1)
            assert not changed.done() and camera.backend=='orbbec'
        finally:release.set()
        event=saved.result(timeout=2)
        changed.result(timeout=2)
    assert event['frame_metadata']['device']['name']=='Femto Bolt'
    assert camera.backend=='realsense'
    assert store.get(event['event_id'])


def test_switch_preserves_missing_detector_error_until_model_recovers(switching):
    _,camera,_,_,_,_,inventory,_=switching
    camera.detector_status={'state':'unavailable','error':'Missing detector model'}
    camera.select(inventory[1])
    assert camera.detector_status['state']=='unavailable'
    assert camera.detector_status['error']=='Missing detector model'
