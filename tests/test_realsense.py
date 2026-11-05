"""RealSense SDK, USB ownership, synchronization, and native plane contracts."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from rgb.camera import Camera, CameraBusy, CameraUnavailable
from rgb import realsense


class Profile:
    def __init__(self, stream, width, height, fmt, index=0, fps=15):
        self.stream, self.w, self.h, self.fmt, self.index, self.rate = stream, width, height, fmt, index, fps
        self.coeffs = [0.] * 5
        self.model = 'distortion.inverse_brown_conrady' if stream == 'color' else 'distortion.brown_conrady'
        self.rotation = [0., 1., 0., -1., 0., 0., 0., 0., 1.]  # Column-major.
    def stream_type(self): return self.stream
    def stream_index(self): return self.index
    def width(self): return self.w
    def height(self): return self.h
    def fps(self): return self.rate
    def format(self): return self.fmt
    def is_video_stream_profile(self): return True
    def as_video_stream_profile(self): return self
    def get_intrinsics(self):
        return NS(width=self.w, height=self.h, fx=500., fy=500., ppx=self.w/2, ppy=self.h/2,
                  model=self.model, coeffs=self.coeffs)
    def get_extrinsics_to(self, other):
        return NS(rotation=self.rotation, translation=[.01, -.02, .03])


class Plane:
    def __init__(self, array, timestamp=100., domain='timestamp_domain.global_time'):
        self.array, self.timestamp, self.domain = array, timestamp, domain
    def get_data(self): return self.array
    def get_timestamp(self): return self.timestamp
    def get_frame_timestamp_domain(self): return self.domain
    def get_frame_number(self): return 7


class Frames:
    def __init__(self, color, depth, ir):
        self.color, self.depth, self.ir = color, depth, ir
    def get_color_frame(self): return self.color
    def get_depth_frame(self): return self.depth
    def get_infrared_frame(self, index):
        assert index == 1
        return self.ir


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    monkeypatch.setattr(realsense, 'usb_serials', lambda: ['usb-asic-serial'])
    profiles = [Profile('color', 1280, 720, 'RGB8'), Profile('depth', 640, 480, 'Z16'),
                Profile('ir', 640, 480, 'Y8', 1)]
    color = Plane(np.full((720, 1280, 3), [30, 70, 110], np.uint8), timestamp=100.6)
    depth = Plane(np.full((480, 640), 1500, np.uint16))
    ir = Plane(np.full((480, 640), 123, np.uint8))
    valid = Frames(color, depth, ir)
    aligned = Plane(np.full((720, 1280), 1500, np.uint16))
    state = NS(profiles=profiles, valid=valid, frames=[valid], aligned=aligned,
               scale=.001, starts=0, stops=0, enabled=[], devices=1,
               start_error=False, wait_error=False, has_asic=True)
    info = {'name': 'RealSense D415 CI', 'serial': 'sdk-camera-serial', 'asic': 'usb-asic-serial',
            'firmware': 'test-1'}
    sensor = NS(get_stream_profiles=lambda: state.profiles, get_depth_scale=lambda: state.scale)
    device = NS(supports=lambda key: state.has_asic, get_info=info.__getitem__,
                query_sensors=lambda: [sensor], first_depth_sensor=lambda: sensor)
    active = NS(get_stream=lambda stream, index: next(
        p for p in state.profiles if p.stream_type() == stream and p.stream_index() == index),
        get_device=lambda: device)
    context = NS(query_devices=lambda: [device] * state.devices)

    class Pipeline:
        def __init__(self, selected):
            assert selected is context
        def start(self, config):
            state.starts += 1
            if state.start_error:
                raise RuntimeError('synthetic busy device')
            return active
        def stop(self): state.stops += 1
        def wait_for_frames(self, timeout):
            assert timeout == 1000
            if state.wait_error:
                raise RuntimeError('synthetic disconnect')
            return state.frames.pop(0)

    fake = NS(context=lambda: context, pipeline=Pipeline,
              config=lambda: NS(enable_device=lambda serial: setattr(state, 'selected', serial),
                                enable_stream=lambda *args: state.enabled.append(args)),
              align=lambda target: NS(process=lambda frames: Frames(None, state.aligned, None)),
              stream=NS(color='color', depth='depth', infrared='ir'),
              format=NS(rgb8='RGB8', z16='Z16', y8='Y8'),
              camera_info=NS(name='name', serial_number='serial', asic_serial_number='asic',
                             firmware_version='firmware'))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', fake)
    camera = Camera(backend='realsense', serial='sdk-camera-serial')
    original = camera.publish
    def publish(frame):
        original(frame)
        camera.stop_event.set()
    monkeypatch.setattr(camera, 'publish', publish)
    yield camera, state, fake, device
    if camera.lock:
        camera.lock.close()


def test_realsense_retains_native_ir_scale_timestamps_and_factory_calibration(sdk):
    camera, state, _, _ = sdk
    realsense.capture(camera)
    frame = camera.get()
    assert state.starts == state.stops == 1
    assert state.selected == 'sdk-camera-serial'
    assert state.enabled == [('color', 0, 1280, 720, 'RGB8', 15),
                             ('depth', 0, 640, 480, 'Z16', 15), ('ir', 1, 640, 480, 'Y8', 15)]
    assert frame.ir_native.dtype == np.uint8 and frame.depth_native.dtype == np.uint16
    assert frame.depth_registered.shape == frame.color_rgb.shape[:2] == (720, 1280)
    assert frame.metadata['depth_scale_mm'] == frame.metadata['registered_depth_scale_mm'] == 1.
    assert frame.metadata['timestamps_us'] == {'color': 100600, 'depth': 100000, 'ir': 100000}
    assert frame.metadata['channel_skew_ms'] == .6
    assert frame.metadata['raw_bit_depth'] == {'depth': 16, 'ir': 8}
    assert frame.metadata['device']['usb_serial'] == 'usb-asic-serial'
    assert frame.metadata['device']['serial'] == 'sdk-camera-serial'
    planes = frame.metadata['calibration']['planes']
    assert planes['color']['distortion']['model'] == 'NONE'
    assert planes['color']['sdk_distortion']['model'] == 'distortion.inverse_brown_conrady'
    assert planes['depth']['extrinsics_to_color']['rotation'] == [0., -1., 0., 1., 0., 0., 0., 0., 1.]
    assert planes['depth']['extrinsics_to_color']['translation_mm'] == [10., -20., 30.]
    for array in [frame.color_rgb, frame.depth_native, frame.ir_native, frame.depth_registered]:
        assert not array.flags.writeable
    state.valid.depth.array[:] = 999
    state.valid.ir.array[:] = 42
    state.valid.color.array[:] = 0
    assert frame.depth_native[0, 0] == 1500 and frame.ir_native[0, 0] == 123
    assert frame.color_rgb[0, 0].tolist() == [30, 70, 110]


@pytest.mark.parametrize('problem', ['missing', 'skew', 'domain'])
def test_realsense_rejects_incomplete_or_unsynchronized_sets(sdk, problem):
    camera, state, _, _ = sdk
    rejected = Frames(state.valid.color, state.valid.depth, None)
    if problem != 'missing':
        rejected.ir = Plane(state.valid.ir.array, timestamp=150. if problem == 'skew' else 100.,
                            domain='hardware_clock' if problem == 'domain' else 'timestamp_domain.global_time')
    state.frames.insert(0, rejected)
    realsense.capture(camera)
    assert camera.counters['accepted'] == 1
    assert camera.counters['skew_rejected' if problem == 'skew' else 'incomplete'] == 1
    assert state.stops == 1


@pytest.mark.parametrize('problem', ['scale', 'native', 'registered', 'alignment', 'ir', 'timeout', 'incomplete'])
def test_realsense_failures_stop_the_pipeline_without_publishing(sdk, problem):
    camera, state, _, _ = sdk
    if problem == 'scale': state.scale = float('nan')
    if problem == 'native': state.valid.depth.array = np.zeros((1, 1), np.uint16)
    if problem == 'registered': state.aligned.array = np.zeros((1, 1), np.uint16)
    if problem == 'alignment': state.aligned = None
    if problem == 'ir': state.valid.ir.array = np.zeros((480, 640), np.uint16)
    if problem == 'timeout': state.wait_error = True
    if problem == 'incomplete': state.frames = [Frames(None, None, None)] * 5
    with pytest.raises((CameraUnavailable, RuntimeError)):
        realsense.capture(camera)
    assert not camera.frames and state.stops == 1


def test_realsense_start_failure_does_not_stop_an_unstarted_pipeline(sdk):
    camera, state, _, _ = sdk
    state.start_error = True
    with pytest.raises(RuntimeError, match='busy'):
        realsense.capture(camera)
    assert state.starts == 1 and state.stops == 0


@pytest.mark.parametrize('devices', [0, 2])
def test_realsense_requires_unique_device_selection(sdk, devices):
    camera, state, _, _ = sdk
    state.devices = devices
    with pytest.raises(CameraUnavailable, match='unique'):
        realsense.capture(camera)
    assert not state.starts


def test_realsense_can_capture_rgb_depth_without_ir(sdk):
    camera, state, _, _ = sdk
    state.profiles.pop()
    realsense.capture(camera)
    frame = camera.get()
    assert set(frame.previews) == {'color', 'depth'} and frame.ir_native is None
    assert set(frame.metadata['timestamps_us']) == {'color', 'depth'}


def test_realsense_unsupported_rate_is_explicit(sdk):
    camera, state, _, _ = sdk
    camera.fps = 5
    with pytest.raises(CameraUnavailable, match='try 15 or 30'):
        realsense.capture(camera)
    assert not state.starts


def test_realsense_selection_and_v4l2_use_the_same_physical_lock(sdk, monkeypatch):
    camera, _, _, _ = sdk
    camera.reserve()
    monkeypatch.setattr('rgb.camera.v4l2_serial', lambda path: 'usb-asic-serial')
    with pytest.raises(CameraBusy):
        Camera(backend='v4l2', device='/dev/video4').reserve()
    with pytest.raises(CameraBusy):
        Camera(backend='realsense', serial='sdk-camera-serial').reserve()


def test_realsense_multiple_devices_need_sdk_serial_then_claim_selected_usb(sdk, monkeypatch):
    camera, state, rs, device = sdk
    monkeypatch.setattr(realsense, 'usb_serials', lambda: ['other-usb', 'usb-asic-serial'])
    with pytest.raises(CameraUnavailable, match='Multiple'):
        Camera(backend='realsense').reserve()
    camera.reserve()
    assert camera.lock is None
    realsense.claim_device(camera, device, rs)
    assert camera.usb_serial == 'usb-asic-serial'
    realsense.claim_device(camera, device, rs)  # Claiming the same device is safe.


@pytest.mark.parametrize('problem', ['unsupported', 'absent', 'mismatch'])
def test_realsense_refuses_unverified_physical_identity(sdk, monkeypatch, problem):
    camera, state, rs, device = sdk
    if problem == 'unsupported': state.has_asic = False
    if problem == 'absent': monkeypatch.setattr(realsense, 'usb_serials', lambda: [])
    if problem == 'mismatch':
        camera.reserve()
        camera.usb_serial = 'different-usb'
    with pytest.raises(CameraUnavailable):
        realsense.claim_device(camera, device, rs)


def test_realsense_calibration_preserves_nonzero_distortion_models(sdk):
    _, state, _, _ = sdk
    color, depth, _ = state.profiles
    color.coeffs = [.1, -.03, .001, -.002, .01]
    depth.coeffs = [.1, -.03, .001, -.002, .01]
    planes = realsense.calibration({'color': color, 'depth': depth})['planes']
    assert planes['color']['distortion']['model'] == 'REALSENSE_INVERSE_BROWN_CONRADY'
    assert planes['depth']['distortion']['model'] == 'BROWN_CONRADY'
    assert planes['depth']['distortion']['p1'] == .001


def test_passive_realsense_discovery_ignores_other_intel_devices(tmp_path, monkeypatch):
    devices = tmp_path / 'usb'
    for name, vendor, product, serial in [('camera', '8086', 'Intel RealSense D415', 'usb-serial'),
                                           ('other', '8086', 'Bluetooth', 'bt-serial'),
                                           ('bolt', '2bc5', 'Femto Bolt', 'bolt-serial')]:
        path = devices / name
        path.mkdir(parents=True)
        for field, value in [('idVendor', vendor), ('product', product), ('serial', serial)]:
            (path / field).write_text(value)
    (devices / 'missing').mkdir()
    original = realsense.Path.glob
    monkeypatch.setattr(realsense.Path, 'glob', lambda self, pattern:
                        original(devices, pattern) if str(self) == '/sys/bus/usb/devices' else original(self, pattern))
    assert realsense.usb_serials() == ['usb-serial']
