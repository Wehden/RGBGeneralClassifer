"""Run the camera adapter against deterministic SDK protocol doubles."""
import sys
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from rgb.camera import Camera, CameraUnavailable


class Profile:
    def __init__(self, width, height, fmt, fps):
        self.width, self.height, self.fmt, self.fps = width, height, fmt, fps

    def get_width(self): return self.width
    def get_height(self): return self.height
    def get_fps(self): return self.fps
    def get_format(self): return self.fmt
    def get_intrinsic(self):
        return NS(width=self.width, height=self.height, fx=500., fy=500.,
                  cx=self.width / 2, cy=self.height / 2)
    def get_distortion(self):
        return NS(model='NONE', **{key: 0. for key in ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2']})
    def get_extrinsic_to(self, other):
        return NS(rot=np.eye(3).reshape(-1).tolist(), transform=[0., 0., 0.])


class Plane:
    def __init__(self, array, timestamp=1000, scale=1, encoded=False):
        self.array, self.timestamp, self.scale, self.encoded = array, timestamp, scale, encoded
    def get_data(self):
        return self.array if self.encoded else self.array.tobytes()
    def get_width(self): return self.array.shape[1]
    def get_height(self): return self.array.shape[0]
    def get_timestamp_us(self): return self.timestamp
    def get_system_timestamp_us(self): return self.timestamp + 100000
    def get_index(self): return 7
    def get_depth_scale(self): return self.scale


class Frames:
    def __init__(self, color, depth, ir):
        self.color, self.depth, self.ir = color, depth, ir
    def get_color_frame(self): return self.color
    def get_depth_frame(self): return self.depth
    def get_ir_frame(self): return self.ir
    def as_frame_set(self): return self


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    requests, enabled = [], []
    color = np.full((720, 1280, 3), (40, 80, 120), np.uint8)
    jpeg = cv2.imencode('.jpg', color)[1].tobytes()
    native = np.full((576, 640), 1234, np.uint16)
    valid = Frames(Plane(jpeg, encoded=True), Plane(native), Plane(native + 20))
    aligned = Frames(None, Plane(np.full((720, 1280), 1234, np.uint16), scale=2), None)
    state = NS(frames=[valid], aligned=aligned, starts=0, stops=0,
               devices=1, start_error=False, requests=requests, enabled=enabled, valid=valid)
    info = NS(get_serial_number=lambda: 'synthetic-bolt', get_name=lambda: 'Femto Bolt CI',
              get_firmware_version=lambda: 'fixture-1')
    device = NS(get_device_info=lambda: info)

    class Pipeline:
        def __init__(self, selected):
            assert selected is device
        def get_stream_profile_list(self, sensor):
            def select(w, h, fmt, fps):
                requests.append((sensor, w, h, fmt, fps))
                return Profile(w, h, fmt, fps)
            return NS(get_video_stream_profile=select)
        def enable_frame_sync(self): state.synced = True
        def start(self, config):
            state.starts += 1
            if state.start_error: raise RuntimeError('synthetic start failure')
        def wait_for_frames(self, timeout):
            assert timeout == 1000
            return state.frames.pop(0)
        def stop(self): state.stops += 1

    context = NS(set_logger_level=lambda level: None, query_devices=lambda: NS(
        get_count=lambda: state.devices, get_device_by_index=lambda i: device))
    fake = NS(
        Context=lambda: context, Pipeline=Pipeline,
        Config=lambda: NS(enable_stream=enabled.append,
                          set_frame_aggregate_output_mode=lambda mode: setattr(state, 'aggregate', mode)),
        AlignFilter=lambda **kwargs: NS(process=lambda frames: state.aligned),
        OBLogLevel=NS(ERROR='ERROR'),
        OBSensorType=NS(COLOR_SENSOR='color', DEPTH_SENSOR='depth', IR_SENSOR='ir'),
        OBFormat=NS(MJPG='MJPG', Y16='Y16'),
        OBFrameAggregateOutputMode=NS(FULL_FRAME_REQUIRE='FULL_FRAME_REQUIRE'),
        OBStreamType=NS(COLOR_STREAM='color'),
    )
    monkeypatch.setitem(sys.modules, 'pyorbbecsdk', fake)
    camera = Camera()
    monkeypatch.setattr(camera, 'reserve', lambda: None)
    original = camera.publish
    def publish(frame):
        original(frame)
        camera.stop_event.set()
    monkeypatch.setattr(camera, 'publish', publish)
    yield camera, state
    if camera.lock: camera.lock.close()


def test_sdk_sync_calibration_native_planes_and_separate_depth_scales(sdk):
    camera, state = sdk
    camera._orbbec()
    frame = camera.get()
    assert state.requests == [
        ('color', 1280, 720, 'MJPG', 15), ('depth', 640, 576, 'Y16', 15),
        ('ir', 640, 576, 'Y16', 15)]
    assert state.synced and state.aggregate == 'FULL_FRAME_REQUIRE'
    assert state.starts == state.stops == 1
    assert frame.depth_native.shape == (576, 640)
    assert frame.depth_registered.shape == (720, 1280)
    assert frame.metadata['depth_scale_mm'] == 1
    assert frame.metadata['registered_depth_scale_mm'] == 2
    assert frame.metadata['timestamps_us'] == dict.fromkeys(['color', 'depth', 'ir'], 1000)
    assert frame.metadata['sdk_frame_indices'] == dict.fromkeys(['color', 'depth', 'ir'], 7)
    assert frame.metadata['calibration_sha256'] == camera.calibration_hash
    assert frame.metadata['device']['serial'] == 'synthetic-bolt'
    assert all(not array.flags.writeable for array in
               [frame.color_rgb, frame.depth_native, frame.ir_native, frame.depth_registered])


@pytest.mark.parametrize('problem', ['missing', 'skew'])
def test_sdk_rejects_incomplete_and_unsynchronized_sets_before_publishing(sdk, problem):
    camera, state = sdk
    first = Frames(state.valid.color, state.valid.depth, None)
    if problem == 'skew':
        first.ir = Plane(state.valid.ir.array, timestamp=100000)
    state.frames.insert(0, first)
    camera._orbbec()
    assert camera.counters['accepted'] == 1
    assert camera.counters['incomplete' if problem == 'missing' else 'skew_rejected'] == 1
    assert state.stops == 1


@pytest.mark.parametrize('plane', ['native', 'registered'])
@pytest.mark.parametrize('scale', [0, float('nan')])
def test_sdk_invalid_scale_is_unavailable_and_stops_pipeline(sdk, plane, scale):
    camera, state = sdk
    (state.valid.depth if plane == 'native' else state.aligned.depth).scale = scale
    with pytest.raises(CameraUnavailable, match='Invalid depth scale'):
        camera._orbbec()
    assert not camera.frames and state.stops == 1


@pytest.mark.parametrize('problem', ['mjpeg', 'dimensions', 'timeout'])
def test_sdk_frame_failures_do_not_publish_corrupt_data(sdk, problem):
    camera, state = sdk
    if problem == 'mjpeg':
        state.valid.color.array = b'broken jpeg'
    elif problem == 'dimensions':
        state.aligned.depth.array = np.zeros((8, 10), np.uint16)
    else:
        state.frames = [None] * 5
    with pytest.raises(CameraUnavailable):
        camera._orbbec()
    assert not camera.frames and state.stops == 1


@pytest.mark.parametrize('devices', [0, 2])
def test_sdk_requires_a_unique_matching_device(sdk, devices):
    camera, state = sdk
    state.devices = devices
    with pytest.raises(CameraUnavailable, match='unique device'):
        camera._orbbec()
    assert state.starts == state.stops == 0


def test_sdk_start_failure_never_calls_stop_on_unstarted_pipeline(sdk):
    camera, state = sdk
    state.start_error = True
    with pytest.raises(RuntimeError, match='start failure'):
        camera._orbbec()
    assert state.starts == 1 and state.stops == 0
