"""Basler SDK buffer ownership, cleanup, selection and USB lock contracts."""
import sys
from types import SimpleNamespace as NS

import numpy as np
import cv2
import pytest

from rgb import basler, camera_devices
from rgb.camera import Camera, CameraBusy, CameraLock, CameraUnavailable


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path))
    inventory = [{'backend': 'basler', 'usb_serial': '23854255'}]
    monkeypatch.setattr(camera_devices, 'discover', lambda: inventory)
    camera = Camera(backend='basler', serial='23854255')
    state = NS(opened=False, grabbing=False, starts=0, stops=0, closes=0, released=0,
               open_error=False, start_error=False, convert_error=False, grab_ok=True,
               array=np.full((12, 16, 3), [10, 20, 30], np.uint8), timeouts=0)

    class Node:
        def __init__(self, value, minimum=1, maximum=4, writable=True):
            self.value, self.minimum, self.maximum, self.writable = value, minimum, maximum, writable
        def GetMin(self): return self.minimum
        def GetMax(self): return self.maximum
        def GetInc(self): return 1
        def GetValue(self): return self.value
        def SetValue(self, value): self.value = value

    nodes = {name: Node('Off') for name in [
        'AcquisitionMode', 'TriggerSelector', 'TriggerMode', 'ExposureAuto', 'GainAuto', 'BalanceWhiteAuto']}
    nodes.update({name: Node(1) for name in ['BinningHorizontal', 'BinningVertical']})
    nodes.update(Width=Node(16, maximum=16), Height=Node(12, maximum=12),
                 AcquisitionFrameRateEnable=Node(False), AcquisitionFrameRate=Node(30, maximum=20))
    state.nodes = nodes
    info = NS(GetDeviceClass=lambda: 'BaslerUsb', GetSerialNumber=lambda: '23854255',
              GetModelName=lambda: 'acA5472-17uc')
    state.devices = [info]
    result = NS(GrabSucceeded=lambda: state.grab_ok, ErrorDescription='synthetic failed grab', TimeStamp=123456)
    def release():
        state.released += 1
        state.array[:] = 0  # Simulate SDK buffer reuse immediately after Release.
    result.Release = release

    class Sensor:
        MaxNumBuffer = NS(Value=10)
        PixelFormat = NS(Value='BayerRG8')
        def Open(self):
            if state.open_error: raise RuntimeError('synthetic open failure')
            state.opened = True
        def GetNodeMap(self): return NS(GetNode=nodes.get)
        def IsOpen(self): return state.opened
        def IsGrabbing(self): return state.grabbing
        def StartGrabbing(self, strategy):
            assert strategy == 'latest'
            if state.start_error: raise RuntimeError('synthetic start failure')
            state.starts += 1
            state.grabbing = True
        def RetrieveResult(self, timeout, policy):
            assert (timeout, policy) == (500, 'return')
            if state.timeouts:
                state.timeouts -= 1
                return None
            return result
        def StopGrabbing(self):
            state.stops += 1
            state.grabbing = False
        def Close(self):
            state.closes += 1
            state.opened = False

    sensor = Sensor()
    class Converter:
        def Convert(self, selected):
            assert selected is result
            if state.convert_error: raise RuntimeError('synthetic conversion failure')
            return NS(GetArray=lambda: state.array)
    factory = NS(EnumerateDevices=lambda: state.devices, CreateDevice=lambda selected: selected)
    pylon = NS(TlFactory=NS(GetInstance=lambda: factory), InstantCamera=lambda selected: sensor,
               ImageFormatConverter=Converter, PixelType_BGR8packed='BGR8',
               OutputBitAlignment_MsbAligned='msb', GrabStrategy_LatestImageOnly='latest',
               TimeoutHandling_Return='return')
    genicam = NS(IsWritable=lambda node: node.writable)
    monkeypatch.setitem(sys.modules, 'pypylon', NS(pylon=pylon, genicam=genicam))
    original = camera.publish
    def publish(frame):
        original(frame)
        camera.stop_event.set()
    monkeypatch.setattr(camera, 'publish', publish)
    yield camera, state, inventory, genicam, sensor
    camera.stop()


def test_capture_owns_color_pixels_and_reports_rgb_only_metadata(sdk):
    camera, state, _, _, sensor = sdk
    camera._run()
    frame = camera.get()
    assert state.starts == state.stops == state.closes == state.released == 1
    assert sensor.MaxNumBuffer.Value == 4
    assert frame.color_rgb[0, 0].tolist() == [30, 20, 10]
    assert not frame.color_rgb.flags.writeable and not state.array.any()
    assert frame.depth_native is frame.ir_native is frame.depth_registered is None
    assert list(frame.previews) == ['color']
    assert frame.metadata['profiles']['color'] == {
        'width': 16, 'height': 12, 'fps': 15., 'format': 'BGR8', 'source_format': 'BayerRG8'}
    assert frame.metadata['device']['serial'] == frame.metadata['device']['usb_serial'] == '23854255'
    assert frame.metadata['sdk_timestamp_ticks'] == 123456
    assert frame.metadata['timestamps_us'] == {}
    assert frame.metadata['timestamp_domain'] == 'unknown'
    assert not frame.metadata['calibration']['available']


def test_configuration_preserves_field_of_view_and_clamps_frame_rate(sdk):
    _, state, _, genicam, sensor = sdk
    assert basler.configure(sensor, genicam, 30) == 20
    for name in ['BinningHorizontal', 'BinningVertical']:
        assert state.nodes[name].value == 4
    assert state.nodes['Width'].value == 16 and state.nodes['Height'].value == 12
    assert state.nodes['TriggerMode'].value == 'Off'
    assert state.nodes['ExposureAuto'].value == 'Continuous'
    state.nodes.clear()  # Optional features may be absent on another camera.
    assert basler.configure(sensor, genicam, 15) is None


def test_large_color_frames_keep_full_resolution_and_bound_the_jpeg_preview(sdk):
    camera, state, _, _, _ = sdk
    state.array = np.full((1200,1600,3), [10,20,30], np.uint8)
    basler.capture(camera)
    frame = camera.get()
    assert frame.color_rgb.shape == (1200,1600,3)
    assert frame.metadata['profiles']['color']['width'] == 1600
    assert frame.metadata['color_preview'] == {'width':1280,'height':960,'resized':True}
    preview = cv2.imdecode(np.frombuffer(frame.previews['color'],np.uint8),cv2.IMREAD_COLOR)
    assert preview.shape == (960,1280,3)
    assert len(frame.previews['color']) < 4*1024*1024
    assert frame.color_rgb[0,0].tolist() == [30,20,10] and not state.array.any()

@pytest.mark.parametrize('problem', ['open', 'start', 'convert', 'grab', 'dtype', 'shape', 'timeout'])
def test_failures_release_grabs_and_close_opened_sensor(sdk, problem):
    camera, state, _, _, _ = sdk
    if problem in {'open', 'start', 'convert'}: setattr(state, problem + '_error', True)
    if problem == 'grab': state.grab_ok = False
    if problem == 'dtype': state.array = state.array.astype(np.uint16)
    if problem == 'shape': state.array = state.array[:, :, 0]
    if problem == 'timeout': state.timeouts = 10
    with pytest.raises((RuntimeError, CameraUnavailable)):
        basler.capture(camera)
    assert not camera.frames
    assert state.closes == (0 if problem == 'open' else 1)
    assert state.stops == (0 if problem in {'open', 'start'} else 1)
    assert state.released == (1 if problem in {'convert', 'grab', 'dtype', 'shape'} else 0)


def test_transient_timeout_recovers_without_reopening_sensor(sdk):
    camera, state, _, _, _ = sdk
    state.timeouts = 1
    basler.capture(camera)
    assert camera.counters['accepted'] == 1 and state.starts == 1


@pytest.mark.parametrize('problem', ['absent', 'wrong_serial', 'wrong_transport', 'duplicate'])
def test_sdk_device_must_match_reserved_usb_serial_and_transport(sdk, problem):
    camera, state, _, _, _ = sdk
    if problem == 'absent': state.devices.clear()
    if problem == 'duplicate': state.devices *= 2
    if problem == 'wrong_serial': state.devices[0].GetSerialNumber = lambda: 'other'
    if problem == 'wrong_transport': state.devices[0].GetDeviceClass = lambda: 'BaslerGigE'
    with pytest.raises(CameraUnavailable, match='USB permissions'):
        basler.capture(camera)
    assert not state.opened and not state.starts


def test_usb_reservation_is_unique_and_honors_selection(sdk):
    camera, _, inventory, _, _ = sdk
    lock = CameraLock('23854255')
    try:
        with pytest.raises(CameraBusy): camera.reserve()
    finally:
        lock.close()
    inventory.append({'backend': 'basler', 'usb_serial': 'another'})
    camera.serial = None
    with pytest.raises(CameraUnavailable): camera.reserve()
    camera.serial = '23854255'
    camera.reserve()
    assert camera.usb_serial == '23854255'
    inventory.clear()
    camera.stop()
    with pytest.raises(CameraUnavailable): camera.reserve()
