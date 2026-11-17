"""Basler USB color acquisition through pylon, using the shared frame contract."""
import hashlib
import time
from uuid import uuid4

import numpy as np

from rgb.camera import CameraLock, CameraUnavailable, make_frame
from rgb.local_store import encode
from rgb.store import now


def reserve(camera):
    """Lock the physical USB serial before loading pylon or opening a device."""
    from rgb.camera_devices import discover
    selected = getattr(camera, 'usb_serial', None) or camera.serial
    candidates = [item for item in discover() if item['backend'] == 'basler'
                  and (selected is None or item['usb_serial'] == selected)]
    if len(candidates) != 1:
        raise CameraUnavailable('No matching unique Basler USB camera; select --camera-serial')
    physical = candidates[0]['usb_serial']
    camera.lock = CameraLock(physical)
    camera.serial = camera.usb_serial = physical


def configure(sensor, genicam, fps):
    def writable(name):
        node = sensor.GetNodeMap().GetNode(name)
        return node if node is not None and genicam.IsWritable(node) else None

    # Disable hardware triggering for a continuous live preview. These changes
    # apply to the open session; no persistent camera user set is written.
    for name, value in [('AcquisitionMode', 'Continuous'), ('TriggerSelector', 'FrameStart'),
                        ('TriggerMode', 'Off')]:
        node = writable(name)
        if node is not None:
            node.SetValue(value)
    # Preserve the sensor's field of view instead of cropping a small preview ROI.
    for name in ['BinningHorizontal', 'BinningVertical']:
        node = writable(name)
        if node is not None:
            maximum = min(4, node.GetMax())
            value = node.GetMin() + (maximum - node.GetMin()) // node.GetInc() * node.GetInc()
            node.SetValue(value)
    for name in ['Width', 'Height']:
        node = writable(name)
        if node is not None:
            node.SetValue(node.GetMax())
    enable = writable('AcquisitionFrameRateEnable')
    if enable is not None:
        enable.SetValue(True)
    rate = writable('AcquisitionFrameRate')
    if rate is not None:
        rate.SetValue(max(rate.GetMin(), min(fps, rate.GetMax())))
    for name in ['ExposureAuto', 'GainAuto', 'BalanceWhiteAuto']:
        node = writable(name)
        if node is not None:
            node.SetValue('Continuous')
    return float(rate.GetValue()) if rate is not None else None


def capture(camera):
    camera.reserve()
    from pypylon import genicam, pylon
    factory = pylon.TlFactory.GetInstance()
    devices = [device for device in factory.EnumerateDevices()
               if device.GetDeviceClass() == 'BaslerUsb' and device.GetSerialNumber() == camera.serial]
    if len(devices) != 1:
        raise CameraUnavailable('Selected Basler camera is not visible to pylon; check USB permissions and other camera applications')
    sensor = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    try:
        sensor.Open()
        sensor.MaxNumBuffer.Value = 4
        fps = configure(sensor, genicam, camera.fps)
        source_format = str(sensor.PixelFormat.Value)
        camera.info = {'name': 'Basler ' + devices[0].GetModelName(), 'serial': camera.serial,
                       'usb_serial': camera.usb_serial, 'source_pixel_format': source_format}
        camera.calibration = {'schema_version': 1, 'available': False, 'backend': 'basler',
                              'serial': camera.serial}
        camera.calibration_hash = hashlib.sha256(encode(camera.calibration).encode()).hexdigest()
        camera.session_id = str(uuid4())
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        sensor.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        sequence, misses = 0, 0
        while not camera.stop_event.is_set():
            result = sensor.RetrieveResult(500, pylon.TimeoutHandling_Return)
            if result is None:
                misses += 1
                if misses >= 10:
                    raise CameraUnavailable('Basler capture timed out')
                continue
            try:
                camera.counters['received'] += 1
                if not result.GrabSucceeded():
                    raise CameraUnavailable('Basler capture failed: ' + str(result.ErrorDescription)[:160])
                received_at, received_monotonic = now(), time.monotonic()
                # SDK grab/converter buffers are reused; cached frames must own
                # their pixels before the grab result is released.
                color = np.array(converter.Convert(result).GetArray(), copy=True)
                if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
                    raise CameraUnavailable('Basler converter did not produce an 8-bit color image')
                sequence += 1
                misses = 0
                metadata = {'frame_id': str(uuid4()), 'session_id': camera.session_id,
                            'source_id': 'femto-bolt', 'backend': 'basler', 'sequence': sequence,
                            'received_at': received_at, 'received_monotonic_s': received_monotonic,
                            'timestamps_us': {}, 'timestamp_domain': 'unknown',
                            'sdk_timestamp_ticks': int(result.TimeStamp),
                            'profiles': {'color': {'width': color.shape[1], 'height': color.shape[0],
                                                  'fps': fps, 'format': 'BGR8',
                                                  'source_format': source_format}},
                            'calibration': camera.calibration,
                            'calibration_sha256': camera.calibration_hash, 'device': camera.info}
                camera.publish(make_frame(color, None, None, None, metadata,color_preview_max_size=1280))
            finally:
                result.Release()
    finally:
        try:
            if sensor.IsGrabbing():
                sensor.StopGrabbing()
        finally:
            if sensor.IsOpen():
                sensor.Close()
