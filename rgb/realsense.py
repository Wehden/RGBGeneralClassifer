"""USB RealSense capture into the shared immutable RGB/depth/IR frame contract."""
import hashlib
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np

from rgb.camera import CameraLock, CameraUnavailable, make_frame
from rgb.local_store import encode
from rgb.store import now


def usb_serials():
    """Find physical USB identities without acquiring a sensor."""
    serials = []
    for path in Path('/sys/bus/usb/devices').glob('*'):
        try:
            if ((path / 'idVendor').read_text().strip().lower() == '8086'
                    and 'realsense' in (path / 'product').read_text().lower()):
                serial = (path / 'serial').read_text().strip()
                if serial:
                    serials.append(serial)
        except OSError:
            continue
    return serials


def reserve(camera):
    if camera.lock is not None:
        return
    candidates = usb_serials()
    if len(candidates) > 1 and camera.serial is None:
        raise CameraUnavailable('Multiple RealSense cameras found; select --camera-serial')
    if len(candidates) == 1:
        # RealSense's SDK serial can differ from its USB/ASIC serial. V4L2 and
        # native SDK capture must lock the same physical USB camera.
        camera.lock = CameraLock(candidates[0])
        camera.usb_serial = candidates[0]


def claim_device(camera, device, rs):
    key = rs.camera_info.asic_serial_number
    if not device.supports(key):
        raise CameraUnavailable('Cannot establish RealSense USB/ASIC identity')
    physical = device.get_info(key)
    if physical not in usb_serials():
        raise CameraUnavailable('RealSense USB identity does not match a connected camera')
    if camera.lock is not None and getattr(camera, 'usb_serial', None) != physical:
        raise CameraUnavailable('RealSense device does not match the reserved USB camera')
    if camera.lock is None:
        camera.lock = CameraLock(physical)
        camera.usb_serial = physical


def select_profiles(device, rs, fps):
    available = [profile.as_video_stream_profile()
                 for sensor in device.query_sensors()
                 for profile in sensor.get_stream_profiles() if profile.is_video_stream_profile()]
    selected = {}
    for name, stream, fmt, sizes in [
        ('color', rs.stream.color, rs.format.rgb8, [(1280, 720), (640, 480)]),
        ('depth', rs.stream.depth, rs.format.z16, [(640, 480), (848, 480), (1280, 720)]),
    ]:
        for width, height in sizes:
            profile = next((p for p in available if p.stream_type() == stream
                            and p.format() == fmt and p.fps() == fps
                            and (p.width(), p.height()) == (width, height)), None)
            if profile is not None:
                selected[name] = profile
                break
        if name not in selected:
            raise CameraUnavailable(f'RealSense has no supported {name} profile at {fps} FPS; try 15 or 30')
    depth = selected['depth']
    infrared = next((p for p in available if p.stream_type() == rs.stream.infrared
                     and p.stream_index() == 1 and p.format() == rs.format.y8
                     and p.fps() == fps and (p.width(), p.height()) == (depth.width(), depth.height())), None)
    if infrared is not None:
        selected['ir'] = infrared
    return selected


def calibration(profiles):
    result = {'schema_version': 1, 'backend': 'realsense',
              'registration': 'librealsense align to color', 'planes': {}}
    for name, profile in profiles.items():
        intrinsic = profile.get_intrinsics()
        coefficients = list(intrinsic.coeffs)
        sdk_model = str(intrinsic.model)
        # Zero coefficients are exactly pinhole, including D415's RGB profile
        # reported as inverse Brown-Conrady. Preserve the SDK declaration too.
        pinhole_models = {'distortion.none', 'distortion.brown_conrady',
                          'distortion.inverse_brown_conrady', 'distortion.modified_brown_conrady'}
        model = ('NONE' if not any(coefficients) and sdk_model.lower() in pinhole_models else 'BROWN_CONRADY'
                 if sdk_model.lower() == 'distortion.brown_conrady'
                 else 'REALSENSE_' + sdk_model.rsplit('.', 1)[-1].upper())
        distortion = dict(zip(['k1', 'k2', 'p1', 'p2', 'k3'], coefficients))
        distortion.update(model=model, k4=0., k5=0., k6=0.)
        plane = {
            'profile': {'width': profile.width(), 'height': profile.height(),
                        'fps': profile.fps(), 'format': str(profile.format())},
            'intrinsics': {'width': intrinsic.width, 'height': intrinsic.height,
                           'fx': intrinsic.fx, 'fy': intrinsic.fy, 'cx': intrinsic.ppx, 'cy': intrinsic.ppy},
            'distortion': distortion,
            'sdk_distortion': {'model': sdk_model, 'coefficients': coefficients},
        }
        if name != 'color':
            extrinsic = profile.get_extrinsics_to(profiles['color'])
            # librealsense stores rotation column-major and translation in metres.
            plane['extrinsics_to_color'] = {
                'rotation': np.asarray(extrinsic.rotation).reshape(3, 3, order='F').ravel().tolist(),
                'translation_mm': (np.asarray(extrinsic.translation) * 1000).tolist(),
            }
        result['planes'][name] = plane
    return result


def capture(camera):
    camera.reserve()
    import pyrealsense2 as rs
    context = rs.context()
    devices = list(context.query_devices())
    if camera.serial:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == camera.serial]
    if len(devices) != 1:
        raise CameraUnavailable('No matching unique RealSense device; select --camera-serial')
    device = devices[0]
    claim_device(camera, device, rs)
    serial = device.get_info(rs.camera_info.serial_number)
    camera.serial = serial
    camera.info = {'name': device.get_info(rs.camera_info.name), 'serial': serial,
                   'usb_serial': camera.usb_serial,
                   'firmware': device.get_info(rs.camera_info.firmware_version)}
    selected = select_profiles(device, rs, camera.fps)
    config = rs.config()
    config.enable_device(serial)
    for profile in selected.values():
        config.enable_stream(profile.stream_type(), profile.stream_index(),
                             profile.width(), profile.height(), profile.format(), profile.fps())
    pipeline = rs.pipeline(context)
    started = False
    try:
        active = pipeline.start(config)
        started = True
        profiles = {name: active.get_stream(p.stream_type(), p.stream_index()).as_video_stream_profile()
                    for name, p in selected.items()}
        scale = float(active.get_device().first_depth_sensor().get_depth_scale()) * 1000
        if not np.isfinite(scale) or scale <= 0:
            raise CameraUnavailable('Invalid RealSense depth scale')
        camera.calibration = calibration(profiles)
        camera.calibration_hash = hashlib.sha256(encode(camera.calibration).encode()).hexdigest()
        camera.session_id = str(uuid4())
        align = rs.align(rs.stream.color)
        sequence = 0
        misses = 0
        while not camera.stop_event.is_set():
            frames = pipeline.wait_for_frames(1000)
            received_at, received_monotonic = now(), time.monotonic()
            camera.counters['received'] += 1
            planes = {'color': frames.get_color_frame(), 'depth': frames.get_depth_frame()}
            if 'ir' in profiles:
                planes['ir'] = frames.get_infrared_frame(1)
            if not all(planes.values()):
                camera.counters['incomplete'] += 1
                misses += 1
                if misses >= 5:
                    raise CameraUnavailable('No complete RealSense frames received')
                continue
            domains = {name: str(frame.get_frame_timestamp_domain()) for name, frame in planes.items()}
            if len(set(domains.values())) != 1:
                camera.counters['incomplete'] += 1
                continue
            timestamps = {name: round(frame.get_timestamp() * 1000) for name, frame in planes.items()}
            if not camera.accept_timestamps(timestamps, expected=set(profiles)):
                continue
            color = cv2.cvtColor(np.asanyarray(planes['color'].get_data()), cv2.COLOR_RGB2BGR)
            depth = np.asanyarray(planes['depth'].get_data()).copy()
            ir = np.asanyarray(planes['ir'].get_data()).copy() if 'ir' in planes else None
            aligned = align.process(frames).get_depth_frame()
            if not aligned:
                raise CameraUnavailable('RealSense alignment returned no depth')
            registered = np.asanyarray(aligned.get_data()).copy()
            expected_depth = (profiles['depth'].height(), profiles['depth'].width())
            if (depth.dtype != np.uint16 or depth.shape != expected_depth
                    or registered.dtype != np.uint16 or registered.shape != color.shape[:2]):
                raise CameraUnavailable('Invalid RealSense native or registered depth plane')
            if ir is not None and (ir.dtype != np.uint8 or ir.shape != expected_depth):
                raise CameraUnavailable('Invalid RealSense native IR plane')
            sequence += 1
            misses = 0
            metadata = {
                'frame_id': str(uuid4()), 'session_id': camera.session_id, 'source_id': 'femto-bolt',
                'backend': 'realsense', 'sequence': sequence, 'received_at': received_at,
                'received_monotonic_s': received_monotonic, 'timestamps_us': timestamps,
                'timestamp_domain': next(iter(domains.values())),
                'sdk_timestamp_domains': domains,
                'channel_skew_ms': (max(timestamps.values()) - min(timestamps.values())) / 1000,
                'sdk_frame_indices': {name: frame.get_frame_number() for name, frame in planes.items()},
                'profiles': {name: plane['profile'] for name, plane in camera.calibration['planes'].items()},
                'depth_scale_mm': scale, 'registered_depth_scale_mm': scale,
                'depth_units': 'raw_uint16 * plane_depth_scale_mm / 1000 = metres',
                'calibration': camera.calibration, 'calibration_sha256': camera.calibration_hash,
                'device': camera.info,
            }
            camera.publish(make_frame(color, depth, ir, registered, metadata))
    finally:
        if started:
            pipeline.stop()
