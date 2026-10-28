#!/usr/bin/env python3
"""Compare quantitative native-depth backprojection with the attached camera SDK."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgb.camera import Camera, _calibration
from rgb.metrology import backproject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    import pyorbbecsdk as ob
    owner = Camera(fps=15)
    pipeline = None;started = False
    try:
        owner.reserve()
        context = ob.Context();context.set_logger_level(ob.OBLogLevel.ERROR)
        devices = context.query_devices()
        selected = [devices.get_device_by_index(i) for i in range(devices.get_count())
                    if devices.get_device_by_index(i).get_device_info().get_serial_number() == owner.serial]
        if len(selected) != 1:
            raise RuntimeError('No unique camera matching the singleton reservation')
        pipeline = ob.Pipeline(selected[0]);config = ob.Config();profiles = {}
        for name,sensor,w,h,fmt in [
            ('color',ob.OBSensorType.COLOR_SENSOR,1280,720,ob.OBFormat.MJPG),
            ('depth',ob.OBSensorType.DEPTH_SENSOR,640,576,ob.OBFormat.Y16),
            ('ir',ob.OBSensorType.IR_SENSOR,640,576,ob.OBFormat.Y16),
        ]:
            profile = pipeline.get_stream_profile_list(sensor).get_video_stream_profile(w,h,fmt,15)
            profiles[name] = profile;config.enable_stream(profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        pipeline.enable_frame_sync();pipeline.start(config);started = True
        calibration = _calibration(profiles)
        camera_param = pipeline.get_camera_param()
        point_filter = ob.PointCloudFilter()
        point_filter.set_camera_param(camera_param)
        point_filter.set_create_point_format(ob.OBFormat.POINT)
        deadline = time.monotonic()+20
        frames = None
        while time.monotonic() < deadline:
            candidate = pipeline.wait_for_frames(1000)
            if candidate is not None and candidate.get_depth_frame() is not None:
                frames = candidate;break
        if frames is None:
            raise RuntimeError('No depth frame received')
        source = frames.get_depth_frame()
        depth = np.frombuffer(source.get_data(),np.uint16).reshape(source.get_height(),source.get_width()).copy()
        scale = float(source.get_depth_scale())
        point_filter.set_position_data_scaled(scale/1000)
        point_frame = point_filter.process(frames)
        sdk = np.asarray(point_filter.calculate(point_frame)).copy()
        if sdk.size != depth.size*3:
            raise RuntimeError('Unexpected SDK point layout: '+str(sdk.shape))
        sdk = sdk.reshape(depth.shape[0],depth.shape[1],3)
        points,pixels = backproject(depth,calibration['planes']['depth'],scale)
        indices = pixels.astype(np.int64)
        references = sdk[indices[:,1],indices[:,0]]
        valid = np.isfinite(references).all(axis=1) & (references[:,2] > 0)
        if valid.sum() < 100:
            raise RuntimeError('Insufficient corresponding SDK points')
        errors = np.linalg.norm(points[valid]-references[valid],axis=1)
        raw_plane = {**calibration['planes']['depth'],'distortion':{'model':'NONE'}}
        pinhole,_ = backproject(depth,raw_plane,scale)
        pinhole_errors = np.linalg.norm(pinhole[valid]-references[valid],axis=1)
        rotation = np.asarray(calibration['planes']['depth']['extrinsics_to_color']['rotation']).reshape(3,3)
        param_rotation = np.asarray(camera_param.transform.rot).reshape(3,3)
        document = {
            'state':'passed' if np.percentile(errors,95) < .001 and errors.max() < .002 else 'failed',
            'scope':'numerical SDK parity; no physical measurement accuracy claim',
            'samples':int(valid.sum()),'depth_scale_mm':scale,
            'native_distorted_p50_error_m':float(np.median(errors)),
            'native_distorted_p95_error_m':float(np.percentile(errors,95)),
            'native_distorted_max_error_m':float(errors.max()),
            'pinhole_without_distortion_p95_error_m':float(np.percentile(pinhole_errors,95)),
            'profile_rotation':rotation.tolist(),
            'pipeline_rotation':param_rotation.tolist(),
            'profile_rotation_orthogonality_error':float(np.max(np.abs(rotation.T@rotation-np.eye(3)))),
            'pipeline_rotation_orthogonality_error':float(np.max(np.abs(param_rotation.T@param_rotation-np.eye(3)))),
            'calibration':calibration,
        }
        np.savez_compressed(args.output_dir/'sdk-native-depth.npz',depth_native=depth,sdk_points_m=sdk)
        (args.output_dir/'sdk-native-parity.json').write_text(json.dumps(document,indent=2)+'\n')
        print(json.dumps({key:value for key,value in document.items()
                          if key not in ['calibration','profile_rotation','pipeline_rotation']}))
        if document['state'] != 'passed':
            raise SystemExit('Native SDK parity exceeded the numerical tolerance')
    finally:
        if started:pipeline.stop()
        if owner.lock:owner.lock.close()


if __name__ == '__main__':
    main()
