"""Real offline inference: no mocked runtimes, downloaded weights or cameras."""
import hashlib
import json
from pathlib import Path

import boto3
from botocore.stub import Stubber
import cv2
import numpy as np
import onnxruntime
import openvino as ov
import pyorbbecsdk
import pyrealsense2
import pytest

from board.yoloe import Detector, prepare
from rgb.materials import MaterialClassifier
from rgb.vlm_providers import BedrockProvider, ReviewRequest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', params=['yoloe-26n-pf-320', 'yoloe-26n-materials-320'])
def detectors(request):
    directory = ROOT / 'models' / request.param
    return [Detector(directory / 'model.onnx', directory / 'manifest.json',
                     threshold=0, backend=backend, threads=2)
            for backend in ['onnxruntime', 'openvino']]


def test_yoloe_real_runtimes_agree_before_decoding(detectors):
    # A fixed textured scene exercises nonzero outputs without claiming detection accuracy.
    image = np.random.default_rng(7).integers(0, 256, (180, 320, 3), dtype=np.uint8)
    ort, vino = detectors
    tensor, _ = prepare(image, ort.size)
    outputs_ort = ort.session.run(None, {ort.input_name: tensor})
    outputs_vino = vino.session([tensor])
    assert len(outputs_ort) == len(vino.session.outputs)
    # Compare rows without relying on equal confidence tie ordering.
    actual = outputs_vino[vino.output_port][0]
    expected = outputs_ort[0][0]
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    # Match classes, then nearest boxes (top-k ordering can differ across kernels).
    meaningful = expected[expected[:, 4] >= .01]
    assert len(meaningful) > 0
    for row in meaningful:
        candidates = actual[actual[:, 5] == row[5]]
        assert len(candidates)
        distance = np.max(np.abs(candidates[:, :4] - row[:4]), axis=1)
        match = candidates[np.argmin(distance)]
        assert np.min(distance) < .5
        assert abs(float(match[4] - row[4])) < .005
    if len(outputs_ort) > 1 and not np.allclose(
            outputs_vino[vino.session.output(1)], outputs_ort[1], atol=.005, rtol=.005):
        model_path = next(directory / 'model.onnx' for directory in (ROOT / 'models').iterdir()
                          if (directory / 'model.onnx').is_file() and hashlib.sha256(
                              (directory / 'model.onnx').read_bytes()).hexdigest() == ort.sha256)
        for level in [onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL,
                      onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC]:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            options.graph_optimization_level = level
            options.add_session_config_entry('mlas.enable_gemm_fastmath_arm64_bfloat16', '0')
            session = onnxruntime.InferenceSession(
                str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
            proto = session.run(None, {ort.input_name: tensor})[1]
            print('DIAGNOSTIC', level, 'vs optimized ORT',
                  float(np.max(np.abs(proto - outputs_ort[1]))), 'vs OpenVINO',
                  float(np.max(np.abs(proto - outputs_vino[vino.session.output(1)]))))
    if len(outputs_ort) > 1:
        np.testing.assert_allclose(
            outputs_vino[vino.session.output(1)], outputs_ort[1], atol=.005, rtol=.005)


@pytest.mark.parametrize('with_masks', [False, True])
def test_yoloe_runtime_detections_and_mask_contract(detectors, with_masks):
    image = np.random.default_rng(7).integers(0, 256, (180, 320, 3), dtype=np.uint8)
    for detector in detectors:
        detections, elapsed = detector.detect(image, with_masks=with_masks)
        assert detections and 0 <= elapsed < 120000
        assert len(detector.instance_masks) == len(detections)
        assert detector.mask_error is None
        for detection, mask in zip(detections, detector.instance_masks):
            y1, x1, y2, x2 = detection['box']
            assert 0 <= y1 < y2 <= 1 and 0 <= x1 < x2 <= 1
            assert detection['label'] == detector.labels[detection['class_id']]
            assert 0 <= detection['confidence'] <= 1
            assert isinstance(detection['confidence'], float)
            if with_masks:
                assert mask is not None and mask.dtype == bool and mask.shape == image.shape[:2]
            else:
                assert mask is None
        json.dumps(detections, allow_nan=False)


@pytest.fixture
def student(tmp_path):
    # An intentionally tiny encoder exercises real OpenVINO loading/inference.
    # This is a runtime fixture, not a trained material model or accuracy test.
    pixels = ov.opset13.parameter([1, 3, 16, 16], np.float32, name='pixel_values')
    embedding = ov.opset13.reduce_mean(pixels, [2, 3], False)
    graph = ov.Model([embedding], [pixels], 'synthetic-material-encoder')
    ov.save_model(graph, tmp_path / 'vision.xml', compress_to_fp16=False)
    np.save(tmp_path / 'text_embeddings.npy', np.eye(3, dtype=np.float32), allow_pickle=False)
    np.savez(tmp_path / 'classifier.npz', weight=np.eye(3, dtype=np.float32) * 12,
             bias=np.zeros(3, dtype=np.float32))
    manifest = {
        'model': 'synthetic-ci-student', 'revision': 'fixture-v1', 'input_size': 16,
        'embedding_size': 3, 'labels': ['plastic', 'metal', 'unknown'],
        'prompts': ['plastic', 'metal', 'unknown'], 'rejection_labels': ['unknown'],
        'processor': {'do_resize': True, 'do_rescale': True, 'do_normalize': True,
                      'do_center_crop': False, 'resample': 3, 'rescale_factor': 1 / 255,
                      'image_mean': [0, 0, 0], 'image_std': [1, 1, 1]},
        'minimum_similarity': .2, 'minimum_margin': .02,
        'classifier': {'kind': 'linear_softmax', 'file': 'classifier.npz', 'temperature': 1.,
                       'minimum_probability': .8, 'minimum_margin': .05,
                       'qualified_for_acceptance': True},
        'training': {'train_material_counts': {'plastic': 10, 'metal': 10}},
    }
    manifest['files'] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                         for name in ['vision.xml', 'vision.bin', 'text_embeddings.npy', 'classifier.npz']}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return tmp_path, manifest


@pytest.mark.parametrize(('rgb', 'label'), [((255, 0, 0), 'plastic'), ((0, 255, 0), 'metal'),
                                          ((0, 0, 255), None), ((255, 255, 0), None)])
def test_student_runs_offline_and_preserves_acceptance_and_rejection(student, rgb, label):
    directory, manifest = student
    classifier = MaterialClassifier(directory)
    result = classifier.classify(np.full((24, 32, 3), rgb, np.uint8))
    assert result['label'] == label
    assert result['status'] == ('identified' if label else 'uncertain')
    assert result['score_type'] == 'supervised_material_head_score'
    assert result['head_sha256'] == manifest['files']['classifier.npz']
    assert result['runtime'] == 'openvino' and result['device'] == 'CPU'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('name', ['vision.xml', 'vision.bin', 'text_embeddings.npy', 'classifier.npz'])
def test_student_rejects_corrupt_artifacts_before_compilation(student, name):
    directory, _ = student
    with (directory / name).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        MaterialClassifier(directory)


def test_orbbec_installed_sdk_can_initialize_without_usb_hardware():
    # Import/init the shipped wheel and its native libraries; never open a camera.
    context = pyorbbecsdk.Context()
    context.set_logger_level(pyorbbecsdk.OBLogLevel.ERROR)
    devices = context.query_devices()
    assert devices.get_count() >= 0
    assert onnxruntime.get_available_providers()
    assert cv2.imencode('.jpg', np.zeros((16, 16, 3), np.uint8))[0]


def test_realsense_installed_sdk_can_initialize_without_usb_hardware():
    context = pyrealsense2.context()
    assert len(context.query_devices()) >= 0


def test_bedrock_uses_real_botocore_protocol_without_network():
    # Stub the transport after SDK validation, with explicit synthetic credentials.
    client = boto3.client('bedrock-runtime', region_name='us-west-2',
                          aws_access_key_id='ci-placeholder', aws_secret_access_key='ci-placeholder')
    with Stubber(client) as stubber:
        stubber.add_response('converse', {
            'output': {'message': {'role': 'assistant', 'content': [{'text': '{"readings":[]}'}]}},
            'stopReason': 'end_turn', 'usage': {'inputTokens': 1, 'outputTokens': 1, 'totalTokens': 2},
            'metrics': {'latencyMs': 1},
            'ResponseMetadata': {'HTTPStatusCode': 200, 'RequestId': 'synthetic-ci'},
        })
        # See the existing provider unit tests for error/fallback/response contracts.
        provider = BedrockProvider('default', 'us-west-2', client=client)
        jpeg = cv2.imencode('.jpg', np.zeros((16, 16, 3), np.uint8))[1].tobytes()
        result = provider.review(ReviewRequest(
            model='us.amazon.nova-2-lite-v1:0', prompt='Synthetic protocol check',
            system='Return JSON.', images=(('Synthetic RGB scene', jpeg),), max_tokens=100))
        assert result.text == '{"readings":[]}'
        stubber.assert_no_pending_responses()
