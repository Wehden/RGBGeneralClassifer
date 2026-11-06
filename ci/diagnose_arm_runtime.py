"""Temporary native runner diagnostics for the ARM prototype parity failure."""
import copy
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import openvino as ov
from board.yoloe import prepare, _openvino_gather_contract

path = Path('models/yoloe-26n-pf-320/model.onnx')
model = onnx.load(str(path))
names = [node.output[0] for node in model.graph.node
         if node.op_type in {'Conv', 'ConvTranspose', 'Sigmoid', 'Mul', 'Resize'}]
for name in names:
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
onnx.save(model, '/tmp/rgb-diagnostic.onnx')
image = np.random.default_rng(7).integers(0, 256, (180, 320, 3), dtype=np.uint8)
tensor, _ = prepare(image, 320)
options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession('/tmp/rgb-diagnostic.onnx', sess_options=options,
                               providers=['CPUExecutionProvider'])
expected = session.run(None, {session.get_inputs()[0].name: tensor})
core = ov.Core()
graph = core.read_model('/tmp/rgb-diagnostic.onnx')
_openvino_gather_contract(graph)
compiled = core.compile_model(graph, 'CPU', {'INFERENCE_PRECISION_HINT':'f32',
    'EXECUTION_MODE_HINT':'ACCURACY','INFERENCE_NUM_THREADS':2})
actual = compiled([tensor])
for port, reference in zip(compiled.outputs, expected):
    output = actual[port]
    print('LAYER', port.get_any_name(), 'max', float(np.max(np.abs(output-reference))),
          'mismatches', np.count_nonzero(~np.isclose(output, reference, atol=.005,rtol=.005)), flush=True)
for node in compiled.get_runtime_model().get_ordered_ops():
    info = node.get_rt_info()
    print('EXEC', node.friendly_name, {key: str(info[key]) for key in info}, flush=True)
