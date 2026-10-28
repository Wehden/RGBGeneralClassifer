#!/usr/bin/env python3
"""Build offline local models in a separate export environment."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlretrieve

SIGLIP_MODEL = 'google/siglip2-base-patch16-224'
SIGLIP_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
MATERIALS = ['plastic', 'metal', 'glass', 'wood', 'paper', 'cardboard',
             'fabric', 'rubber', 'ceramic', 'leather', 'stone', 'liquid']
REJECTIONS = {'human skin':'This is a photo of a person or human skin.',
              'food':'This is a photo of food.',
              'unknown':'This is a blurry photo of an unidentifiable object.'}
CHECKPOINT_SHA256 = {
    'small':'01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a',
    'medium':'f37e6c82d9d40763b254aff0f846cb5b3153c4726a2ac0003e9890f996e5ec03',
    'large':'8b0ff5ecfd5baa0ebed509efb00aacff52faae2f6770e7ffa2a223231b9653c0',
    'extra-large':'6591e328fc0012a00beba95710d159c26b0e2dd728ee04b0c8eca1fa0ff859f2',
}
SIGLIP_VARIANTS = {
    'base-224': (SIGLIP_MODEL, SIGLIP_REVISION, 224),
    'base-384': ('google/siglip2-base-patch16-384', 'f775b65a79762255128c981547af89addcfe0f88', 384),
    'large-256': ('google/siglip2-large-patch16-256', '787800c8990e6f058423089178e718139608408c', 256),
    'large-384': ('google/siglip2-large-patch16-384', '1b426889ea62b5a72bf9839009a1b184bfc9c178', 384),
    'so400m-256': ('google/siglip2-so400m-patch16-256', 'e8708ab72d125807e45b36fb7d4e0aacbb59f379', 256),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_manifest(directory, manifest):
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def export_detector(root, size, input_size=640):
    import torch
    import ultralytics
    from ultralytics import YOLOE
    if ultralytics.__version__ != '8.4.188':
        raise RuntimeError('Expected ultralytics==8.4.188')
    name = {'small': 's', 'medium': 'm', 'large': 'l', 'extra-large': 'x'}[size]
    directory = root/f'yoloe-26{name}-pf-{input_size}'
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory/f'yoloe-26{name}-seg-pf.pt'
    url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'+checkpoint.name
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.download')
        urlretrieve(url, temporary)
        temporary.replace(checkpoint)
    if digest(checkpoint)!=CHECKPOINT_SHA256[size]:
        raise ValueError('Official YOLOE checkpoint checksum mismatch')
    torch.set_num_threads(4)
    model = YOLOE(str(checkpoint))
    artifact = Path(model.export(format='onnx', imgsz=input_size, nms=False, agnostic_nms=True,
                                 dynamic=False, opset=17, simplify=True, device='cpu', max_det=100))
    shutil.copyfile(artifact, directory/'model.onnx')
    import onnx
    graph = onnx.load(directory/'model.onnx')
    output_shapes = [[d.dim_value for d in output.type.tensor_type.shape.dim] for output in graph.graph.output]
    save_manifest(directory, {
        'schema_version': 1, 'model': directory.name, 'sha256': digest(directory/'model.onnx'),
        'input_size': input_size, 'prompt_mode': 'prompt_free',
        'checkpoint': checkpoint.name, 'checkpoint_url': url, 'checkpoint_sha256': digest(checkpoint),
        'exporter': 'ultralytics=='+ultralytics.__version__, 'torch': torch.__version__,
        'opset': 17, 'precision': 'float32', 'end2end': True, 'agnostic_nms': True,
        'max_det': 100, 'output_contract': output_shapes, 'license': 'AGPL-3.0',
        'labels': [model.names[i] for i in range(len(model.names))],
    })
    print(json.dumps({'detector': directory.name, 'sha256': digest(directory/'model.onnx')}), flush=True)


def export_materials(root, variant='base-224'):
    import numpy as np
    import torch
    import openvino as ov
    import transformers
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(4)
    model_id, revision, input_size = SIGLIP_VARIANTS[variant]
    model = AutoModel.from_pretrained(model_id, revision=revision,
                                     attn_implementation='eager').eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision,
                                             use_fast=False)
    labels=MATERIALS+list(REJECTIONS)
    prompts = [f'This is a photo of an object made of {label}.' for label in MATERIALS]+list(REJECTIONS.values())
    text = processor(text=prompts, padding='max_length',
                     max_length=model.config.text_config.max_position_embeddings, return_tensors='pt')
    with torch.inference_mode():
        embeddings = model.get_text_features(**text)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).numpy()
    directory = root/f'siglip2-{variant}'
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory/'text_embeddings.npy', embeddings, allow_pickle=False)

    class Vision(torch.nn.Module):
        def __init__(self, vision):
            super().__init__()
            self.vision = vision

        def forward(self, pixels):
            result = self.vision(pixel_values=pixels).pooler_output
            return torch.nn.functional.normalize(result, dim=-1)

    wrapper = Vision(model.vision_model).eval()
    example = torch.zeros((1, 3, input_size, input_size))
    converted = ov.convert_model(wrapper, example_input=example)
    converted.reshape([1, 3, input_size, input_size])
    # FP32 storage avoids introducing an unmeasured precision change.
    ov.save_model(converted, directory/'vision.xml', compress_to_fp16=False)
    config = processor.image_processor.to_dict()
    manifest = {
        'schema_version': 1, 'model': model_id, 'revision': revision,
        'license': 'Apache-2.0', 'input_size': input_size, 'embedding_size': int(embeddings.shape[1]),
        'labels': labels, 'rejection_labels':list(REJECTIONS), 'prompts': prompts, 'processor': config,
        'text_max_length': model.config.text_config.max_position_embeddings,
        'logit_scale': float(model.logit_scale.exp().detach()),
        'logit_bias': float(model.logit_bias.detach()),
        'score_type': 'uncalibrated_image_text_similarity',
        'minimum_similarity': .20, 'minimum_margin': .02,
        'exporter': {'transformers': transformers.__version__, 'torch': torch.__version__,
                     'openvino': ov.__version__},
        'files': {name: digest(directory/name)
                  for name in ['vision.xml', 'vision.bin', 'text_embeddings.npy']},
    }
    # Qualify preprocessing and the vision graph before publishing its manifest.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rgb.materials import prepare
    from PIL import Image
    compiled = ov.Core().compile_model(converted, 'CPU', {
        'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2,
        'NUM_STREAMS': 1,
    })
    rng = np.random.default_rng(20261028)
    pixel_error = embedding_error = 0.0
    for height, width in [(80, 120), (121, 79), (224, 224)]:
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        actual_pixels = prepare(rgb, manifest)
        source_pixels = processor(images=Image.fromarray(rgb), return_tensors='pt')['pixel_values']
        pixel_error = max(pixel_error, float(np.max(np.abs(actual_pixels-source_pixels.numpy()))))
        with torch.inference_mode():
            source = wrapper(source_pixels).numpy()
        exported = compiled([actual_pixels])[compiled.output(0)]
        embedding_error = max(embedding_error, float(np.max(np.abs(source-exported))))
    if pixel_error > 2e-6 or embedding_error > 2e-4:
        raise ValueError(f'SigLIP2 export parity failed: pixels={pixel_error}, embedding={embedding_error}')
    manifest['export_verification'] = {
        'synthetic_rgb_inputs': 3, 'maximum_preprocessing_error': pixel_error,
        'maximum_embedding_error': embedding_error, 'precision': 'float32',
    }
    save_manifest(directory, manifest)
    print(json.dumps({'materials': model_id, 'revision': revision,
                      'verification': manifest['export_verification']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path.home()/'.cache/bolt-rgb/models')
    parser.add_argument('--detectors', nargs='*', choices=list(CHECKPOINT_SHA256), default=['small', 'medium'])
    parser.add_argument('--detector-input-size', type=int, choices=[640, 960], default=640)
    parser.add_argument('--material-models', nargs='+', choices=list(SIGLIP_VARIANTS), default=['base-224'])
    parser.add_argument('--skip-materials', action='store_true')
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for size in args.detectors:
        export_detector(args.models_dir, size, args.detector_input_size)
    if not args.skip_materials:
        for variant in args.material_models:
            export_materials(args.models_dir, variant)


if __name__ == '__main__':
    main()
