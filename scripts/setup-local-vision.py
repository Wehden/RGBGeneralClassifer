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
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_manifest(directory, manifest):
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def export_detector(root, size):
    import torch
    import ultralytics
    from ultralytics import YOLOE
    if ultralytics.__version__ != '8.4.188':
        raise RuntimeError('Expected ultralytics==8.4.188')
    name = {'small': 's', 'medium': 'm'}[size]
    directory = root/f'yoloe-26{name}-pf-640'
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
    artifact = Path(model.export(format='onnx', imgsz=640, nms=False, agnostic_nms=True,
                                 dynamic=False, opset=17, simplify=True, device='cpu', max_det=100))
    shutil.copyfile(artifact, directory/'model.onnx')
    import onnx
    graph = onnx.load(directory/'model.onnx')
    output_shapes = [[d.dim_value for d in output.type.tensor_type.shape.dim] for output in graph.graph.output]
    save_manifest(directory, {
        'schema_version': 1, 'model': directory.name, 'sha256': digest(directory/'model.onnx'),
        'input_size': 640, 'prompt_mode': 'prompt_free',
        'checkpoint': checkpoint.name, 'checkpoint_url': url, 'checkpoint_sha256': digest(checkpoint),
        'exporter': 'ultralytics=='+ultralytics.__version__, 'torch': torch.__version__,
        'opset': 17, 'precision': 'float32', 'end2end': True, 'agnostic_nms': True,
        'max_det': 100, 'output_contract': output_shapes, 'license': 'AGPL-3.0',
        'labels': [model.names[i] for i in range(len(model.names))],
    })
    print(json.dumps({'detector': directory.name, 'sha256': digest(directory/'model.onnx')}), flush=True)


def export_materials(root):
    import numpy as np
    import torch
    import openvino as ov
    import transformers
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(SIGLIP_MODEL, revision=SIGLIP_REVISION,
                                     attn_implementation='eager').eval()
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, revision=SIGLIP_REVISION,
                                             use_fast=False)
    labels=MATERIALS+list(REJECTIONS)
    prompts = [f'This is a photo of an object made of {label}.' for label in MATERIALS]+list(REJECTIONS.values())
    text = processor(text=prompts, padding='max_length',
                     max_length=model.config.text_config.max_position_embeddings, return_tensors='pt')
    with torch.inference_mode():
        embeddings = model.get_text_features(**text)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).numpy()
    directory = root/'siglip2-base-224'
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
    example = torch.zeros((1, 3, 224, 224))
    converted = ov.convert_model(wrapper, example_input=example)
    converted.reshape([1, 3, 224, 224])
    # FP32 storage avoids introducing an unmeasured precision change.
    ov.save_model(converted, directory/'vision.xml', compress_to_fp16=False)
    config = processor.image_processor.to_dict()
    save_manifest(directory, {
        'schema_version': 1, 'model': SIGLIP_MODEL, 'revision': SIGLIP_REVISION,
        'license': 'Apache-2.0', 'input_size': 224, 'embedding_size': int(embeddings.shape[1]),
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
    })
    print(json.dumps({'materials': SIGLIP_MODEL, 'revision': SIGLIP_REVISION}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path.home()/'.cache/bolt-rgb/models')
    parser.add_argument('--detectors', nargs='*', choices=['small', 'medium'], default=['small', 'medium'])
    parser.add_argument('--skip-materials', action='store_true')
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for size in args.detectors:
        export_detector(args.models_dir, size)
    if not args.skip_materials:
        export_materials(args.models_dir)


if __name__ == '__main__':
    main()
