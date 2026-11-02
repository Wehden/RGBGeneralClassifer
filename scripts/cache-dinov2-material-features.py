#!/usr/bin/env python3
"""Pinned public DINOv2 source; offline frozen CLS/patch feature extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

MODEL = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'
FILES = ['config.json', 'preprocessor_config.json', 'model.safetensors']
SOURCES = {
    'small': (MODEL, REVISION),
    'base': ('facebook/dinov2-base', 'f9e44c814b77203eaa57a6bdbbd535f21ede1415'),
    'large': ('facebook/dinov2-large', '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'),
}


def square_crop(image):
    """Match SigLIP2's RGB/PIL bicubic 384 square before model normalization."""
    return image.convert('RGB').resize((384, 384), Image.Resampling.BICUBIC)


def prepare_inputs(processor, images, mode):
    if mode == 'canonical':
        return processor(images=images, return_tensors='pt')
    if mode != 'siglip384':
        raise ValueError('Unsupported crop mode')
    # Do not let the source processor resize or centre-crop the matched square.
    inputs = processor(images=[square_crop(image) for image in images],
                       do_resize=False, do_center_crop=False, return_tensors='pt')
    if tuple(inputs['pixel_values'].shape[-2:]) != (384, 384):
        raise ValueError('Matched crop input shape changed')
    return inputs


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalized_features(tokens):
    if tokens.ndim != 3 or tokens.shape[1] < 2 or not np.isfinite(tokens).all():
        raise ValueError('Expected finite CLS and patch tokens')
    cls = tokens[:, 0].astype(np.float32)
    patches = tokens[:, 1:].mean(axis=1).astype(np.float32)
    def normalize(values):
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(lengths < 1e-12):
            raise ValueError('Zero feature vector')
        return values/lengths
    return normalize(cls), normalize(patches)


def extract(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty feature directory; preserve prior measurements')
    import torch
    from transformers import AutoImageProcessor, AutoModel
    source = args.source_directory.resolve()
    model_name, revision = SOURCES[args.model_size]
    if source.name != revision:
        raise ValueError('Use the pinned snapshot directory returned by the cache stage')
    model = AutoModel.from_pretrained(str(source), local_files_only=True).eval()
    processor = AutoImageProcessor.from_pretrained(str(source), local_files_only=True, use_fast=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    rows = json.loads(args.rows.read_text())
    if not rows:
        raise ValueError('No sample rows')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample identity')
    cls_features, patch_features = [], []
    shapes = set()
    token_counts = set()
    for start in range(0, len(rows), args.batch_size):
        selected = rows[start:start+args.batch_size]
        images = []
        for row in selected:
            path = (args.data/row['file']).resolve()
            if not path.is_relative_to(args.data.resolve()) or checksum(path) != row['sha256']:
                raise ValueError('Dataset pixel identity mismatch')
            with Image.open(path) as source_image:
                images.append(source_image.convert('RGB'))
        inputs = prepare_inputs(processor, images, args.crop_mode)
        shapes.add(tuple(inputs['pixel_values'].shape[-2:]))
        with torch.inference_mode():
            tokens = model(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state.float().cpu().numpy()
        token_counts.add(tokens.shape[1])
        cls, patches = normalized_features(tokens)
        cls_features.append(cls); patch_features.append(patches)
    ids = np.array([row['id'] for row in rows])
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, values in [('cls', cls_features), ('mean_patch', patch_features)]:
        np.savez_compressed(args.output/(name+'-features.npz'), features=np.concatenate(values), ids=ids)
    patch_size = model.config.patch_size
    receipt = {'model': model_name, 'revision': revision, 'license': 'Apache-2.0', 'source_model_type': model.config.model_type,
               'processor': processor.to_dict(), 'source_files': {name: checksum(source/name) for name in FILES},
               'rows_sha256': checksum(args.rows), 'samples': len(rows), 'device': str(device),
               'extractor_sha256': checksum(Path(__file__)),
               'embedding_size': cls_features[0].shape[1], 'torch': torch.__version__,
               'scope': 'frozen feature caching only; no classifier selection or historical test evaluation',
               'pooling': ['CLS token', 'mean patch tokens excluding CLS'],
               'preprocessing': {
                   'mode': args.crop_mode, 'input_shapes_hw': sorted(shapes),
                   'square_resize': {'size': 384, 'resample': 'PIL bicubic', 'resample_enum': 3}
                                    if args.crop_mode == 'siglip384' else None,
                   'center_crop_enabled': args.crop_mode == 'canonical' and processor.do_center_crop,
                   'normalization': 'source DINOv2 mean/std retained',
                   'patch_size': patch_size, 'token_counts_including_cls': sorted(token_counts),
                   'patch_grid_hw': [[h//patch_size, w//patch_size] for h, w in sorted(shapes)],
                   'unsampled_bottom_right_hw': [[h % patch_size, w % patch_size] for h, w in sorted(shapes)],
               },
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'source_weight_bytes': (source/'model.safetensors').stat().st_size,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
               'batch_size': args.batch_size,
               'feature_files': {name: checksum(args.output/name) for name in ['cls-features.npz', 'mean_patch-features.npz']}}
    (args.output/'feature-manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: receipt[key] for key in ['model', 'revision', 'samples', 'device', 'embedding_size']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    caching = stages.add_parser('cache', help='Explicit download of pinned public source')
    caching.add_argument('--model-size', choices=list(SOURCES), default='small')
    command = stages.add_parser('extract', help='Offline feature extraction; use a container for GPU jobs')
    command.add_argument('--source-directory', type=Path, required=True)
    command.add_argument('--data', type=Path, required=True)
    command.add_argument('--rows', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    command.add_argument('--batch-size', type=int, default=16)
    command.add_argument('--model-size', choices=list(SOURCES), default='small')
    command.add_argument('--crop-mode', choices=['canonical', 'siglip384'], default='canonical')
    args = parser.parse_args()
    if args.stage == 'cache':
        from huggingface_hub import snapshot_download
        model_name, revision = SOURCES[args.model_size]
        source = snapshot_download(model_name, revision=revision, allow_patterns=FILES)
        print(json.dumps({'model': model_name, 'revision': revision, 'directory': source,
                          'files': {name: checksum(Path(source)/name) for name in FILES}}))
    else:
        if args.batch_size < 1:
            parser.error('batch-size must be positive')
        started = time.monotonic()
        extract(args)
        print(json.dumps({'elapsed_s': time.monotonic()-started}))


if __name__ == '__main__':
    main()
