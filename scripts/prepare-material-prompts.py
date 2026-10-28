#!/usr/bin/env python3
"""Prepare a private material-prompt comparison using an unchanged vision graph."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('Use a separate experimental output directory')
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor
    manifest = json.loads((args.source/'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if filename not in {'vision.xml','vision.bin','text_embeddings.npy'}:
            raise ValueError('Unexpected source artifact')
        with (args.source/filename).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
                raise ValueError('Source artifact checksum mismatch')
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(manifest['model'],revision=manifest['revision'],
                                     attn_implementation='eager').eval()
    processor = AutoProcessor.from_pretrained(manifest['model'],revision=manifest['revision'],use_fast=False)
    vectors = []; prompts = []
    for index,label in enumerate(manifest['labels']):
        if label in manifest['rejection_labels']:
            alternatives = [manifest['prompts'][index]]
        else:
            alternatives = [
                f'This is a photo of {label}.',
                f'This is a close-up photo of a {label} surface.',
                f'This is a photo of {label} material.',
                manifest['prompts'][index],
            ]
        inputs = processor(text=alternatives,padding='max_length',
                           max_length=manifest['text_max_length'],return_tensors='pt')
        with torch.inference_mode():
            encoded = torch.nn.functional.normalize(model.get_text_features(**inputs),dim=-1)
            averaged = torch.nn.functional.normalize(encoded.mean(dim=0),dim=-1)
        vectors.append(averaged.numpy());prompts.append(alternatives)
    args.output.mkdir(parents=True,exist_ok=True)
    for filename in ['vision.xml','vision.bin']:
        shutil.copyfile(args.source/filename,args.output/filename)
    np.save(args.output/'text_embeddings.npy',np.stack(vectors),allow_pickle=False)
    files = {}
    for filename in ['vision.xml','vision.bin','text_embeddings.npy']:
        with (args.output/filename).open('rb') as stream:
            files[filename] = hashlib.file_digest(stream,'sha256').hexdigest()
    experimental = {**manifest,'prompts':prompts,'files':files,
                    'prompt_aggregation':'mean of normalized prompt vectors, then normalize; rejection prompts unchanged',
                    'source_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
                    'experimental_policy':'original score thresholds retained; no camera-domain calibration'}
    (args.output/'manifest.json').write_text(json.dumps(experimental,indent=2)+'\n')
    print(json.dumps({'model':manifest['model'],'output':str(args.output),
                      'vision_graph_unchanged':files['vision.bin']==manifest['files']['vision.bin']}))


if __name__ == '__main__':
    main()
