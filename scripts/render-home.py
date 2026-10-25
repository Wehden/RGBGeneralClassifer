#!/usr/bin/env python3
"""Render an explicitly chosen immutable service image; never builds or deploys."""
import argparse
from pathlib import Path
import re

p=argparse.ArgumentParser()
p.add_argument('image',help='registry/repository@sha256:...')
p.add_argument('--output',default='/tmp/rgb-app.yaml')
a=p.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9.:/_-]+@sha256:[a-f0-9]{64}',a.image):
    p.error('An immutable image digest is required')
Path(a.output).write_text(Path('deploy/app.yaml').read_text().replace('RGB_IMAGE',a.image))
print(a.output)
