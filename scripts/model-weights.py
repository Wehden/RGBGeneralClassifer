#!/usr/bin/env python3
"""Publish verified exports or download a pinned private Hugging Face release."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgb.model_artifacts import ArtifactError, REPOSITORY, fetch, publish, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('validate', help='Check an export locally; no Hub dependency needed')
    check.add_argument('directory', type=Path)
    upload = sub.add_parser('publish', help='Publish only manifest-listed inference files')
    upload.add_argument('directory', type=Path)
    pull = sub.add_parser('fetch', help='Verify and install a full commit-pinned release')
    pull.add_argument('--revision', required=True)
    pull.add_argument('--cache-dir', type=Path, default=Path.home()/'.cache/bolt-rgb/models')
    pull.add_argument('--activate', action='store_true', help='Atomically select this student for the next service start')
    for command in [upload, pull]:
        command.add_argument('--repo', default=REPOSITORY)
        command.add_argument('--token-file', type=Path,
                             help='Read token from this file; otherwise use HF_TOKEN or normal Hub credentials')
    args = parser.parse_args()
    if args.command == 'validate':
        _, hashes = validate(args.directory)
        return {'directory':str(args.directory), 'files':hashes}
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError:
        raise ArtifactError('Install requirements-artifacts.txt in a tooling environment') from None
    token = args.token_file.expanduser().read_text().strip() if args.token_file else None
    if args.token_file and not token:
        raise ArtifactError('Token file is empty')
    if args.command == 'publish':
        return publish(args.directory, HfApi(token=token), CommitOperationAdd, repo=args.repo)
    return fetch(args.revision, args.cache_dir, partial(hf_hub_download, token=token),
                 repo=args.repo, activate=args.activate)


if __name__ == '__main__':
    try:
        print(json.dumps(main(), indent=2))
    except ArtifactError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # Remote SDK exceptions may include request details. Never print credentials or tracebacks.
        print('Weight operation failed ('+type(exc).__name__+'); no credential details displayed', file=sys.stderr)
        sys.exit(1)
