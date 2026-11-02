"""Qualify deployment images with real Redis on an isolated, unpublished network."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from uuid import uuid4

REDIS_IMAGE = 'redis:7.4.9-alpine@sha256:6ab0b6e7381779332f97b8ca76193e45b0756f38d4c0dcda72dbb3c32061ab99'
ROOT = Path(__file__).resolve().parents[1]


def docker(*args, capture=False):
    result = subprocess.run(['docker', *map(str, args)], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-image', default='rgb-classifier:ci')
    parser.add_argument('--board-image', default='rgb-board-yoloe:ci')
    args = parser.parse_args()
    prefix = 'rgb-ci-' + uuid4().hex[:12]
    network, volume = prefix + '-network', prefix + '-images'
    redis_name, service_name = prefix + '-redis', prefix + '-service'
    token, password = secrets.token_hex(32), secrets.token_hex(32)
    if os.environ.get('GITHUB_ACTIONS'):
        print('::add-mask::' + token, flush=True)
        print('::add-mask::' + password, flush=True)
    docker('pull', REDIS_IMAGE)
    try:
        docker('network', 'create', '--internal', network)
        docker('volume', 'create', volume)
        with tempfile.TemporaryDirectory(prefix='rgb-ci-') as directory:
            directory = Path(directory)
            config = directory / 'redis.conf'
            config.write_text('appendonly yes\nprotected-mode yes\nrequirepass ' + password + '\n')
            config.chmod(0o644)  # Only the private temporary parent and container can access it.
            env = directory / 'runtime.env'
            env.write_text(
                f'REDIS_URL=redis://:{password}@redis:6379/0\nINGEST_TOKEN={token}\n'
                'SOURCES_JSON={"synthetic-ci":{"location":"Synthetic CI bench"}}\n'
                'IMAGE_DIR=/data/images\nVLM_ENDPOINTS_JSON=[]\n'
                f'CI_EVENT_ID={uuid4()}\n')
            env.chmod(0o600)
            docker('run', '--rm', '--network', 'none', '--user', '0:0',
                   '--mount', f'type=volume,src={volume},dst=/data',
                   '--entrypoint', 'python', args.service_image, '-c',
                   "import os;os.chown('/data',1000,1000);os.chmod('/data',0o700)")
            docker('run', '-d', '--name', redis_name, '--network', network,
                   '--network-alias', 'redis', '--mount', f'type=bind,src={config},dst=/ci/redis.conf,readonly',
                   REDIS_IMAGE, 'redis-server', '/ci/redis.conf')
            docker('run', '-d', '--name', service_name, '--network', network,
                   '--network-alias', 'service', '--env-file', env,
                   '--mount', f'type=volume,src={volume},dst=/data', args.service_image)
            network_info = json.loads(docker('network', 'inspect', network, capture=True))[0]
            assert network_info['Internal']
            for name in [service_name, redis_name]:
                info = json.loads(docker('inspect', name, capture=True))[0]
                assert not info['HostConfig']['PortBindings'], 'Test ports must stay unpublished'
                if name == service_name:
                    assert info['Config']['User'] == '1000:1000'

            def phase(name):
                docker('run', '--rm', '--network', network, '--env-file', env,
                       '--env', 'PYTHONPATH=/app',
                       '--mount', f'type=bind,src={ROOT / "ci"},dst=/ci,readonly',
                       '--entrypoint', 'python', args.service_image, '/ci/container_contract.py', name)

            phase('initial')
            docker('stop', redis_name)
            phase('outage')
            docker('start', redis_name)
            docker('restart', service_name)
            phase('recovered')
            # The actual board dependencies, LiteRT/ONNX runtimes and model files.
            docker('run', '--rm', '--network', 'none',
                   '--env', 'PYTHONPATH=/app',
                   '--mount', f'type=bind,src={ROOT / "ci"},dst=/ci,readonly',
                   '--entrypoint', 'python', args.board_image, '/ci/board_contract.py')
    finally:
        reports = ROOT / 'ci-results'
        reports.mkdir(exist_ok=True)
        with (reports / 'container-service.log').open('w') as log:
            subprocess.run(['docker', 'logs', service_name], stdout=log, stderr=subprocess.STDOUT)
        for name in [service_name, redis_name]:
            subprocess.run(['docker', 'rm', '-f', '-v', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        for kind, name in [('volume', volume), ('network', network)]:
            subprocess.run(['docker', kind, 'rm', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
