import json
import logging
import os

from redis import Redis
from waitress import serve
from rgb.enrichment import start
from rgb.store import Store
from rgb.web import create_app

logging.basicConfig(level=logging.INFO)
endpoints = json.loads(os.getenv('VLM_ENDPOINTS_JSON','[]'))
store = Store(Redis.from_url(os.environ['REDIS_URL'], decode_responses=True, socket_connect_timeout=3, socket_timeout=10), os.getenv('IMAGE_DIR','/data/images'), analysis_enabled=bool(endpoints))
app = create_app(store)
if __name__ == '__main__':
    if os.getenv('ROLE') != 'ingest':
        start(store, endpoints)
    serve(app, host='0.0.0.0', port=8080, threads=16, channel_timeout=30)
