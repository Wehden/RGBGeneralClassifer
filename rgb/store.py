"""Redis stream plus immutable images. No automatic observation expiry."""
import hashlib
import json
import os
from pathlib import Path

# Atomically deduplicate and append. IDs remain forever so retries cannot duplicate.
APPEND = """
local existing = redis.call('HGET', KEYS[1], ARGV[1])
if existing then return {existing, 0} end
local id = redis.call('XADD', KEYS[2], '*', 'event', ARGV[2])
redis.call('HSET', KEYS[1], ARGV[1], id)
return {id, 1}
"""


class Store:
    def __init__(self, redis, root):
        self.redis = redis
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, event, image):
        # Content addressing keeps conflicting/retried payloads from replacing images.
        digest = hashlib.sha256(image).hexdigest()
        path = self.root / (digest+'.jpg')
        if not path.exists():
            tmp = self.root / (digest+'.'+os.urandom(6).hex()+'.tmp')
            try:
                with tmp.open('xb') as f:
                    f.write(image)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        event['image_sha256'] = digest
        row_id, created = self.redis.eval(APPEND, 2, 'rgb:ids', 'rgb:events', event['event_id'], json.dumps(event))
        return row_id, bool(created)

    def get(self, event_id):
        row_id = self.redis.hget('rgb:ids', event_id)
        if not row_id:
            return None
        rows = self.redis.xrange('rgb:events', row_id, row_id)
        if not rows:
            return None
        event = json.loads(rows[0][1]['event'])
        opinions = self.redis.hget('rgb:opinions', event_id)
        if opinions:
            event['opinions'] = json.loads(opinions)
        return event

    def list(self, before=None, limit=50):
        rows = self.redis.xrevrange('rgb:events', max='('+before if before else '+', count=limit)
        return [{'cursor':row_id, **self.get(json.loads(fields['event'])['event_id'])} for row_id, fields in rows]
