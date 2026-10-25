from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from redis.exceptions import ConnectionError
from rgb.enrichment import run


class Finished(BaseException):
    pass


def test_startup_redis_outage_retries(monkeypatch):
    redis=Mock()
    redis.get.side_effect=[ConnectionError('startup'),None]
    redis.xread.side_effect=Finished()
    monkeypatch.setattr('rgb.enrichment.time.sleep',lambda seconds:None)
    with pytest.raises(Finished):
        run(SimpleNamespace(redis=redis),[])
    assert redis.get.call_count == 2
    redis.xread.assert_called_once()
