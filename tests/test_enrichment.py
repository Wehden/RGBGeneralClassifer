import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from redis.exceptions import ConnectionError
from rgb.enrichment import run


class Finished(BaseException):
    pass


def test_startup_redis_outage_retries(monkeypatch):
    redis=Mock()
    redis.get.side_effect=[ConnectionError('startup'),None,None]
    redis.xread.side_effect=Finished()
    monkeypatch.setattr('rgb.enrichment.time.sleep',lambda seconds:None)
    with pytest.raises(Finished):
        run(SimpleNamespace(redis=redis,settings=lambda:{'vlm_paused':False}),[])
    assert redis.get.call_count == 3
    redis.xread.assert_called_once()


def test_completed_job_replay_advances_cursor_without_another_request(monkeypatch):
    redis=Mock()
    redis.get.return_value=None
    redis.xread.side_effect=[[('rgb:events',[('1-0',{'event':json.dumps({'event_id':'scene'})})])],Finished()]
    store=SimpleNamespace(redis=redis,settings=lambda:{'vlm_paused':False},get=lambda key:{'analysis':{'schema_version':2,'status':'complete'}})
    analyze=Mock()
    monkeypatch.setattr('rgb.enrichment.analyze',analyze)
    with pytest.raises(Finished):run(store,[])
    analyze.assert_not_called()
    redis.set.assert_called_once_with('rgb:vlm:cursor','1-0')


def test_processing_failure_is_recorded_and_cursor_advances(monkeypatch):
    redis=Mock()
    redis.get.return_value=None
    redis.xread.side_effect=[[('rgb:events',[('1-0',{'event':json.dumps({'event_id':'scene'})})])],Finished()]
    event={'analysis':{'status':'queued','schema_version':2}}
    store=SimpleNamespace(redis=redis,settings=lambda:{'vlm_paused':False},get=lambda key:event,processing_failure=Mock())
    monkeypatch.setattr('rgb.enrichment.analyze',Mock(side_effect=OSError('image missing')))
    with pytest.raises(Finished):run(store,[])
    store.processing_failure.assert_called_once()
    assert json.loads(redis.hset.call_args.args[2])['status']=='failed'
    redis.set.assert_called_once_with('rgb:vlm:cursor','1-0')
