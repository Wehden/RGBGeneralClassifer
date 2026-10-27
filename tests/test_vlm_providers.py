import base64
import json

import pytest
import requests
from requests.adapters import HTTPAdapter

from rgb.vlm_providers import (
    BedrockProvider, OpenRouterProvider, ProviderError, ReviewRequest, PRIMARY_MODEL,
)


class RecordedHTTP(HTTPAdapter):
    """Exercise requests serialization without sending images or credentials."""
    def __init__(self):
        super().__init__()
        self.calls=[]
        self.status=200
        self.payload={'id':'gen-test','model':'amazon/nova-2-lite-v1','provider':'Amazon Bedrock',
                      'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'{"observations":[]}'}}],
                      'usage':{'prompt_tokens':123,'completion_tokens':20,'cost':.0001,
                               'completion_tokens_details':{'reasoning_tokens':0}}}
        self.error=None
        self.raw_body=None

    def send(self,request,**kwargs):
        self.calls.append((request,kwargs))
        if self.error:raise self.error
        response=requests.Response()
        response.status_code=self.status
        response.headers['x-request-id']='http-test'
        response.request=request
        response._content=self.raw_body if self.raw_body is not None else json.dumps(self.payload).encode()
        return response


@pytest.fixture
def transport(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','test-private-key')
    session=requests.Session();session.trust_env=False
    adapter=RecordedHTTP();session.mount('https://openrouter.ai/',adapter)
    yield OpenRouterProvider(connect_timeout=3,read_timeout=12,session=session),adapter
    session.close()


def review_request(model='amazon/nova-2-lite-v1'):
    return ReviewRequest(model,'system contract','review these crops',
                         (('Image 1: color_scene',b'\xff\xd8scene'),('Image 2: color_crop, box_index 0',b'\xff\xd8crop')),1600)


def test_openrouter_serializes_images_reasoning_routing_and_metadata(transport):
    provider,adapter=transport
    result=provider.review(review_request())
    request,options=adapter.calls[0]
    assert request.url=='https://openrouter.ai/api/v1/chat/completions'
    assert request.headers['Authorization']=='Bearer test-private-key'
    body=json.loads(request.body)
    assert body['reasoning']=={'enabled':False}
    assert body['provider']=={'sort':'latency','require_parameters':True,'allow_fallbacks':False}
    assert body['model']=='amazon/nova-2-lite-v1' and body['temperature']==0 and body['max_tokens']==1600
    assert body['stream'] is False and 'models' not in body
    assert body['messages'][0]=={'role':'system','content':'system contract'}
    content=body['messages'][1]['content']
    assert content[0]['text']=='review these crops'
    assert content[1]['text']=='Image 1: color_scene'
    assert content[3]['text']=='Image 2: color_crop, box_index 0'
    assert [base64.b64decode(content[i]['image_url']['url'].split(',')[1]) for i in [2,4]]==[b'\xff\xd8scene',b'\xff\xd8crop']
    assert options['timeout']==(3,12) and adapter.max_retries.total==0
    assert result.text=='{"observations":[]}'
    assert result.audit['request_id']=='gen-test' and result.audit['http_status']==200
    assert result.audit['resolved_model']=='amazon/nova-2-lite-v1'
    assert result.audit['upstream_provider']=='Amazon Bedrock'
    assert result.audit['cost']==.0001 and result.audit['reasoning_tokens']==0
    assert 'test-private-key' not in json.dumps(result.audit)


@pytest.mark.parametrize('model,reasoning',[
    ('amazon/nova-lite-v1',None),('google/gemini-2.5-flash-lite',{'enabled':False}),
])
def test_openrouter_model_specific_reasoning(transport,model,reasoning):
    provider,adapter=transport
    provider.review(review_request(model))
    assert json.loads(adapter.calls[0][0].body).get('reasoning')==reasoning


@pytest.mark.parametrize('status',[400,401,429,500,503])
def test_openrouter_http_failure_retains_evidence_without_retry(transport,status):
    provider,adapter=transport;adapter.status=status
    adapter.payload={'error':{'message':'upstream unavailable','code':status}}
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert caught.value.audit['http_status']==status
    assert caught.value.audit['raw_response']==adapter.payload
    assert len(adapter.calls)==1


def test_openrouter_redirect_does_not_forward_credentials(transport):
    provider,adapter=transport;adapter.status=302
    with pytest.raises(ProviderError):provider.review(review_request())
    assert len(adapter.calls)==1


@pytest.mark.parametrize('change',[
    {'finish_reason':'length'}, {'finish_reason':'content_filter'},
    {'message':{'content':'valid-looking text','refusal':'denied'}},
    {'message':{'content':'valid-looking text','tool_calls':[{'id':'tool'}]}},
    {'message':{'content':None}}, {'message':{'content':[{'text':'not a text completion'}]}},
])
def test_openrouter_rejects_incomplete_refused_or_tool_responses(transport,change):
    provider,adapter=transport
    adapter.payload['choices'][0].update(change)
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert caught.value.audit['raw_response']==adapter.payload
    assert caught.value.audit['output_tokens']==20


def test_openrouter_reports_in_band_error(transport):
    provider,adapter=transport
    adapter.payload={'error':{'message':'provider failed'}}
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert caught.value.audit['http_status']==200


def test_openrouter_invalid_json_retains_raw_response(transport):
    provider,adapter=transport;adapter.raw_body=b'<html>upstream unavailable</html>'
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert caught.value.audit['http_status']==200
    assert caught.value.audit['raw_response']=='<html>upstream unavailable</html>'


@pytest.mark.parametrize('payload',[[],None,{'choices':[]},{'choices':None},{'choices':[{}]}])
def test_openrouter_malformed_envelope_is_a_recorded_failure(transport,payload):
    provider,adapter=transport;adapter.payload=payload
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert caught.value.audit['raw_response']==payload


def test_missing_key_and_unsupported_model_never_dispatch(transport,monkeypatch):
    provider,adapter=transport
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(ProviderError,match='OPENROUTER_API_KEY'):provider.review(review_request())
    with pytest.raises(ValueError):provider.review(review_request(PRIMARY_MODEL))
    assert not adapter.calls


def test_transport_error_does_not_retry_or_echo_request_details(transport):
    provider,adapter=transport
    adapter.error=requests.ReadTimeout('request detail test-private-key')
    with pytest.raises(ProviderError) as caught:provider.review(review_request())
    assert len(adapter.calls)==1 and 'test-private-key' not in str(caught.value)


def test_bedrock_maps_the_same_review_request():
    class Client:
        def converse(self,**body):
            self.body=body
            return {'ResponseMetadata':{'HTTPStatusCode':200,'RequestId':'aws-test'},'stopReason':'end_turn',
                    'output':{'message':{'content':[{'text':'{"observations":[]}'}]}},
                    'usage':{'inputTokens':123,'outputTokens':20},'metrics':{'latencyMs':100}}
    client=Client();provider=BedrockProvider(client=client)
    result=provider.review(review_request(PRIMARY_MODEL))
    body=client.body
    assert body['additionalModelRequestFields']=={'reasoningConfig':{'type':'disabled'}}
    assert body['inferenceConfig']=={'maxTokens':1600,'temperature':0}
    content=body['messages'][0]['content']
    assert content[0]['text']=='review these crops'
    assert content[1]['text']=='Image 1: color_scene'
    assert [content[i]['image']['source']['bytes'] for i in [2,4]]==[b'\xff\xd8scene',b'\xff\xd8crop']
    assert result.audit['bedrock_latency_ms']==100 and result.audit['request_id']=='aws-test'
