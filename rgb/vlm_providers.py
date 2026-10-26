"""Provider transports for the local runner's shared visual review workload."""
import base64
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import requests

PRIMARY_MODEL='us.amazon.nova-2-lite-v1:0'
FALLBACK_MODEL='us.amazon.nova-lite-v1:0'
MODELS={
    'bedrock':{
        PRIMARY_MODEL:('Nova 2 Lite','restore_primary'),
        FALLBACK_MODEL:('Nova Lite','latency_fallback'),
    },
    'openrouter':{
        'amazon/nova-2-lite-v1':('Nova 2 Lite','restore_primary'),
        'amazon/nova-lite-v1':('Nova Lite','latency_fallback'),
        'google/gemini-2.5-flash-lite':('Gemini 2.5 Flash-Lite','model_comparison'),
    },
}
PROVIDER_LABELS={'bedrock':'Amazon Bedrock','openrouter':'OpenRouter'}
OPENROUTER_URL='https://openrouter.ai/api/v1/chat/completions'
IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def model_selection(provider,model):
    if not isinstance(provider,str) or not isinstance(model,str) or model not in MODELS.get(provider,{}):
        raise ValueError('Unsupported provider/model combination')
    return MODELS[provider][model]


def provider_catalog():
    return [{'id':provider,'label':PROVIDER_LABELS[provider],
             'models':[{'id':model,'label':label,'selection_reason':reason}
                       for model,(label,reason) in models.items()]} for provider,models in MODELS.items()]


@dataclass(frozen=True)
class ReviewRequest:
    model:str
    system:str
    prompt:str
    images:tuple  # Ordered (label, JPEG bytes) pairs, shared across providers.
    max_tokens:int


@dataclass(frozen=True)
class ReviewResponse:
    text:str
    audit:dict=field(default_factory=dict)


class ReviewProvider(Protocol):
    def review(self,request:ReviewRequest)->ReviewResponse: ...


class ProviderError(Exception):
    def __init__(self,message,audit=None):
        super().__init__(message)
        self.audit=audit or {}


class BedrockProvider:
    def __init__(self,profile='default',region='us-west-2',connect_timeout=5,read_timeout=15,client=None):
        self.profile=profile;self.region=region
        self.connect_timeout=connect_timeout;self.read_timeout=read_timeout
        self.client=client

    def _client(self):
        if self.client is None:
            import boto3
            from botocore.config import Config
            self.client=boto3.Session(profile_name=self.profile,region_name=self.region).client('bedrock-runtime',
                config=Config(connect_timeout=self.connect_timeout,read_timeout=self.read_timeout,
                              retries={'total_max_attempts':1},max_pool_connections=1))
        return self.client

    def review(self,request):
        model_selection('bedrock',request.model)
        content=[{'text':request.prompt}]
        for label,data in request.images:
            content.extend([{'text':label},{'image':{'format':'jpeg','source':{'bytes':data}}}])
        body={'modelId':request.model,'messages':[{'role':'user','content':content}],
              'system':[{'text':request.system}],
              'inferenceConfig':{'maxTokens':request.max_tokens,'temperature':0}}
        if request.model==PRIMARY_MODEL:
            body['additionalModelRequestFields']={'reasoningConfig':{'type':'disabled'}}
        response=self._client().converse(**body)
        audit={'raw_response':response,'http_status':response['ResponseMetadata']['HTTPStatusCode'],
               'request_id':response['ResponseMetadata'].get('RequestId'),'finish_reason':response['stopReason'],
               'prompt_tokens':response.get('usage',{}).get('inputTokens'),
               'output_tokens':response.get('usage',{}).get('outputTokens'),
               'bedrock_latency_ms':response.get('metrics',{}).get('latencyMs')}
        if response['stopReason'] not in {'end_turn','stop_sequence'}:
            raise ProviderError('Non-success stop reason: '+response['stopReason'],audit)
        blocks=response['output']['message']['content']
        if any('text' not in block for block in blocks):raise ProviderError('Expected text-only model response',audit)
        return ReviewResponse(''.join(block['text'] for block in blocks),audit)


class OpenRouterProvider:
    def __init__(self,connect_timeout=5,read_timeout=15,session=None):
        self.connect_timeout=connect_timeout;self.read_timeout=read_timeout
        self.session=session or requests.Session()

    def review(self,request):
        model_selection('openrouter',request.model)
        key=os.environ.get('OPENROUTER_API_KEY','').strip()
        if not key:raise ProviderError('OPENROUTER_API_KEY is required for OpenRouter')
        content=[{'type':'text','text':request.prompt}]
        for label,data in request.images:
            content.extend([{'type':'text','text':label},
                            {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}}])
        body={'model':request.model,'messages':[{'role':'system','content':request.system},{'role':'user','content':content}],
              'temperature':0,'max_tokens':request.max_tokens,'stream':False,
              'provider':{'sort':'latency','require_parameters':True,'allow_fallbacks':False}}
        if request.model!='amazon/nova-lite-v1':body['reasoning']={'enabled':False}
        # No transport retries, redirects, or automatic model/provider fallback.
        try:
            response=self.session.post(OPENROUTER_URL,json=body,
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-OpenRouter-Title':'Bolt RGB'},
                timeout=(self.connect_timeout,self.read_timeout),allow_redirects=False)
        except requests.RequestException as exc:
            raise ProviderError('OpenRouter transport failed: '+type(exc).__name__) from exc
        audit={'http_status':response.status_code,'request_id':response.headers.get('x-request-id')}
        try:
            try:raw=response.json()
            except ValueError as exc:
                audit['raw_response']=response.text
                raise ProviderError('OpenRouter returned invalid JSON',audit) from exc
            audit['raw_response']=raw
            if response.status_code!=200 or not isinstance(raw,dict) or raw.get('error'):
                raise ProviderError('OpenRouter request failed (HTTP '+str(response.status_code)+')',audit)
            audit.update(request_id=raw.get('id') or audit['request_id'],resolved_model=raw.get('model'),
                         upstream_provider=raw.get('provider'))
            try:
                usage=raw.get('usage') or {}
                audit.update(prompt_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'),
                             cost=usage.get('cost'),reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
                choices=raw['choices']
                if len(choices)!=1:raise ValueError()
                choice=choices[0];message=choice['message']
                audit['finish_reason']=choice.get('finish_reason')
                if (choice.get('finish_reason')!='stop' or message.get('refusal') or message.get('tool_calls')
                        or not isinstance(message.get('content'),str) or not message['content'].strip()):
                    raise ValueError()
            except (KeyError,TypeError,ValueError,AttributeError) as exc:
                raise ProviderError('OpenRouter returned an incomplete or non-text completion',audit) from exc
            return ReviewResponse(message['content'],audit)
        finally:response.close()
