const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

class Element {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.style={};this.value='';this.open=false;this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(name,value){this[name]=value;}
  querySelectorAll(selector){const result=[];const matches=e=>selector==='option'?e.tag==='option':selector==='form'?e.tag==='form':selector==='details[data-evaluation]'?e.tag==='details'&&e.dataset.evaluation:selector.startsWith('.')?e.className===selector.slice(1):false;function walk(e){for(const child of e.children){if(matches(child))result.push(child);walk(child);}}walk(this);return result;}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
}
function setup(){
  const nodes=new Map();
  const context=vm.createContext({LiveVideo:class { constructor(image,report){this.report=report;} start(url){this.url=url;this.report('Connecting live frame…');} stop(){this.url=null;} },document:{getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==='[data-channel]'?[...nodes.values()].filter(n=>n.dataset.channel):[]},fetch:()=>new Promise(()=>{}),window:{addEventListener(){}},location:{hash:''},navigator:{},setInterval:()=>0,setTimeout:()=>0,clearInterval(){},clearTimeout(){}});
  vm.runInContext(fs.readFileSync('rgb/static/app.js','utf8'),context);
  return {context,node:id=>context.document.getElementById(id)};
}
const json=body=>({ok:true,json:async()=>body});
const preventDefault=()=>{};

const providerState=()=>({enabled:true,paused:false,selected_provider:'bedrock',selected_model:'us.amazon.nova-2-lite-v1:0',
  policy_revision:1,providers:[
    {id:'bedrock',label:'Amazon Bedrock',models:[{id:'us.amazon.nova-2-lite-v1:0',label:'Nova 2 Lite',selection_reason:'restore_primary'}]},
    {id:'openrouter',label:'OpenRouter',models:[{id:'amazon/nova-2-lite-v1',label:'Nova 2 Lite',selection_reason:'restore_primary'},
      {id:'google/gemini-2.5-flash-lite',label:'Gemini 2.5 Flash-Lite',selection_reason:'model_comparison'}]},
  ]});

test('provider selection sends the provider and model together and survives polling',async()=>{
  const {context,node}=setup();context.renderVlm(providerState());
  assert.equal(node('vlm-provider').value,'bedrock');
  node('vlm-provider').value='openrouter';node('vlm-provider').onchange();
  assert.equal(node('vlm-model-choice').value,'amazon/nova-2-lite-v1');
  node('vlm-model-choice').value='google/gemini-2.5-flash-lite';node('vlm-model-choice').onchange();
  context.renderVlm(providerState());
  assert.equal(node('vlm-provider').value,'openrouter');
  assert.equal(node('vlm-model-choice').value,'google/gemini-2.5-flash-lite');
  let saved;
  context.fetch=async(url,options)=>{
    if(options){saved=JSON.parse(options.body);return json(saved);}
    return json({...providerState(),selected_provider:'openrouter',selected_model:'google/gemini-2.5-flash-lite'});
  };
  await node('select-vlm-model').onclick();
  assert.deepEqual(saved,{selected_provider:'openrouter',selected_model:'google/gemini-2.5-flash-lite',model_selection_reason:'model_comparison'});
  assert.equal(node('vlm-model-choice').value,'google/gemini-2.5-flash-lite');
});

test('failed provider save retains the pending selection',async()=>{
  const {context,node}=setup();context.renderVlm(providerState());
  node('vlm-provider').value='openrouter';node('vlm-provider').onchange();
  context.fetch=async()=>({ok:false,status:400});
  await node('select-vlm-model').onclick();context.renderVlm(providerState());
  assert.equal(node('vlm-provider').value,'openrouter');
  assert.equal(node('vlm-model-choice').value,'amazon/nova-2-lite-v1');
});

test('status polled during a provider save cannot undo its acknowledgement',async()=>{
  const {context,node}=setup();context.renderVlm(providerState());
  node('vlm-provider').value='openrouter';node('vlm-provider').onchange();
  let saveResolve,statusResolve;
  context.fetch=async(url,options)=>new Promise(resolve=>{if(options)saveResolve=resolve;else statusResolve=resolve;});
  const saving=node('select-vlm-model').onclick();const stale=context.vlmStatus();
  assert.equal(node('vlm-provider').disabled,true);
  context.fetch=async()=>json({...providerState(),selected_provider:'openrouter',selected_model:'amazon/nova-2-lite-v1'});
  saveResolve(json({}));await saving;
  statusResolve(json(providerState()));await stale;
  assert.equal(node('vlm-provider').value,'openrouter');
  assert.equal(node('vlm-provider').value,'openrouter');
});

test('provider measurements are never pooled across transports',()=>{
  const {context,node}=setup();
  const workload={crop_count:1};
  context.renderVlm({...providerState(),latencies:[
    {provider:'bedrock',model:'us.amazon.nova-2-lite-v1:0',workload,p50_ms:1000,p95_ms:2000,samples:10},
    {provider:'openrouter',model:'us.amazon.nova-2-lite-v1:0',workload,p50_ms:9000,p95_ms:10000,samples:10},
  ]});
  assert.match(node('provider-latency').textContent,/p50 1.00 s/);
});

test('camera mode icons follow capabilities and active view without showing unavailable modes',()=>{
  const {context,node}=setup();
  for(const channel of ['color','depth','ir','pointcloud']){
    const option=new Element('option');option.value=channel;node('camera-channel').append(option);
    node('mode-'+channel).dataset.channel=channel;
  }
  node('camera-channel').value='color';
  context.renderCamera({backend:'v4l2',name:'Webcam',views:['color']});
  assert.equal(node('view-switch').hidden,true);
  assert.equal(node('mode-depth').hidden,true);
  assert.equal(node('camera-channel').children[1].disabled,true);
  context.renderCamera({backend:'orbbec',views:['color','depth','ir','pointcloud']});
  assert.equal(node('view-switch').hidden,false);
  assert.equal(node('mode-ir').hidden,false);
  assert.equal(node('camera-channel').children[1].disabled,false);
  node('camera-channel').value='depth';context.syncViewIcons();
  assert.equal(node('mode-depth')['aria-pressed'],'true');
  assert.equal(node('mode-color')['aria-pressed'],'false');
});

test('evaluation sends the displayed context, independent of global mode',async()=>{
  const {context}=setup();vm.runInContext("settings={display_mode:'materials'}",context);
  const revision={candidate_revision:'a'.repeat(64),mode:'objects',settings_version:3};
  const details=context.evaluation({detection_id:'box',review_context:revision});
  const form=details.querySelector('form');form.children[0].value='monitor';form.oninput();
  let request;context.fetch=async(url,options)=>{request=JSON.parse(options.body);return json({});};
  await form.onsubmit({preventDefault});
  assert.deepEqual(request.review_context,revision);assert.equal(request.mode,undefined);
  assert.equal(form.dataset.dirty,'false');
});

test('polling holds open or dirty evaluation forms and preserves failed drafts',async()=>{
  const {context,node}=setup();
  const details=context.evaluation({detection_id:'box',review_context:{}});details.open=true;
  const form=details.querySelector('form');form.children[0].value='unsaved monitor';
  node('detail').append(details);
  const event={analysis:{status:'complete'},detections:[],opinions:[],detector:{},settings:{},event_id:'scene'};
  context.fetch=async()=>json(event);
  await context.show('scene',true);
  assert.equal(node('detail').children[0],details);assert.equal(form.children[0].value,'unsaved monitor');
  details.open=false;form.onchange();await context.show('scene',true);
  assert.equal(node('detail').children[0],details);
  context.fetch=async()=>({ok:false,status:409,json:async()=>({error:'Reading changed'})});
  await form.onsubmit({preventDefault});
  assert.equal(form.children.at(-1).textContent,'Reading changed');assert.equal(form.dataset.dirty,'true');
  context.fetch=async()=>json(event);form.dataset.dirty='false';await context.show('scene',true);
  assert.notEqual(node('detail').children[0],details);
});

test('geolocation cannot assign a different camera after source selection changes',async()=>{
  const {context,node}=setup();let callbacks;let requests=0;
  context.navigator.geolocation={getCurrentPosition:(...args)=>{callbacks=args;}};
  node('source').value='A';node('location-label').value='Desk A';node('browser-location').onclick();
  assert.equal(callbacks[2].enableHighAccuracy,false);assert.equal(callbacks[2].timeout,20000);
  node('source').value='B';node('location-label').value='Desk B';context.fetch=async()=>{requests++;return json({});};
  await callbacks[0]({coords:{latitude:1,longitude:2,accuracy:3}});
  assert.equal(requests,0);assert.equal(node('location-label').value,'Desk B');assert.equal(node('location-lat').value,'');
});

test('received coordinates remain populated when saving fails; capture label is stable',async()=>{
  const {context,node}=setup();let callbacks,request;
  context.navigator.geolocation={getCurrentPosition:(...args)=>{callbacks=args;}};
  node('source').value='A';node('location-label').value='Desk A';node('browser-location').onclick();
  node('location-label').value='Edited while waiting';
  context.fetch=async(url,options)=>{request={url,body:JSON.parse(options.body)};return {ok:false,status:503};};
  await callbacks[0]({coords:{latitude:1.25,longitude:2.5,accuracy:3}});
  assert.equal(node('location-lat').value,1.25);assert.equal(node('location-lon').value,2.5);
  assert.equal(request.url,'/api/sources/A/location');assert.equal(request.body.label,'Desk A');
  assert.equal(node('location-status').textContent,'Location save failed (503)');
  callbacks[1]({code:3,message:'Timeout expired'});
  assert.match(node('location-status').textContent,/No coordinates received/);
});

test('late location response does not populate another camera',async()=>{
  const {context,node}=setup();let resolve;
  node('source').value='A';context.fetch=()=>new Promise(r=>{resolve=r;});
  const loading=context.loadLocation();node('source').value='B';node('location-label').value='Desk B';
  resolve(json({label:'Desk A',latitude:1,longitude:2}));await loading;
  assert.equal(node('location-label').value,'Desk B');
});

test('pause control saves durable intent and distinguishes finishing from idle',async()=>{
  const {context,node}=setup();context.renderVlm({enabled:true,paused:false,active:{}});
  let saved;
  context.fetch=async(url,options)=>{if(options){saved=JSON.parse(options.body);return json({vlm_paused:true});}return json({enabled:true,paused:true,active:{attempt_id:'running'}});};
  await node('pause-vlm').onclick();assert.deepEqual(saved,{vlm_paused:true});
  assert.equal(node('pause-vlm').textContent,'Resume VLM');assert.match(node('vlm-state').textContent,/finishing/);
  context.renderVlm({enabled:true,paused:true,active:null});assert.equal(node('vlm-state').textContent,'Paused');
});

test('a pre-pause status response cannot overwrite the confirmed paused state',async()=>{
  const {context,node}=setup();let resolve;context.fetch=()=>new Promise(r=>{resolve=r;});
  const old=context.vlmStatus();
  context.fetch=async(url,options)=>json(options?{vlm_paused:true}:{paused:true,enabled:true,active:null});
  await node('pause-vlm').onclick();resolve(json({paused:false,enabled:true,active:{}}));await old;
  assert.equal(node('pause-vlm').textContent,'Resume VLM');assert.equal(node('vlm-state').textContent,'Paused');
});

test('capture pause remains separate from VLM pause and keeps manual capture available',async()=>{
  const {context,node}=setup();
  context.renderCapture({capture:{paused:false,interval_s:60,next_in_s:20}});
  let saved;
  context.fetch=async(url,options)=>{
    if(options){saved=JSON.parse(options.body);return json({capture_paused:true,capture_interval_s:60});}
    return json({enabled:true,paused:false,capture:{paused:true,interval_s:60,next_in_s:null}});
  };
  await node('pause-capture').onclick();
  assert.deepEqual(saved,{capture_paused:true});
  assert.equal(node('pause-capture').textContent,'Resume capture');
  assert.equal(node('capture-state').textContent,'Paused');
  assert.equal(node('pause-vlm').textContent,'Pause VLM');
  assert.notEqual(node('capture-observation').disabled,true);
});

test('a stale status cannot undo capture pause; edited intervals survive polling',async()=>{
  const {context,node}=setup();let resolve;
  context.fetch=()=>new Promise(r=>{resolve=r;});
  const stale=context.vlmStatus();
  context.fetch=async(url,options)=>json(options?{capture_paused:true,capture_interval_s:60}:
    {enabled:true,paused:false,capture:{paused:true,interval_s:60}});
  await node('pause-capture').onclick();
  resolve(json({enabled:true,paused:false,capture:{paused:false,interval_s:60}}));await stale;
  assert.equal(node('pause-capture').textContent,'Resume capture');
  node('capture-interval').value='300';node('capture-interval').oninput();
  context.renderCapture({capture:{paused:true,interval_s:60}});
  assert.equal(node('capture-interval').value,'300');
});

test('capture interval input validation avoids an invalid settings request',async()=>{
  const {context,node}=setup();let calls=0;
  context.fetch=async()=>{calls++;return json({});};
  node('capture-interval').value='0';await node('apply-capture-interval').onclick();
  assert.equal(calls,0);assert.match(node('capture-state').textContent,/1 to 86400/);
});

test('local classification preview sends no-recording policy independently of VLM',async()=>{
  const {context,node}=setup();
  node('classification-mode').value='preview';
  let request;
  context.fetch=async(url,options)=>{request={url,body:JSON.parse(options.body)};return json({classification_running:true,record_observations:false});};
  await context.setClassification(true);
  assert.equal(request.url,'/api/settings');
  assert.deepEqual(request.body,{classification_running:true,record_observations:false});
  assert.equal(node('classification-state').textContent,'Control saved · awaiting Uno Q');
  context.acknowledgeClassification({classification_running:true,record_observations:false,applied_settings_version:0});
  assert.equal(node('classification-state').textContent,'Running · preview only');
  assert.equal(request.body.vlm_paused,undefined);
});

test('catch up keeps one snapshot and source across pages',async()=>{
  const {context,node}=setup();
  node('source').value='unoq';
  const bodies=[];
  context.fetch=async(url,options)=>{bodies.push(JSON.parse(options.body));return json({batch_id:'batch',queued:bodies.length,done:bodies.length===2});};
  await node('catch-up').onclick();
  assert.deepEqual(bodies,[{source_id:'unoq'},{source_id:'unoq',batch_id:'batch'}]);
  assert.equal(node('catch-up').disabled,false);
});

test('healthy camera status cannot hide a stalled image connection',async()=>{
  const {context,node}=setup();
  vm.runInContext("sources=[{id:'unoq'}]",context);node('source').value='unoq';
  context.live();assert.equal(node('live-state').textContent,'Connecting live frame…');
  node('live-state').textContent='Stale · showing last frame · reconnecting';
  context.fetch=async()=>json({inference_ms:320});await context.status();
  assert.equal(node('live-state').textContent,'Stale · showing last frame · reconnecting');
  assert.equal(vm.runInContext('liveVideo.url',context),'/api/live/unoq/stream');
});

test('local depth and IR use recoverable streams; switching to 3D releases the stream',()=>{
  const {context,node}=setup();
  vm.runInContext("sources=[{id:'femto-bolt',local:true}]",context);
  node('source').value='femto-bolt';node('camera-channel').value='depth';
  context.live();
  assert.equal(vm.runInContext('liveVideo.url',context),'/api/camera/depth/stream');
  node('camera-channel').value='pointcloud';context.live();
  assert.equal(vm.runInContext('liveVideo.url',context),null);
  assert.equal(node('live').hidden,true);
  node('camera-channel').value='ir';context.live();
  assert.equal(vm.runInContext('liveVideo.url',context),'/api/camera/ir/stream');
  assert.equal(node('live').hidden,false);
  assert.equal(node('pointcloud-view').hidden,true);
});

test('local material readings are shown as similarity scores, independent of VLM',async()=>{
  const {context,node}=setup();
  const local={status:'uncertain',label:null,best_candidate:'plastic',similarity:.321,margin:.003};
  const event={analysis:{status:'not_analyzed'},frame_id:'local-frame',session_id:'session',
    detections:[{detection_id:'box',box:[0,0,1,1],display_label:'Material uncertain',label:'bottle',
      label_provider:'siglip2',display_confidence:null,local_material:local,review_context:{}},
      {detection_id:'pending',box:[0,0,1,1],display_label:'Materials pending',label:'box',
        label_provider:'vlm',display_confidence:null,review_context:{}}],
    opinions:[],detector:{model:'yoloe-26s'},settings:{display_mode:'materials'},event_id:'scene'};
  context.fetch=async()=>json(event);
  await context.show('scene');
  const card=node('detail').children.find(c=>c.className==='reading');
  assert.equal(card.children[1].textContent,'SigLIP2');
  assert.match(card.children[2].textContent,/uncertain.*similarity 0.321.*uncalibrated/);
  assert.doesNotMatch(card.children[2].textContent,/%/);
  assert.equal(node('detail').children.filter(c=>c.className==='reading')[1].children[1].textContent,'VLM');
  context.renderVlm({enabled:false,paused:false,models:[],local_materials:{state:'running',inference_ms:25}});
  assert.match(node('local-material-state').textContent,/running.*25 ms per crop/);
});

test('empty local archive explains capture pause and preserves archive errors',async()=>{
  const {context,node}=setup();
  context.fetch=async()=>json([]);
  context.renderCapture({capture:{paused:true,interval_s:60}});
  await context.load();
  assert.match(node('archive-status').textContent,/Capture paused.*Capture once/);
  context.renderCapture({capture:{paused:false,interval_s:60}});
  assert.equal(node('archive-status').textContent,'Waiting for the first capture.');
  context.fetch=async()=>({ok:false,status:503});
  await context.load();
  context.renderCapture({capture:{paused:true,interval_s:60}});
  assert.match(node('archive-status').textContent,/Archive unavailable.*503/);
});

test('individual VLM request refreshes a terminal detail and restarts polling',async()=>{
  const {context,node}=setup();let gets=0,timers=0;
  context.setTimeout=()=>{timers++;return 1;};
  const event={analysis:{status:'complete'},detections:[{detection_id:'box',box:[0,0,1,1],display_label:'pen',label:'pen',label_provider:'detector',review_context:{}}],opinions:[],detector:{model:'yoloe'},settings:{display_mode:'objects'},event_id:'scene'};
  context.fetch=async(url,options)=>{if(options){event.analysis.status='queued';return json({});}gets++;return json(event);};
  await context.show('scene');
  const card=node('detail').children.find(c=>c.className==='reading');
  await card.children.find(c=>c.tag==='button').onclick();
  assert.equal(gets,2);assert.equal(timers,1);
  assert.match(node('detail').querySelector('.analysis-state').textContent,/queued/);
});

test('activity lights distinguish detection, capture, and an active paused VLM',()=>{
  const {context,node}=setup();
  context.renderDetection({state:'running',detector:{state:'running'}});
  assert.equal(node('detection-indicator').textContent,'On');
  context.renderCapture({capture:{paused:true,interval_s:60}});
  assert.equal(node('capture-indicator').textContent,'Off');
  context.renderVlm({...providerState(),paused:true,active:{job_id:'running'},manual_queue:3,live_pending:'frame'});
  assert.equal(node('vlm-indicator').textContent,'Active');
  assert.equal(node('vlm-backlog').textContent,'Queue: 4 · 3 manual · 1 latest frame');
  context.renderVlm({...providerState(),paused:true,active:null,manual_queue:0});
  assert.equal(node('vlm-indicator').textContent,'Off');
  assert.equal(node('vlm-backlog').textContent,'Queue: 0 · 0 manual');
  context.renderDetection({state:'stale',detector:{state:'running'}});
  assert.equal(node('detection-indicator').dataset.state,'unknown');
});

test('board capture light waits for acknowledgement and separates preview from recording',()=>{
  const {context,node}=setup();
  context.renderClassification({version:7,classification_running:true,record_observations:true});
  assert.equal(node('capture-indicator').textContent,'Pending');
  context.acknowledgeClassification({applied_settings_version:6,classification_running:true,record_observations:true});
  assert.equal(node('capture-indicator').textContent,'Pending');
  context.acknowledgeClassification({applied_settings_version:7,classification_running:true,record_observations:true});
  assert.equal(node('capture-indicator').textContent,'Recording');
  context.renderClassification({version:8,classification_running:true,record_observations:false});
  context.acknowledgeClassification({applied_settings_version:8,classification_running:true,record_observations:false});
  assert.equal(node('capture-indicator').textContent,'Off');
});

test('provider Apply feedback survives polling and edits require another Apply',async()=>{
  const {context,node}=setup();context.renderVlm(providerState());
  node('vlm-provider').value='openrouter';node('vlm-provider').onchange();
  assert.equal(node('provider-status').textContent,'Not applied');
  context.fetch=async(url,options)=>json(options?{}:{...providerState(),selected_provider:'openrouter',selected_model:'amazon/nova-2-lite-v1'});
  await node('select-vlm-model').onclick();
  assert.equal(node('provider-status').textContent,'Applied · OpenRouter · Nova 2 Lite');
  context.renderVlm({...providerState(),selected_provider:'openrouter',selected_model:'amazon/nova-2-lite-v1'});
  assert.equal(node('provider-status').textContent,'Applied · OpenRouter · Nova 2 Lite');
});
