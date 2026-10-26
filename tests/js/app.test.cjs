const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

class Element {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.style={};this.value='';this.open=false;this.textContent='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(name,value){this[name]=value;}
  querySelectorAll(selector){const result=[];const matches=e=>selector==='form'?e.tag==='form':selector==='details[data-evaluation]'?e.tag==='details'&&e.dataset.evaluation:selector.startsWith('.')?e.className===selector.slice(1):false;function walk(e){for(const child of e.children){if(matches(child))result.push(child);walk(child);}}walk(this);return result;}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
}
function setup(){
  const nodes=new Map();
  const context=vm.createContext({document:{getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},createElement:tag=>new Element(tag)},fetch:()=>new Promise(()=>{}),window:{addEventListener(){}},location:{hash:''},navigator:{},setInterval:()=>0,setTimeout:()=>0,clearInterval(){},clearTimeout(){}});
  vm.runInContext(fs.readFileSync('rgb/static/app.js','utf8'),context);
  return {context,node:id=>context.document.getElementById(id)};
}
const json=body=>({ok:true,json:async()=>body});
const preventDefault=()=>{};

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
  context.renderVlm({enabled:true,paused:true,active:null});assert.equal(node('vlm-state').textContent,'Paused · VLM idle');
});

test('a pre-pause status response cannot overwrite the confirmed paused state',async()=>{
  const {context,node}=setup();let resolve;context.fetch=()=>new Promise(r=>{resolve=r;});
  const old=context.vlmStatus();
  context.fetch=async(url,options)=>json(options?{vlm_paused:true}:{paused:true,enabled:true,active:null});
  await node('pause-vlm').onclick();resolve(json({paused:false,enabled:true,active:{}}));await old;
  assert.equal(node('pause-vlm').textContent,'Resume VLM');assert.equal(node('vlm-state').textContent,'Paused · VLM idle');
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

test('live reconnection changes the resource URL and healthy status does not hide a failed image',async()=>{
  const {context,node}=setup();
  vm.runInContext("sources=[{id:'unoq'}]",context);node('source').value='unoq';
  context.live();const first=node('live').src;
  context.live();assert.notEqual(node('live').src,first);
  context.fetch=async()=>json({inference_ms:320});
  node('live').naturalWidth=0;await context.status();
  assert.equal(node('live-state').textContent,'Connecting live frame…');
  node('live').complete=true;await context.status();
  assert.equal(node('live-state').textContent,'Camera unavailable · reconnecting');
  node('live').onload();assert.equal(node('live-state').textContent,'Camera unavailable · reconnecting');
  node('live').naturalWidth=640;await context.status();
  assert.equal(node('live-state').textContent,'LIVE');
});
