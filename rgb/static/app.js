const $ = id => document.getElementById(id);
let sources = [], cursor = null, selected = null, timer, detailTimer, settings;
const openedHistories = new Set();
function element(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
async function get(url) { const r = await fetch(url); if (!r.ok) throw new Error(`Request failed (${r.status})`); return r.json(); }
function date(value) { return new Date(value).toLocaleString(); }
async function status() {
  try { const s = await get(`/api/live/${encodeURIComponent($('source').value)}/status`); $('live-state').textContent = s.error ? s.error : 'LIVE'; $('timing').textContent = s.classification_running===false?'Classification paused':`${s.inference_ms ?? '?'} ms inference${s.detections?.length===0&&s.proposal_max!=null?' · highest proposal '+(s.proposal_max*100).toFixed(1)+'%':''}`; acknowledgeClassification(s); if(s.model)$('detector-model').textContent=`Detection: ${s.model} · Uno Q ${s.detector_config?.runtime ?? 'CPU'}`; }
  catch { $('live-state').textContent = 'Camera unavailable · reconnecting'; $('timing').textContent = ''; }
}
function live() {
  const s = sources.find(x => x.id === $('source').value); if (!s) return;
  $('live').src = `/api/live/${encodeURIComponent(s.id)}/stream`;
  $('camera-location').textContent = s.location ? `${s.location} · ${s.location_method}` : 'Location unassigned';
  loadLocation(); status(); clearInterval(timer); timer = setInterval(status, 10000);
}
$('live').onerror = () => { $('live-state').textContent = 'Camera unavailable · reconnecting'; setTimeout(live, 5000); };
async function history(d, node) {
  try { const entries = await get(`/api/detections/${d.detection_id}/history`); node.replaceChildren();
    if (!entries.length) node.append(element('p',`Classifier: ${d.label} · ${(d.confidence*100).toFixed(1)}%`));
    for (const entry of entries) { const c = entry.candidate; let text; if (entry.kind === 'initial') text=`${date(entry.recorded_at)} · Classifier: ${entry.initial.label} ${(entry.initial.confidence*100).toFixed(1)}%`; else if (entry.kind === 'enhanced') text=`${date(entry.updated_at)} · ${c.model}: ${c.label} ${(c.confidence*100).toFixed(1)}% · ${c.comparison}`; else if (entry.kind === 'model_attempt') text=`${date(entry.started_at)} · ${entry.model} · ${entry.outcome} · ${entry.latency_ms == null ? '' : entry.latency_ms+' ms'} · ${entry.error_type ?? ''}`; else if (entry.kind === 'processing_failure') text=`${date(entry.recorded_at)} · Processing failed · ${entry.error_type}`; else text=`${date(entry.reviewed_at)} · Evaluation: ${entry.reference_label} · classifier ${entry.detector_result} · VLM ${entry.vlm_result}`; node.append(element('p',text)); }
  } catch (e) { node.textContent = e.message; }
}
function evaluation(d) {
  const details=element('details'); details.dataset.evaluation='true'; details.append(element('summary','Evaluate detection')); const form=element('form',undefined,'evaluation');
  const reference=element('input'); reference.placeholder='Reference label'; reference.required=true; reference.maxLength=120; reference.setAttribute('aria-label','Reference label');
  const notes=element('textarea'); notes.placeholder='Evaluation notes'; notes.maxLength=4000; notes.setAttribute('aria-label','Evaluation notes');
  function outcome(name) { const label=element('label',name); const select=element('select'); for (const value of ['unscored','correct','incorrect']) { const option=element('option',value); option.value=value; select.append(option); } label.append(select); return [label,select]; }
  const [detectorLabel,detector]=outcome('Classifier'); const [vlmLabel,vlm]=outcome('VLM'); const button=element('button','Save evaluation'); const discard=element('button','Discard draft and refresh'); discard.type='button'; const message=element('span'); form.append(reference,detectorLabel,vlmLabel,notes,button,discard,message);
  form.oninput=form.onchange=()=>{form.dataset.dirty='true';};
  discard.onclick=()=>{form.dataset.dirty='false';details.open=false;if(selected)show(selected,true);};
  form.onsubmit=async e => { e.preventDefault(); button.disabled=true; try { const r=await fetch(`/api/detections/${d.detection_id}/evaluate`,{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'evaluation'},body:JSON.stringify({reference_label:reference.value,detector_result:detector.value,vlm_result:vlm.value,notes:notes.value,review_context:d.review_context})}); if (!r.ok) {const body=await r.json();throw new Error(body.error||`Save failed (${r.status})`);} message.textContent='Evaluation recorded'; form.dataset.dirty='false'; } catch (e) { message.textContent=e.message; } button.disabled=false; };
  details.append(form); return details;
}
async function show(id, refresh=false) {
  selected = id; if (location.hash !== '#event/'+id) location.hash='event/'+id; clearTimeout(detailTimer); const target = $('detail'); if (!refresh) target.replaceChildren(element('p','Loading…'));
  try {
    const event = await get(`/api/events/${id}`); if (selected !== id) return;
    if (refresh && reviewIsEditing(target)) { const state=target.querySelector('.analysis-state'); if(state)state.textContent=`Analysis: ${event.analysis.status.replaceAll('_',' ')} · Review held while editing`;detailTimer=setTimeout(()=>show(id,true),3000);return; }
    target.replaceChildren(); const frame = element('div',undefined,'evidence-image'); const img = element('img'); img.src = `/api/events/${id}/image.jpg`; img.alt = 'Archived camera observation'; frame.append(img);
    event.detections.forEach((d,i) => { const b = element('span',undefined,'box'); b.style.top = `${d.box[0]*100}%`; b.style.left = `${d.box[1]*100}%`; b.style.height = `${(d.box[2]-d.box[0])*100}%`; b.style.width = `${(d.box[3]-d.box[1])*100}%`; b.append(element('span',`${i+1}. ${d.display_label}`)); frame.append(b); });
    target.append(frame,element('h3',date(event.observed_at)),element('p',`${event.source_id} · ${event.location ?? 'Location unknown'}`,'muted'));
    const state = event.analysis; const progress = state.total_boxes ? ` · ${state.completed_boxes}/${state.total_boxes} boxes` : '';
    target.append(element('p',`Analysis: ${state.status.replaceAll('_',' ')}${progress}`,'analysis-state'));
    event.detections.forEach((d,i) => {
      const card = element('section',undefined,'reading'); const c = d.enhancement;
      card.append(element('h3',`${i+1}. ${d.display_label}`),element('p',`${d.label_provider === 'detector' ? event.detector.model : c?.model ?? 'Qwen'}${d.display_confidence == null ? '' : ' · '+(d.display_confidence*100).toFixed(1)+'%'}`,'muted'));
      if (event.settings.display_mode === 'materials' && c?.materials) { for (const m of c.materials) { card.append(element('p',`${m.label} · ${(m.confidence*100).toFixed(1)}%`,'comparison'),element('p',m.reason)); } }
      if (c && event.settings.display_mode !== 'materials') { card.append(element('p',`${d.label} → ${c.label} · ${c.comparison}`,'comparison'),element('p',c.reason)); if (!d.enhancement_accepted) card.append(element('p',`VLM ${(c.confidence*100).toFixed(1)}% · threshold ${(event.settings.vlm_threshold*100).toFixed(0)}%`,'muted')); }
      const details = element('details'); const summary = element('summary','Detection history'); const content = element('div',undefined,'detection-history'); details.append(summary,content); details.open = openedHistories.has(d.detection_id);
      details.ontoggle = () => { if (details.open) { openedHistories.add(d.detection_id); history(d,content); } else openedHistories.delete(d.detection_id); };
      const requestButton=element('button','Request VLM for this detection');
      requestButton.onclick=async()=>{requestButton.disabled=true;try {const r=await fetch(`/api/detections/${d.detection_id}/analyze`,{method:'POST',headers:{'X-RGB-Request':'analysis'}});if(!r.ok)throw new Error(`Request failed (${r.status})`);requestButton.textContent='VLM requested';}catch(e){requestButton.textContent=e.message;requestButton.disabled=false;}};
      if (details.open) history(d,content); card.append(requestButton,details,evaluation(d)); target.append(card);
    });
    for (const opinion of event.opinions.filter(o=>o.kind === 'box_comparisons')) { if (opinion.errors.length) target.append(element('p',`${opinion.model}: ${opinion.errors.map(e=>e.error).join('; ')}`,'muted')); }
    const button = element('button',state.status === 'queued' || state.status === 'running' ? 'Analysis in progress' : 'Analyze detections'); button.disabled = ['queued','running','disabled'].includes(state.status);
    button.onclick = async () => { button.disabled = true; try { const r = await fetch(`/api/events/${id}/analyze`,{method:'POST',headers:{'X-RGB-Request':'analysis'}}); if (!r.ok) throw new Error(`Request failed (${r.status})`); show(id,true); } catch (e) { button.textContent = e.message; button.disabled = false; } };
    target.append(button,element('p',`Observation ID: ${event.event_id}`,'muted'));
    if (['queued','running','paused','manual_only'].includes(state.status)) detailTimer=setTimeout(()=>show(id,true),3000);
  } catch (e) { if(!refresh || !reviewIsEditing(target))target.replaceChildren(element('p',e.message));else detailTimer=setTimeout(()=>show(id,true),3000); }
}
function reviewIsEditing(target) { return [...target.querySelectorAll('details[data-evaluation]')].some(d=>d.open||d.querySelector('form').dataset.dirty==='true'); }
async function load(older=false) {
  try { const events = await get('/api/events'+(older && cursor ? '?before='+encodeURIComponent(cursor) : '')); if (!older) $('events').replaceChildren();
    for (const event of events) { const b = element('button',undefined,'event'); const img = element('img'); img.src = `/api/events/${event.event_id}/image.jpg`; img.alt = ''; img.loading = 'lazy'; const info = element('div'); info.append(element('strong',[...new Set(event.detections.map(d=>d.display_label))].join(', ')||'No detections above threshold'),element('span',date(event.observed_at)),element('span',`${event.source_id} · ${event.analysis.status.replaceAll('_',' ')}`)); b.append(img,info); b.onclick = () => show(event.event_id); $('events').append(b); }
    if (events.length) cursor = events.at(-1).cursor; $('older').disabled = events.length < 50; $('archive-status').textContent = $('events').children.length ? '' : 'Waiting for the first detected object.';
  } catch (e) { $('archive-status').textContent = `Archive unavailable: ${e.message}`; }
}
function syncThresholds() { $('detector-value').textContent=Number($('detector-threshold').value).toFixed(2); $('vlm-value').textContent=Number($('vlm-threshold').value).toFixed(2); }
$('detector-threshold').oninput=syncThresholds; $('vlm-threshold').oninput=syncThresholds;
$('save-settings').onclick=async () => { try { const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'settings'},body:JSON.stringify({detector_threshold:Number($('detector-threshold').value),vlm_threshold:Number($('vlm-threshold').value)})}); if (!r.ok) throw new Error(`Save failed (${r.status})`); settings=await r.json(); $('settings-status').textContent=`Saved revision ${settings.version}`; load(); if (selected) show(selected,true); } catch (e) { $('settings-status').textContent=e.message; } };
window.addEventListener('hashchange',()=>{ const match=location.hash.match(/^#event\/([a-f0-9-]{36})$/i); if (match && match[1]!==selected) show(match[1]); });
$('mode').onchange=async () => { const mode=$('mode').value; $('mode').disabled=true; try { const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'settings'},body:JSON.stringify({display_mode:mode})}); if (!r.ok) throw new Error(`Save failed (${r.status})`); settings=await r.json(); $('mode-status').textContent=`Saved revision ${settings.version}`; load(); if (selected) show(selected,true); } catch (e) { $('mode-status').textContent=e.message; $('mode').value=settings.display_mode; } $('mode').disabled=false; };
$('refresh').onclick = () => load(); $('older').onclick = () => load(true); $('source').onchange = live;
(async () => { try { settings=await get('/api/settings'); $('mode').value=settings.display_mode; $('detector-threshold').value=settings.detector_threshold; $('vlm-threshold').value=settings.vlm_threshold; syncThresholds(); sources=await get('/api/sources'); for (const s of sources) { const o=element('option',s.label); o.value=s.id; $('source').append(o); } live(); } catch { $('live-state').textContent='Sources unavailable'; } load(); const match=location.hash.match(/^#event\/([a-f0-9-]{36})$/i); if (match) show(match[1]); })();

let vlmPaused=false,vlmChanging=false,vlmGeneration=0;
function renderVlm(state) {
  vlmPaused=state.paused;
  $('vlm-model').textContent=`Label refinement: ${(state.models||[]).join(', ')||'Disabled'}`;
  $('pause-vlm').textContent=state.paused?'Resume VLM':'Pause VLM';
  $('pause-vlm').disabled=vlmChanging||!state.enabled;
  $('vlm-state').textContent=!state.enabled?'VLM disabled':state.paused?(state.active?'Paused · current request finishing':'Paused · VLM idle'):state.active?'Running · processing request':'Running · waiting for work';
}
async function vlmStatus(){const generation=vlmGeneration;try {const state=await get('/api/vlm/status');if(!vlmChanging&&generation===vlmGeneration)renderVlm(state);}catch(e){if(generation===vlmGeneration)$('vlm-state').textContent=`VLM status unavailable: ${e.message}`;}}
$('pause-vlm').onclick=async()=>{vlmChanging=true;vlmGeneration++;$('pause-vlm').disabled=true;try {const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'settings'},body:JSON.stringify({vlm_paused:!vlmPaused})});if(!r.ok)throw new Error(`Save failed (${r.status})`);const saved=await r.json();vlmPaused=saved.vlm_paused;$('pause-vlm').textContent=vlmPaused?'Resume VLM':'Pause VLM';vlmChanging=false;await vlmStatus();}catch(e){$('vlm-state').textContent=e.message;}finally{vlmChanging=false;$('pause-vlm').disabled=false;}};
vlmStatus();setInterval(vlmStatus,2000);

async function loadLocation() {
  const sourceId=$('source').value;
  try {const loc=await get(`/api/sources/${encodeURIComponent(sourceId)}/location`);if($('source').value!==sourceId)return;$('location-label').value=loc.label??'';$('location-lat').value=loc.latitude??'';$('location-lon').value=loc.longitude??'';
    $('location-status').textContent=`${loc.method} · revision ${loc.version}`;
    $('google-maps').hidden=loc.latitude==null;
    if (loc.latitude!=null) $('google-maps').href=`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(loc.latitude+','+loc.longitude)}`;
  } catch(e) {if($('source').value===sourceId)$('location-status').textContent=e.message;}
}
async function saveLocation(data,sourceId=$('source').value) {
  const r=await fetch(`/api/sources/${encodeURIComponent(sourceId)}/location`,{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'location'},body:JSON.stringify(data)});
  if (!r.ok) throw new Error(`Location save failed (${r.status})`);
  const loc=await r.json();const source=sources.find(s=>s.id===sourceId);if(source)source.location=loc.label||null;
  if($('source').value!==sourceId)return;
  $('camera-location').textContent=loc.label||(loc.latitude==null?'Location unassigned':`${loc.latitude.toFixed(5)}, ${loc.longitude.toFixed(5)}`);
  await loadLocation();
}
$('location-form').onsubmit=async e=>{e.preventDefault();try {await saveLocation({method:'manual',label:$('location-label').value.trim(),latitude:$('location-lat').value===''?null:Number($('location-lat').value),longitude:$('location-lon').value===''?null:Number($('location-lon').value),accuracy_m:null});} catch(e) {$('location-status').textContent=e.message;}};
$('disable-location').onclick=async()=>{try {await saveLocation({method:'disabled',label:'',latitude:null,longitude:null,accuracy_m:null});} catch(e) {$('location-status').textContent=e.message;}};
$('browser-location').onclick=()=>{
  const sourceId=$('source').value,label=$('location-label').value.trim();
  $('location-status').textContent='Requesting browser location…';
  if (!navigator.geolocation) {$('location-status').textContent='Browser location unavailable';return;}
  navigator.geolocation.getCurrentPosition(async position=>{if($('source').value!==sourceId)return;$('location-lat').value=position.coords.latitude;$('location-lon').value=position.coords.longitude;$('location-status').textContent='Coordinates received; saving…';try {await saveLocation({method:'browser',label,latitude:position.coords.latitude,longitude:position.coords.longitude,accuracy_m:position.coords.accuracy},sourceId);} catch(e) {if($('source').value===sourceId)$('location-status').textContent=e.message;}},error=>{if($('source').value===sourceId)$('location-status').textContent=({1:'Browser location permission denied',2:'Browser location unavailable',3:'Browser location timed out (20s). No coordinates received.'})[error.code]||error.message;},{enableHighAccuracy:false,timeout:20000,maximumAge:60000});
};


$('catch-up').onclick=async()=>{
  const source_id=$('source').value;
  $('catch-up').disabled=true;
  let batch_id;
  try {
    while(true){
      const r=await fetch('/api/analysis/catch-up',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'analysis'},body:JSON.stringify({source_id,...(batch_id?{batch_id}:{})})});
      if(!r.ok)throw new Error(`Catch up failed (${r.status})`);
      const batch=await r.json();batch_id=batch.batch_id;
      $('catch-up-status').textContent=`${batch.queued} observations requested${vlmPaused?' · waiting for Resume VLM':''}`;
      if(batch.done)break;
    }
  }catch(e){$('catch-up-status').textContent=e.message;}
  finally{$('catch-up').disabled=false;}
};
let classificationRunning=true,desiredClassification=null;
function renderClassification(s){
  desiredClassification={version:s.version??0,running:s.classification_running!==false,recording:s.record_observations!==false};
  classificationRunning=s.classification_running!==false;
  $('classification-mode').value=s.record_observations!==false?'record':'preview';
  $('classification-toggle').textContent=classificationRunning?'Pause classification':'Start classification';
  $('classification-toggle').disabled=false;
  $('classification-state').textContent='Control saved · awaiting Uno Q';
}
function acknowledgeClassification(s){
  if(!desiredClassification||s.applied_settings_version<desiredClassification.version||s.classification_running!==desiredClassification.running||s.record_observations!==desiredClassification.recording)return;
  $('classification-state').textContent=s.classification_running?(s.record_observations?'Running · recording observations':'Running · preview only'):'Paused · camera on';
}
async function setClassification(running){
  $('classification-toggle').disabled=true;
  try{
    const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'settings'},body:JSON.stringify({classification_running:running,record_observations:$('classification-mode').value==='record'})});
    if(!r.ok)throw new Error(`Save failed (${r.status})`);
    renderClassification(await r.json());
  }catch(e){$('classification-state').textContent=e.message;}
  finally{$('classification-toggle').disabled=false;}
}
$('classification-toggle').onclick=()=>setClassification(!classificationRunning);
$('classification-mode').onchange=()=>setClassification(classificationRunning);
get('/api/settings').then(renderClassification).catch(e=>{$('classification-state').textContent=e.message;});
