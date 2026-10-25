const $ = id => document.getElementById(id);
let sources = [], cursor = null, selected = null, timer, detailTimer, settings;
const openedHistories = new Set();
function element(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; }
async function get(url) { const r = await fetch(url); if (!r.ok) throw new Error(`Request failed (${r.status})`); return r.json(); }
function date(value) { return new Date(value).toLocaleString(); }
async function status() {
  try { const s = await get(`/api/live/${encodeURIComponent($('source').value)}/status`); $('live-state').textContent = s.error ? s.error : 'LIVE'; $('timing').textContent = `${s.inference_ms ?? '?'} ms inference`; }
  catch { $('live-state').textContent = 'Camera unavailable · reconnecting'; $('timing').textContent = ''; }
}
function live() {
  const s = sources.find(x => x.id === $('source').value); if (!s) return;
  $('live').src = `/api/live/${encodeURIComponent(s.id)}/stream`;
  $('camera-location').textContent = s.location ? `${s.location} · configured location` : 'Location unknown';
  status(); clearInterval(timer); timer = setInterval(status, 10000);
}
$('live').onerror = () => { $('live-state').textContent = 'Camera unavailable · reconnecting'; setTimeout(live, 5000); };
async function history(d, node) {
  try { const entries = await get(`/api/detections/${d.detection_id}/history`); node.replaceChildren();
    if (!entries.length) node.append(element('p',`Classifier: ${d.label} · ${(d.confidence*100).toFixed(1)}%`));
    for (const entry of entries) { const c = entry.candidate; let text; if (entry.kind === 'initial') text=`${date(entry.recorded_at)} · Classifier: ${entry.initial.label} ${(entry.initial.confidence*100).toFixed(1)}%`; else if (entry.kind === 'enhanced') text=`${date(entry.updated_at)} · ${c.model}: ${c.label} ${(c.confidence*100).toFixed(1)}% · ${c.comparison}`; else if (entry.kind === 'model_attempt') text=`${date(entry.started_at)} · ${entry.model} · ${entry.outcome} · ${entry.latency_ms == null ? '' : entry.latency_ms+' ms'} · ${entry.error_type ?? ''}`; else if (entry.kind === 'processing_failure') text=`${date(entry.recorded_at)} · Processing failed · ${entry.error_type}`; else text=`${date(entry.reviewed_at)} · Evaluation: ${entry.reference_label} · classifier ${entry.detector_result} · VLM ${entry.vlm_result}`; node.append(element('p',text)); }
  } catch (e) { node.textContent = e.message; }
}
function evaluation(d) {
  const details=element('details'); details.append(element('summary','Evaluate detection')); const form=element('form',undefined,'evaluation');
  const reference=element('input'); reference.placeholder='Reference label'; reference.required=true; reference.maxLength=120; reference.setAttribute('aria-label','Reference label');
  const notes=element('textarea'); notes.placeholder='Evaluation notes'; notes.maxLength=4000; notes.setAttribute('aria-label','Evaluation notes');
  function outcome(name) { const label=element('label',name); const select=element('select'); for (const value of ['unscored','correct','incorrect']) { const option=element('option',value); option.value=value; select.append(option); } label.append(select); return [label,select]; }
  const [detectorLabel,detector]=outcome('Classifier'); const [vlmLabel,vlm]=outcome('VLM'); const button=element('button','Save evaluation'); const message=element('span'); form.append(reference,detectorLabel,vlmLabel,notes,button,message);
  form.onsubmit=async e => { e.preventDefault(); button.disabled=true; try { const r=await fetch(`/api/detections/${d.detection_id}/evaluate`,{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'evaluation'},body:JSON.stringify({reference_label:reference.value,detector_result:detector.value,vlm_result:vlm.value,notes:notes.value})}); if (!r.ok) throw new Error(`Save failed (${r.status})`); message.textContent='Evaluation recorded'; } catch (e) { message.textContent=e.message; } button.disabled=false; };
  details.append(form); return details;
}
async function show(id, refresh=false) {
  selected = id; if (location.hash !== '#event/'+id) location.hash='event/'+id; clearTimeout(detailTimer); const target = $('detail'); if (!refresh) target.replaceChildren(element('p','Loading…'));
  try {
    const event = await get(`/api/events/${id}`); if (selected !== id) return;
    target.replaceChildren(); const frame = element('div',undefined,'evidence-image'); const img = element('img'); img.src = `/api/events/${id}/image.jpg`; img.alt = 'Archived camera observation'; frame.append(img);
    event.detections.forEach((d,i) => { const b = element('span',undefined,'box'); b.style.top = `${d.box[0]*100}%`; b.style.left = `${d.box[1]*100}%`; b.style.height = `${(d.box[2]-d.box[0])*100}%`; b.style.width = `${(d.box[3]-d.box[1])*100}%`; b.append(element('span',`${i+1}. ${d.display_label}`)); frame.append(b); });
    target.append(frame,element('h3',date(event.observed_at)),element('p',`${event.source_id} · ${event.location ?? 'Location unknown'}`,'muted'));
    const state = event.analysis; const progress = state.total_boxes ? ` · ${state.completed_boxes}/${state.total_boxes} boxes` : '';
    target.append(element('p',`Analysis: ${state.status.replaceAll('_',' ')}${progress}`,'analysis-state'));
    event.detections.forEach((d,i) => {
      const card = element('section',undefined,'reading'); const c = d.enhancement;
      card.append(element('h3',`${i+1}. ${d.display_label}`),element('p',`${d.label_provider === 'vlm' ? c.model : event.detector.model} · ${((d.label_provider === 'vlm' ? c.confidence : d.confidence)*100).toFixed(1)}%`,'muted'));
      if (c) { card.append(element('p',`${d.label} → ${c.label} · ${c.comparison}`,'comparison'),element('p',c.reason)); if (!d.enhancement_accepted) card.append(element('p',`VLM ${(c.confidence*100).toFixed(1)}% · threshold ${(event.settings.vlm_threshold*100).toFixed(0)}%`,'muted')); }
      const details = element('details'); const summary = element('summary','Detection history'); const content = element('div',undefined,'detection-history'); details.append(summary,content); details.open = openedHistories.has(d.detection_id);
      details.ontoggle = () => { if (details.open) { openedHistories.add(d.detection_id); history(d,content); } else openedHistories.delete(d.detection_id); };
      if (details.open) history(d,content); card.append(details,evaluation(d)); target.append(card);
    });
    for (const opinion of event.opinions.filter(o=>o.kind === 'box_comparisons')) { if (opinion.errors.length) target.append(element('p',`${opinion.model}: ${opinion.errors.map(e=>e.error).join('; ')}`,'muted')); }
    const button = element('button',state.status === 'queued' || state.status === 'running' ? 'Analysis in progress' : 'Analyze detections'); button.disabled = ['queued','running','disabled'].includes(state.status);
    button.onclick = async () => { button.disabled = true; try { const r = await fetch(`/api/events/${id}/analyze`,{method:'POST',headers:{'X-RGB-Request':'analysis'}}); if (!r.ok) throw new Error(`Request failed (${r.status})`); show(id,true); } catch (e) { button.textContent = e.message; button.disabled = false; } };
    target.append(button,element('p',`Observation ID: ${event.event_id}`,'muted'));
    if (['queued','running'].includes(state.status)) detailTimer=setTimeout(()=>show(id,true),3000);
  } catch (e) { target.replaceChildren(element('p',e.message)); }
}
async function load(older=false) {
  try { const events = await get('/api/events'+(older && cursor ? '?before='+encodeURIComponent(cursor) : '')); if (!older) $('events').replaceChildren();
    for (const event of events) { const b = element('button',undefined,'event'); const img = element('img'); img.src = `/api/events/${event.event_id}/image.jpg`; img.alt = ''; img.loading = 'lazy'; const info = element('div'); info.append(element('strong',[...new Set(event.detections.map(d=>d.display_label))].join(', ')),element('span',date(event.observed_at)),element('span',`${event.source_id} · ${event.analysis.status.replaceAll('_',' ')}`)); b.append(img,info); b.onclick = () => show(event.event_id); $('events').append(b); }
    if (events.length) cursor = events.at(-1).cursor; $('older').disabled = events.length < 50; $('archive-status').textContent = $('events').children.length ? '' : 'Waiting for the first detected object.';
  } catch (e) { $('archive-status').textContent = `Archive unavailable: ${e.message}`; }
}
function syncThresholds() { $('detector-value').textContent=Number($('detector-threshold').value).toFixed(2); $('vlm-value').textContent=Number($('vlm-threshold').value).toFixed(2); }
$('detector-threshold').oninput=syncThresholds; $('vlm-threshold').oninput=syncThresholds;
$('save-settings').onclick=async () => { try { const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json','X-RGB-Request':'settings'},body:JSON.stringify({detector_threshold:Number($('detector-threshold').value),vlm_threshold:Number($('vlm-threshold').value)})}); if (!r.ok) throw new Error(`Save failed (${r.status})`); settings=await r.json(); $('settings-status').textContent=`Saved revision ${settings.version}`; load(); if (selected) show(selected,true); } catch (e) { $('settings-status').textContent=e.message; } };
window.addEventListener('hashchange',()=>{ const match=location.hash.match(/^#event\/([a-f0-9-]{36})$/i); if (match && match[1]!==selected) show(match[1]); });
$('refresh').onclick = () => load(); $('older').onclick = () => load(true); $('source').onchange = live;
(async () => { try { settings=await get('/api/settings'); $('detector-threshold').value=settings.detector_threshold; $('vlm-threshold').value=settings.vlm_threshold; syncThresholds(); sources=await get('/api/sources'); for (const s of sources) { const o=element('option',s.label); o.value=s.id; $('source').append(o); } live(); } catch { $('live-state').textContent='Sources unavailable'; } load(); const match=location.hash.match(/^#event\/([a-f0-9-]{36})$/i); if (match) show(match[1]); })();
