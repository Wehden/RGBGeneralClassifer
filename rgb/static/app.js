const $ = id => document.getElementById(id);
let sources = [], cursor = null, selected = null, timer;
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
async function show(id) {
  selected = id; const target = $('detail'); target.replaceChildren(element('p','Loading…'));
  try {
    const event = await get(`/api/events/${id}`); if (selected !== id) return;
    target.replaceChildren(); const frame = element('div',undefined,'evidence-image'); const img = element('img'); img.src = `/api/events/${id}/image.jpg`; img.alt = 'Archived camera observation'; frame.append(img);
    for (const d of event.detections) { const b = element('span',undefined,'box'); b.style.top = `${d.box[0]*100}%`; b.style.left = `${d.box[1]*100}%`; b.style.height = `${(d.box[2]-d.box[0])*100}%`; b.style.width = `${(d.box[3]-d.box[1])*100}%`; b.append(element('span',`${d.label} ${Math.round(d.confidence*100)}%`)); frame.append(b); }
    target.append(frame,element('h3',date(event.observed_at)),element('p',`${event.source_id} · ${event.location ?? 'Location unknown'}`,'muted'),element('h3','Detector observations'));
    for (const d of event.detections) target.append(element('p',`${d.label} · ${(d.confidence*100).toFixed(1)}% confidence`));
    target.append(element('p',event.detector.model,'muted'),element('h3','Model opinions'));
    if (!event.opinions.length) target.append(element('p','No additional model opinion available.','muted'));
    for (const o of event.opinions) { target.append(element('h4',o.model),element('p',o.status === 'complete' ? o.text + (o.truncated ? '\n[Response reached its output limit.]' : '') : o.status === 'skipped' ? `Not sampled: ${o.reason}` : `Unavailable: ${o.error}`,'opinion')); }
    target.append(element('h3','Sorting assessment'),element('p','Unassessed. Confirm material, condition, and local rules before choosing a disposal stream.'),element('p',`Observation ID: ${event.event_id}`,'muted'));
  } catch (e) { target.replaceChildren(element('p',e.message)); }
}
async function load(older=false) {
  try { const events = await get('/api/events'+(older && cursor ? '?before='+encodeURIComponent(cursor) : '')); if (!older) $('events').replaceChildren();
    for (const event of events) { const b = element('button',undefined,'event'); const img = element('img'); img.src = `/api/events/${event.event_id}/image.jpg`; img.alt = ''; img.loading = 'lazy'; const info = element('div'); info.append(element('strong',[...new Set(event.detections.map(d=>d.label))].join(', ')),element('span',date(event.observed_at)),element('span',event.source_id)); b.append(img,info); b.onclick = () => show(event.event_id); $('events').append(b); }
    if (events.length) cursor = events.at(-1).cursor; $('older').disabled = events.length < 50; $('archive-status').textContent = $('events').children.length ? '' : 'Waiting for the first detected object.';
  } catch (e) { $('archive-status').textContent = `Archive unavailable: ${e.message}`; }
}
$('refresh').onclick = () => load(); $('older').onclick = () => load(true); $('source').onchange = live;
(async () => { try { sources = await get('/api/sources'); for (const s of sources) { const o = element('option',s.label); o.value = s.id; $('source').append(o); } live(); } catch { $('live-state').textContent = 'Sources unavailable'; } load(); })();
