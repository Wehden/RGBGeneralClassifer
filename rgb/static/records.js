const $=id=>document.getElementById(id);
let active=new URLSearchParams(),next=null;
const sourceNames=fetch('/api/sources').then(r=>r.ok?r.json():[]).then(sources=>{
  const select=$('record-source');
  if(select)for(const source of sources){const option=document.createElement('option');option.value=source.id;option.textContent=source.label;select.append(option);}
  return Object.fromEntries(sources.map(source=>[source.id,source.label]));
}).catch(()=>({}));
function cell(row,text) { const td=document.createElement('td'); td.textContent=text;row.append(td);return td; }
function score(label,n) {return label==null?'—':label+(n==null?'':` ${(n*100).toFixed(1)}%`);}
async function search(before=null) {
  const params=new URLSearchParams(active); if (before) params.set('before',before);
  const url='/api/review/records?'+params.toString();
  $('request-url').textContent=url; $('query-json').href=url;
  for (const format of ['parquet','arrow']) $('query-'+format).href=url+'&format='+format;
  $('copy-query').onclick=async()=>{try {await navigator.clipboard.writeText(location.origin+url); $('query-status').textContent='Request URL copied';} catch {$('query-status').textContent='Select and copy the request URL below';}};
  $('query-status').textContent='Loading records…'; $('next').disabled=true;
  try {
    const response=await fetch(url); if (!response.ok) throw new Error(`Query failed (${response.status})`);
    const data=await response.json(),names=await sourceNames; $('rows').replaceChildren();
    for (const r of data.records) {
      const tr=document.createElement('tr'); cell(tr,new Date(r.observed_at).toLocaleString());cell(tr,(r.camera_name||names[r.source_id]||r.source_id)+(r.location?' · '+r.location:''));
      cell(tr,score(r.initial_label,r.initial_confidence));cell(tr,score(r.enhanced_label,r.enhanced_confidence));
      const local=r.local_material;
      const materialTexts=[];
      if(local) {
        const reading=local.score_type==='supervised_material_head_score'
          ? `learned score ${local.score.toFixed(3)} (appearance estimate)`
          : `similarity ${local.similarity.toFixed(3)} (uncalibrated)`;
        materialTexts.push(`Local: ${local.status==='identified'?local.label:'uncertain'} · ${reading}`);
      }
      if(r.materials?.length)materialTexts.push('VLM: '+r.materials.map(m=>score(m.label,m.confidence)).join(' · '));
      cell(tr,materialTexts.join(' / ')||'Not read');cell(tr,r.analysis_status);
      const link=document.createElement('a'); link.href='/#event/'+r.observation_id;link.textContent='Open observation';cell(tr,'').append(link);$('rows').append(tr);
    }
    next=data.next_cursor; $('next').disabled=!next; $('query-status').textContent=`${data.records.length} records · Queried ${new Date(data.queried_at).toLocaleString()}`;
  } catch(e) {$('query-status').textContent=e.message;}
}
$('query').onsubmit=e=>{e.preventDefault();active=new URLSearchParams();for (const [key,raw] of new FormData(e.target)) {const v=raw.trim();if(v)active.set(key,['from','to'].includes(key)?new Date(v).toISOString():v);}search();};
$('next').onclick=()=>search(next);
search();
