const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

class Element {
  constructor(tag){this.tag=tag;this.children=[];this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
}

test('records keep camera names and distinguish local similarity from VLM confidence',async()=>{
  const nodes=new Map();
  const base={observed_at:'2026-10-27T12:00:00Z',source_id:'femto-bolt',initial_label:'bottle',
    initial_confidence:.8,enhanced_label:null,analysis_status:'complete',observation_id:'scene'};
  const records=[
    {...base,camera_name:'Femto Bolt',location:'Desk',
      local_material:{status:'uncertain',label:null,similarity:.321},
      materials:[{label:'metal',confidence:.8}]},
    {...base,local_material:{status:'identified',label:'plastic',similarity:.38}},
  ];
  const context=vm.createContext({
    document:{getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element('div'));return nodes.get(id);},
      createElement:tag=>new Element(tag)},
    fetch:async url=>({ok:true,json:async()=>url==='/api/sources'?
      [{id:'femto-bolt',label:'Attached camera'}]:{records,next_cursor:null,queried_at:base.observed_at}}),
    URLSearchParams,navigator:{},location:{origin:'http://127.0.0.1:8080'},
  });
  vm.runInContext(fs.readFileSync('rgb/static/records.js','utf8'),context);
  await new Promise(setImmediate);
  const rows=nodes.get('rows').children;
  assert.equal(rows.length,2);
  assert.equal(rows[0].children[1].textContent,'Femto Bolt · Desk');
  assert.equal(rows[0].children[4].textContent,
    'Local: uncertain · similarity 0.321 (uncalibrated) / VLM: metal 80.0%');
  assert.equal(rows[1].children[1].textContent,'Attached camera');
  assert.equal(rows[1].children[4].textContent,'Local: plastic · similarity 0.380 (uncalibrated)');
  assert.equal(rows[0].children[6].children[0].href,'/#event/scene');
});
