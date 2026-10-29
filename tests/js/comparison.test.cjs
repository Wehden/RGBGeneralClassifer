const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.open=true;this.hidden=false;}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  removeAttribute(name){delete this[name];}
}
function setup(){
  const context=vm.createContext({window:{},document:{createElement:tag=>new Element(tag)},
    setTimeout,clearTimeout,AbortController});
  vm.runInContext(fs.readFileSync('rgb/static/comparison.js','utf8'),context);
  const panel=new Element('details'),content=new Element('div'),status=new Element('p'),images=[];
  const viewer=new context.window.RGBComparison.Viewer(panel,content,status,{
    fetch:async()=>{throw new Error('offline');},
    makeImage:()=>{const image=new Element('img');images.push(image);return image;}});
  viewer.running=true;viewer.generation=1;
  return {viewer,content,status,images};
}
function snapshot(id='frame-a'){
  return {frame_id:id,image_url:'/api/camera/comparison/image.jpg?frame_id='+id,
    total_boxes:3,age_s:1.2,status:'complete',rows:[{box_index:1,box:[.1,.2,.8,.9],
      object:{label:'bottle',confidence:.8},material:{status:'uncertain',label:null,
      best_candidate:'plastic',score:.65,similarity:.11,score_type:'supervised_material_head_score'}}]};
}
test('image and score table commit together after the matching image loads',async()=>{
  const {viewer,content,status,images}=setup();
  const loading=viewer.render(snapshot(),1);
  assert.equal(content.children.length,0);
  assert.equal(images[0].src,snapshot().image_url);
  images[0].onload();await loading;
  assert.equal(content.children.length,2);
  const table=content.children[1].children[0],cells=table.children[1].children[0].children;
  assert.deepEqual(Array.from(cells,c=>c.textContent),['2','bottle','80.0%','plastic','0.650 · learned','Uncertain']);
  assert.match(status.textContent,/1\/3 boxes.*1.2s/);
  viewer.stop();
});
test('switching source cancels pending images and cannot commit stale scores',async()=>{
  const {viewer,content,images}=setup();
  const loading=viewer.render(snapshot(),1).catch(()=>{});
  viewer.stop();
  await loading;
  assert.equal(content.children.length,0);
  assert.equal(images[0].src,undefined);
});
test('legacy cosine scores keep their semantics and stale snapshots are marked',async()=>{
  const {viewer,content,status,images}=setup();
  const data=snapshot();data.rows[0].material.score_type='uncalibrated_image_text_similarity';
  data.status='stale';data.age_s=8;
  const loading=viewer.render(data,1);images[0].onload();await loading;
  const cells=content.children[1].children[0].children[1].children[0].children;
  assert.equal(cells[4].textContent,'0.110 · similarity');
  assert.match(status.textContent,/stale snapshot/);
  viewer.stop();
});
test('comparison rejects unrelated image paths',async()=>{
  const {viewer}=setup(),data=snapshot();data.image_url='/api/camera/color/snapshot.jpg';
  await assert.rejects(viewer.render(data,1),/Unexpected comparison image/);
  viewer.stop();
});
test('a camera reset clears the previous image and table while waiting for new scores',async()=>{
  const {viewer,content,status,images}=setup();
  const loading=viewer.render(snapshot(),1);images[0].onload();await loading;
  assert.equal(content.children.length,2);
  viewer.fetch=async()=>({ok:true,json:async()=>({status:'running',rows:[]})});
  await viewer.poll(1);
  assert.equal(content.children.length,0);
  assert.equal(viewer.currentFrame,null);
  assert.match(status.textContent,/waiting for scored boxes/);
  viewer.stop();
});
