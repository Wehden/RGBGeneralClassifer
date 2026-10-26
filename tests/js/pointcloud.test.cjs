const {test}=require('node:test');
const assert=require('node:assert/strict');
const {decode,matrix,Viewer}=require('../../rgb/static/pointcloud.js');

function frame(headerChanges={}) {
  const header={schema_version:1,encoding:'xyz-f32le-rgb-u8-pad1',point_stride_bytes:16,
    point_count:1,units:'metres',frame_id:'test-frame',median_z_m:2,max_range_m:5,...headerChanges};
  const text=new TextEncoder().encode(JSON.stringify(header));
  const buffer=new ArrayBuffer(4+text.length+16),view=new DataView(buffer);
  view.setUint32(0,text.length,true);new Uint8Array(buffer,4,text.length).set(text);
  const offset=4+text.length;
  view.setFloat32(offset,1,true);view.setFloat32(offset+4,2,true);view.setFloat32(offset+8,3,true);
  new Uint8Array(buffer,offset+12,3).set([10,20,30]);
  return buffer;
}
test('point cloud frame preserves little-endian positions and RGB bytes',()=>{
  const result=decode(frame()),view=new DataView(result.data.buffer,result.data.byteOffset,result.data.byteLength);
  assert.equal(view.getFloat32(0,true),1);assert.equal(view.getFloat32(4,true),2);assert.equal(view.getFloat32(8,true),3);
  assert.deepEqual([...result.data.subarray(12,15)],[10,20,30]);
});
test('point cloud decoder rejects truncation, wrong units and excessive point counts',()=>{
  assert.throws(()=>decode(frame().slice(0,-1)));
  assert.throws(()=>decode(frame({units:'millimetres'})));
  assert.throws(()=>decode(frame({point_count:30001})));
  assert.throws(()=>decode(new ArrayBuffer(3)));
});
test('camera-facing 3D projection keeps the optical center centered and depth visible',()=>{
  const m=matrix([0,0,-2],2,0,0,16/9);
  const x=m[8]*-2+m[12],y=m[9]*-2+m[13],z=m[10]*-2+m[14],w=m[11]*-2+m[15];
  assert.equal(x/w,0);assert.equal(y/w,0);assert.ok(z/w>-1&&z/w<1);
});
test('leaving the 3D view discards late frames and stops polling',async()=>{
  const canvas={getContext:kind=>kind==='2d'?{}:null,addEventListener(){}};
  const messages=[],viewer=new Viewer(canvas,text=>messages.push(text));
  let resolve;
  const original=global.fetch;
  global.fetch=()=>new Promise(r=>{resolve=r;});
  try {
    viewer.active=true;viewer.generation=1;viewer.started=performance.now();viewer.frames=0;
    const pending=viewer.poll(1);viewer.stop();
    resolve({ok:true,arrayBuffer:async()=>frame()});await pending;
    assert.equal(viewer.frame,null);assert.equal(viewer.timer,undefined);assert.equal(messages.length,0);
  } finally {global.fetch=original;}
});
