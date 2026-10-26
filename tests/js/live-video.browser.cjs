const {chromium}=require('playwright');
const http=require('node:http'),fs=require('node:fs'),assert=require('node:assert/strict');
const path=require('node:path');
const root=path.resolve(__dirname,'../../rgb/static');
const fixtures=path.resolve(__dirname,'fixtures');
const red=fs.readFileSync(fixtures+'/red.jpg'),green=fs.readFileSync(fixtures+'/green.jpg');
const sockets=new Set(),counts={};
const server=http.createServer((req,res)=>{
  const path=req.url.split('?')[0];
  if(path==='/controller.js'){res.setHeader('Content-Type','application/javascript');return res.end(fs.readFileSync(root+'/live-video.js'));}
  if(path==='/'){res.setHeader('Content-Type','text/html');res.setHeader('Content-Security-Policy',"default-src 'self'; img-src 'self' blob:; script-src 'self'");return res.end('<img id="image"><span id="state"></span><script src="/controller.js"></script>');}
  const n=counts[path]=(counts[path]||0)+1;
  if(path==='/no-headers'&&n===1)return;
  if(path==='/http'&&n===1){res.writeHead(503);return res.end();}
  res.writeHead(200,{'Content-Type':'multipart/x-mixed-replace; boundary=frame','Cache-Control':'no-store'});
  const part=jpeg=>Buffer.concat([Buffer.from('--frame\r\nContent-Type: image/jpeg\r\n\r\n'),jpeg,Buffer.from('\r\n')]);
  if(path==='/corrupt'&&n===1){res.write(part(Buffer.from('bad jpeg')));res.write(Buffer.from('--frame\r\n'));return;}
  if(path==='/fragmented') {
    const packet=Buffer.concat([part(green),Buffer.from('--frame\r\n')]);
    let offset=0;const fragments=setInterval(()=>{if(offset===packet.length){clearInterval(fragments);res.end();return;}res.write(packet.subarray(offset,offset+7));offset=Math.min(packet.length,offset+7);},1);
    res.on('close',()=>clearInterval(fragments));return;
  }
  if(path==='/trickle'&&n===1){const trickle=setInterval(()=>res.write('x'),30);res.on('close',()=>clearInterval(trickle));return;}
  if(path==='/oversized'&&n===1){res.write(Buffer.alloc(4*1024*1024+1,120));return;}
  if(path==='/length') {res.write(Buffer.concat([Buffer.from('--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+green.length+'\r\n\r\n'),green]));return;}
  res.write(part(n===1?red:green));res.write(Buffer.from('--frame\r\n'));
  if(path==='/eof'&&n===1)return res.end();
  if(((path==='/stall'||path==='/defaults')&&n===1)||path==='/slow')return;
  const timer=setInterval(()=>res.write(Buffer.concat([Buffer.from('Content-Type: image/jpeg\r\n\r\n'),green,Buffer.from('\r\n--frame\r\n')])),50);
  res.on('close',()=>clearInterval(timer));
});
server.on('connection',s=>{sockets.add(s);s.on('close',()=>sockets.delete(s));});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const browser=await chromium.launch({...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN} : {}),headless:true,args:['--no-sandbox']});
  try{
    const page=await browser.newPage();const origin=`http://127.0.0.1:${server.address().port}`;
    await page.addInitScript(() => {
      window.activeURLs = new Set();
      const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
      URL.createObjectURL = blob => { const url = create(blob); activeURLs.add(url); return url; };
      URL.revokeObjectURL = url => { activeURLs.delete(url); revoke(url); };
    });
    await page.goto(origin);
    const begin=path=>page.evaluate(path=>{window.player?.stop();window.states=[];window.player=new LiveVideo(document.querySelector('img'),s=>{states.push(s);document.querySelector('span').textContent=s;},{timeout:1200,retry:100});player.start(path);},path);
    const pixel=()=>page.evaluate(()=>{const c=document.createElement('canvas');c.width=32;c.height=24;const x=c.getContext('2d');x.drawImage(document.querySelector('img'),0,0);return Array.from(x.getImageData(4,4,1,1).data).slice(0,3);});
    await begin('/stall');await page.waitForFunction(()=>document.querySelector('img').naturalWidth===32);
    assert((await pixel())[0]>200);
    await page.waitForFunction(()=>states.some(s=>s.startsWith('Stale')));
    assert((await pixel())[0]>200,'last good red frame survives stall');
    await page.waitForFunction(()=>states.filter(s=>s==='LIVE').length>=2);await page.waitForTimeout(60);
    assert((await pixel())[1]>200);assert.equal(counts['/stall'],2);console.log('PASS silent stall: retained last frame and recovered to green');
    for(const path of ['/eof','/http','/corrupt','/trickle','/no-headers','/oversized']){
      await begin(path);await page.waitForFunction(()=>states.includes('LIVE'));await page.waitForFunction(()=>states.some(s=>s.includes('reconnecting')) && states.at(-1)==='LIVE');
      assert.equal(counts[path],2);assert((await pixel())[1]>200);console.log('PASS '+path+': recovered automatically');
    }
    await begin('/slow');await page.waitForFunction(()=>states.includes('LIVE'));
    await page.evaluate(()=>{states=[];player.start('/healthy');if(document.querySelector('img').hasAttribute('src'))throw new Error('Old camera frame retained');});
    await page.waitForTimeout(1600);
    assert.equal(counts['/slow'],1);assert.equal(counts['/healthy'],1);assert((await pixel())[1]>200);console.log('PASS same player source switch: old retries cancelled; healthy stream stays connected');
    await begin('/length');await page.waitForFunction(()=>states.includes('LIVE'));
    assert((await pixel())[1]>200);assert.equal(counts['/length'],1);console.log('PASS Content-Length: displays complete frame before next boundary');
    await page.evaluate(() => {
      player.stop(); states=[];
      window.realDecode=Image.prototype.decode;
      window.holdDecode=true;
      Image.prototype.decode=function(){if(holdDecode){holdDecode=false;return new Promise(()=>{});}return realDecode.call(this);};
      player=new LiveVideo(document.querySelector('img'),s=>states.push(s),{timeout:1200,retry:100});
      player.start('/decode-hang');
    });
    await page.waitForFunction(()=>states.includes('LIVE'));
    assert.equal(counts['/decode-hang'],2);
    await page.evaluate(()=>{Image.prototype.decode=realDecode;});
    console.log('PASS stalled decode: watchdog cancels decode and reconnects');
    await begin('/fragmented');await page.waitForFunction(()=>states.includes('LIVE'));assert((await pixel())[1]>200);console.log('PASS fragmented multipart: boundaries, headers and JPEG span network chunks');
    await begin('/healthy');await page.waitForTimeout(150);
    const healthyCount=counts['/healthy'];
    assert(await page.evaluate(()=>activeURLs.size<=2),'blob URLs remain bounded while rendering');
    await page.evaluate(()=>player.stop());await page.waitForTimeout(650);assert.equal(counts['/healthy'],healthyCount);assert.equal(await page.evaluate(()=>activeURLs.size),0);console.log('PASS stop: no reconnect and all blob URLs released');
    const started=Date.now();
    await page.evaluate(()=>{window.states=[];window.player=new LiveVideo(document.querySelector('img'),s=>states.push(s));player.start('/defaults');});
    await page.waitForFunction(()=>states.some(s=>s.startsWith('Stale')),null,{timeout:18000});
    assert((await pixel())[0]>200);
    await page.waitForFunction(()=>states.filter(s=>s==='LIVE').length>=2,null,{timeout:8000});
    assert(Date.now()-started>=20000);assert.equal(counts['/defaults'],2);
    console.log('PASS production defaults: 15-second frame watchdog + 5-second retry, last frame retained');
    await page.evaluate(()=>player.stop());
    assert.equal(await page.evaluate(()=>activeURLs.size),0);
  }finally{await browser.close();for(const s of sockets)s.destroy();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;for(const s of sockets)s.destroy();server.close();});
