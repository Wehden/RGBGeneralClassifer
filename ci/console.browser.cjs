/* Real Flask/Waitress routes and browser UI; camera/model readings are synthetic. */
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const {once}=require('node:events');
const readline=require('node:readline');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const reports=path.join(root,'ci-results');
fs.mkdirSync(reports,{recursive:true});

async function fixture(rgbOnly){
  const process=spawn(global.process.env.CI_PYTHON||'python3',
    ['ci/synthetic_service.py',...(rgbOnly?['--rgb-only']:['--cameras'])],
    {cwd:root,stdio:['ignore','pipe','pipe']});
  let log='';
  process.stderr.on('data',chunk=>{log+=chunk;});
  const lines=readline.createInterface({input:process.stdout});
  const startup=Promise.race([
    once(lines,'line').then(([line])=>JSON.parse(line)),
    once(process,'exit').then(([code])=>{throw new Error(`Fixture exited ${code}: ${log}`);}),
    new Promise((_,reject)=>{const timer=setTimeout(()=>reject(new Error('Fixture startup timeout')),30000);timer.unref();}),
  ]);
  try{
    const state=await startup;
    return {...state,stop:async()=>{
      const exited=once(process,'exit');process.kill('SIGTERM');
      await exited;lines.close();
      fs.writeFileSync(path.join(reports,`console-${rgbOnly?'rgb':'depth'}.log`),log);
    }};
  }catch(error){process.kill('SIGTERM');lines.close();throw error;}
}

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],
    ...(process.env.CHROME_BIN?{executablePath:process.env.CHROME_BIN}:{})});
  try{
    for(const rgbOnly of [false,true]){
      const service=await fixture(rgbOnly);
      const context=await browser.newContext({acceptDownloads:true,viewport:{width:1440,height:1000}});
      await context.tracing.start({screenshots:true,snapshots:true,sources:false});
      const page=await context.newPage(),errors=[];
      page.on('pageerror',error=>errors.push(error.message));
      try{
        await page.goto(service.url);
        await page.waitForFunction(()=>!document.querySelector('#pause-capture').disabled);
        await page.waitForFunction(()=>document.querySelector('#events .event'));
        const initial=await (await page.request.get(service.url+'/api/settings')).json();
        assert.equal(initial.capture_paused,false);
        await page.click('#pause-capture');
        await page.waitForFunction(()=>document.querySelector('#pause-capture').textContent==='Resume capture');
        const paused=await (await page.request.get(service.url+'/api/settings')).json();
        assert.equal(paused.capture_paused,true);
        await page.fill('#capture-interval','120');
        await page.click('#apply-capture-interval');
        await page.waitForFunction(()=>document.querySelector('#apply-capture-interval').textContent==='Applied');
        assert.equal((await (await page.request.get(service.url+'/api/settings')).json()).capture_interval_s,120);
        await page.reload();
        await page.waitForFunction(()=>document.querySelector('#pause-capture').textContent==='Resume capture');
        assert.equal(await page.inputValue('#capture-interval'),'120');
        // Capture once must remain available while automatic capture is paused.
        const captured=page.waitForResponse(r=>r.url().endsWith('/api/camera/observations')&&r.request().method()==='POST');
        await page.click('#capture-observation');
        assert.equal((await captured).status(),201);
        await page.waitForFunction(()=>document.querySelector('#camera-archive-status').textContent.includes('Archived frame'));
        await page.waitForFunction(()=>document.querySelector('#comparison-content table tbody tr'));
        assert((await page.textContent('#comparison-content')).includes('plastic'));
        if(!rgbOnly){
          assert((await page.textContent('#comparison-content')).includes('(cyl)'));
          for(const channel of ['depth','ir']){
            await page.click(`[data-channel="${channel}"]`);
            await page.waitForFunction(channel=>document.querySelector('#camera-channel').value===channel,channel);
            await page.waitForFunction(()=>document.querySelector('#live').naturalWidth>0);
          }
          await page.click('[data-channel="pointcloud"]');
          await page.waitForFunction(()=>document.querySelector('#live-state').textContent.includes('3D LIVE'));
          assert(await page.locator('#pointcloud-view').isVisible());
          await page.locator('#pointcloud').press('ArrowRight');
          await page.click('#reset-pointcloud');
          const raw=await page.request.get(service.url+'/api/camera/depth/raw.png');
          assert.equal(raw.status(),200);
          assert.equal(raw.headers()['x-rgb-plane'],'depth_native');
          assert.equal(await page.locator('#camera-select').isEnabled(),true);
          assert.equal(await page.locator('#camera-select option').count(),3);
          for(const [backend,bits] of [['realsense',8],['v4l2',null],['orbbec',16]]){
            const switched=page.waitForResponse(response=>response.url().endsWith('/api/camera/select'));
            await page.selectOption('#camera-select',backend);
            assert.equal((await switched).status(),200);
            await page.waitForFunction(backend=>document.querySelector('#camera-select').value===backend,backend);
            const camera=await (await page.request.get(service.url+'/api/camera/status')).json();
            assert.equal(camera.backend,backend);
            for(const channel of ['depth','ir','pointcloud'])
              assert.equal(await page.locator(`[data-channel="${channel}"]`).isVisible(),bits!==null);
            if(bits!==null){
              await page.click('[data-channel="ir"]');
              await page.waitForFunction(bits=>document.querySelector('#raw-link').textContent===`Raw ${bits}-bit ir`,bits);
            }else{
              assert.equal(await page.inputValue('#camera-channel'),'color');
              assert.equal((await page.request.get(service.url+'/api/camera/depth/raw.png')).status(),404);
            }
            assert.equal((await (await page.request.get(service.url+'/api/settings')).json()).capture_paused,true);
          }
        }else{
          assert.equal(await page.locator('#camera-select').isEnabled(),false);
          assert(!(await page.textContent('#comparison-content')).includes('Volume estimate'));
          for(const channel of ['depth','ir','pointcloud'])
            assert.equal(await page.locator(`[data-channel="${channel}"]`).isVisible(),false);
          assert.equal((await page.request.get(service.url+'/api/camera/depth/raw.png')).status(),404);
        }
        await page.click('#pause-capture');
        await page.waitForFunction(()=>document.querySelector('#pause-capture').textContent==='Pause capture');
        assert.equal((await (await page.request.get(service.url+'/api/settings')).json()).capture_paused,false);
        await page.goto(service.url+'/records');
        await page.waitForFunction(()=>document.querySelector('#rows tr'));
        assert.equal(await page.locator('#volume-column').isVisible(),!rgbOnly);
        if(!rgbOnly)assert((await page.textContent('#rows')).includes('(cyl)'));
        await page.fill('input[name="material"]','not-a-material');
        await page.locator('#query button').click();
        await page.waitForFunction(()=>document.querySelector('#rows').children.length===0);
        await page.fill('input[name="material"]','plastic');
        await page.locator('#query button').click();
        await page.waitForFunction(()=>document.querySelector('#rows tr'));
        for(const format of ['parquet','arrow']){
          const [download]=await Promise.all([page.waitForEvent('download'),page.click('#query-'+format)]);
          assert.equal(await download.failure(),null);
          const filename=path.join(reports,`synthetic-records-${rgbOnly?'rgb':'depth'}.${format}`);
          await download.saveAs(filename);
          assert(fs.statSync(filename).size>0);
        }
        assert.deepEqual(errors,[],'Console JavaScript must not raise uncaught exceptions');
        console.log(`PASS real console ${rgbOnly?'RGB':'RGB-D'}: capture controls, paired materials, camera capabilities, records and downloads`);
      }catch(error){
        await page.screenshot({path:path.join(reports,`console-${rgbOnly?'rgb':'depth'}-failure.png`),fullPage:true});
        throw error;
      }finally{
        await context.tracing.stop({path:path.join(reports,`console-${rgbOnly?'rgb':'depth'}-trace.zip`)});
        await context.close();await service.stop();
      }
    }
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
