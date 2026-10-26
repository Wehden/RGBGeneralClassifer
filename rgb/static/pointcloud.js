/* A local RGB-D viewer; no CDN, WebSocket server, or saved observations. */
(function(root) {
  'use strict';
  function decode(buffer) {
    if (buffer.byteLength < 4) throw new Error('Incomplete point cloud frame');
    const length = new DataView(buffer).getUint32(0, true);
    if (length > 65536 || 4 + length > buffer.byteLength) throw new Error('Invalid point cloud header');
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
    if (header.schema_version !== 1 || header.encoding !== 'xyz-f32le-rgb-u8-pad1' ||
        header.point_stride_bytes !== 16 || header.units !== 'metres' ||
        !Number.isInteger(header.point_count) || header.point_count < 0 || header.point_count > 30000 ||
        buffer.byteLength !== 4 + length + header.point_count * 16)
      throw new Error('Unsupported point cloud layout');
    return {header, data: new Uint8Array(buffer, 4 + length)};
  }
  function multiply(a, b) {
    const result = new Float32Array(16);
    for (let column = 0; column < 4; column++)
      for (let row = 0; row < 4; row++)
        for (let k = 0; k < 4; k++) result[column*4+row] += a[k*4+row]*b[column*4+k];
    return result;
  }
  function cross(a,b) { return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]; }
  function normalize(a) { const n=Math.hypot(...a); return a.map(v=>v/n); }
  function dot(a,b) { return a.reduce((sum,v,i)=>sum+v*b[i],0); }
  function matrix(target, distance, yaw, pitch, aspect) {
    const back=normalize([Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)]);
    const eye=target.map((v,i)=>v+distance*back[i]);
    const right=normalize(cross([0,1,0],back)), up=cross(back,right);
    const view=new Float32Array([right[0],up[0],back[0],0,right[1],up[1],back[1],0,
      right[2],up[2],back[2],0,-dot(right,eye),-dot(up,eye),-dot(back,eye),1]);
    const f=1/Math.tan(Math.PI/6),near=.05,far=50;
    const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,
      0,0,2*far*near/(near-far),0]);
    return multiply(projection,view);
  }
  function depthColor(t) {
    t=Math.min(1,Math.max(0,t));
    return [Math.round(255*(.15+.85*t)),Math.round(255*(.15+.7*(1-Math.abs(2*t-1)))),Math.round(255*(.9-.8*t))];
  }
  class Viewer {
    constructor(canvas, status) {
      this.canvas=canvas;this.status=status;this.active=false;this.generation=0;this.frame=null;
      this.target=[0,0,-2];this.distance=2;this.yaw=0;this.pitch=0;this.colorMode='rgb';
      this.pointer=null;this.renderPending=false;this.initialized=false;
      try {
        this.gl=canvas.getContext('webgl',{alpha:false,antialias:false});
        if(this.gl)this.initGL();
        else {this.context=canvas.getContext('2d');if(!this.context)throw new Error('No canvas renderer');}
        this.available=true;
      } catch(error) {this.available=false;this.error=error.message;}
      canvas.addEventListener('contextmenu',event=>event.preventDefault());
      canvas.addEventListener('pointerdown',event=>{
        this.pointer={id:event.pointerId,x:event.clientX,y:event.clientY,pan:event.shiftKey||event.button===2};
        canvas.setPointerCapture(event.pointerId);canvas.focus();event.preventDefault();
      });
      canvas.addEventListener('pointermove',event=>{
        const p=this.pointer;if(!p||p.id!==event.pointerId)return;
        const dx=event.clientX-p.x,dy=event.clientY-p.y;p.x=event.clientX;p.y=event.clientY;
        if(p.pan) {
          const scale=this.distance*.002;
          this.target[0]-=dx*scale*Math.cos(this.yaw);this.target[2]+=dx*scale*Math.sin(this.yaw);
          this.target[1]+=dy*scale;
        } else {this.yaw-=dx*.007;this.pitch=Math.max(-1.5,Math.min(1.5,this.pitch+dy*.007));}
        this.scheduleDraw();
      });
      for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{this.pointer=null;});
      canvas.addEventListener('wheel',event=>{
        event.preventDefault();this.distance=Math.max(.15,Math.min(20,this.distance*Math.exp(event.deltaY*.001)));
        this.scheduleDraw();
      },{passive:false});
      canvas.addEventListener('dblclick',()=>this.reset());
      canvas.addEventListener('keydown',event=>{
        if(event.key==='Home')this.reset();
        else if(event.key==='ArrowLeft')this.yaw-=.1;
        else if(event.key==='ArrowRight')this.yaw+=.1;
        else if(event.key==='ArrowUp')this.pitch=Math.min(1.5,this.pitch+.1);
        else if(event.key==='ArrowDown')this.pitch=Math.max(-1.5,this.pitch-.1);
        else return;
        event.preventDefault();this.scheduleDraw();
      });
      if(typeof ResizeObserver!=='undefined') {
        this.resizeObserver=new ResizeObserver(()=>this.scheduleDraw());this.resizeObserver.observe(canvas);
      }
    }
    initGL() {
      const gl=this.gl;
      const shader=(kind,source)=>{
        const s=gl.createShader(kind);gl.shaderSource(s,source);gl.compileShader(s);
        if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
        return s;
      };
      this.program=gl.createProgram();
      gl.attachShader(this.program,shader(gl.VERTEX_SHADER,`
        attribute vec3 position; attribute vec3 rgb;
        uniform mat4 mvp; uniform float pointSize; uniform float depthMode; uniform float maxRange;
        varying vec3 color;
        void main() {
          gl_Position=mvp*vec4(position.x,-position.y,-position.z,1.0);gl_PointSize=pointSize;
          float t=clamp(position.z/maxRange,0.0,1.0);
          vec3 depthColor=vec3(0.15+0.85*t,0.15+0.7*(1.0-abs(2.0*t-1.0)),0.9-0.8*t);
          color=mix(rgb,depthColor,depthMode);
        }`));
      gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,`
        precision mediump float; varying vec3 color;
        void main() { if(distance(gl_PointCoord,vec2(0.5))>0.5)discard; gl_FragColor=vec4(color,1.0); }`));
      gl.linkProgram(this.program);
      if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
      this.buffer=gl.createBuffer();gl.useProgram(this.program);
      this.uniforms={};
      for(const name of ['mvp','pointSize','depthMode','maxRange'])this.uniforms[name]=gl.getUniformLocation(this.program,name);
      gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
      const position=gl.getAttribLocation(this.program,'position'),rgb=gl.getAttribLocation(this.program,'rgb');
      gl.enableVertexAttribArray(position);gl.vertexAttribPointer(position,3,gl.FLOAT,false,16,0);
      gl.enableVertexAttribArray(rgb);gl.vertexAttribPointer(rgb,3,gl.UNSIGNED_BYTE,true,16,12);
      gl.enable(gl.DEPTH_TEST);
    }
    start() {
      if(this.active)return;
      if(!this.available){this.status('3D renderer unavailable: '+this.error);return;}
      this.active=true;const generation=++this.generation;
      this.started=performance.now();this.frames=0;this.previousFrame=null;
      this.status('Connecting to the RGB-D point cloud…');this.poll(generation);
    }
    stop() {
      this.active=false;this.generation++;
      clearTimeout(this.timer);if(this.request)this.request.abort();
      if(this.renderHandle)cancelAnimationFrame(this.renderHandle);this.renderPending=false;
    }
    async poll(generation) {
      if(!this.active||generation!==this.generation)return;
      const started=performance.now(),controller=new AbortController();this.request=controller;
      const timeout=setTimeout(()=>controller.abort(),3000);
      try {
        const response=await fetch('/api/camera/pointcloud',{signal:controller.signal,cache:'no-store'});
        if(!response.ok) {
          let message=`3D frame unavailable (${response.status})`;
          try {message=(await response.json()).error||message;}catch {}
          throw new Error(message);
        }
        const frame=decode(await response.arrayBuffer());
        if(!this.active||generation!==this.generation)return;
        if(frame.header.frame_id!==this.previousFrame) {
          this.previousFrame=frame.header.frame_id;this.frame=frame;this.frames++;
          if(!this.initialized){this.reset();this.initialized=true;}
          if(this.gl){this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.buffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,frame.data,this.gl.DYNAMIC_DRAW);}
          this.scheduleDraw();
        }
        const fps=this.frames/Math.max((performance.now()-this.started)/1000,1);
        const renderer=this.gl?'WebGL':'CPU canvas';
        this.status(`3D LIVE · ${frame.header.point_count.toLocaleString()} points · ${fps.toFixed(1)} FPS · ${renderer}`);
      } catch(error) {
        if(this.active&&generation===this.generation)this.status(error.name==='AbortError'?'3D frame timed out · retrying':error.message+' · retrying');
      } finally {
        clearTimeout(timeout);if(this.request===controller)this.request=null;
        if(this.active&&generation===this.generation)
          this.timer=setTimeout(()=>this.poll(generation),Math.max(50,250-(performance.now()-started)));
      }
    }
    reset() {
      const z=this.frame?.header.median_z_m;
      this.distance=Math.max(.5,Math.min(8,Number.isFinite(z)?z:2));
      this.target=[0,0,-this.distance];this.yaw=0;this.pitch=0;this.scheduleDraw();
    }
    setColor(mode) {this.colorMode=mode==='depth'?'depth':'rgb';this.scheduleDraw();}
    scheduleDraw() {
      if(!this.active||this.renderPending)return;
      this.renderPending=true;
      this.renderHandle=requestAnimationFrame(()=>{this.renderPending=false;this.draw();});
    }
    draw() {
      if(!this.frame||!this.active)return;
      const limit=this.gl?2:1;
      const ratio=Math.min(root.devicePixelRatio||1,limit);
      const width=Math.max(1,Math.round(this.canvas.clientWidth*ratio)),height=Math.max(1,Math.round(this.canvas.clientHeight*ratio));
      if(!this.canvas.clientWidth||!this.canvas.clientHeight)return;
      if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
      const m=matrix(this.target,this.distance,this.yaw,this.pitch,width/height);
      if(this.gl) {
        const gl=this.gl;gl.viewport(0,0,width,height);gl.clearColor(.035,.06,.045,1);
        gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
        gl.uniformMatrix4fv(this.uniforms.mvp,false,m);gl.uniform1f(this.uniforms.pointSize,2.5*ratio);
        gl.uniform1f(this.uniforms.depthMode,this.colorMode==='depth'?1:0);
        gl.uniform1f(this.uniforms.maxRange,this.frame.header.max_range_m);
        gl.drawArrays(gl.POINTS,0,this.frame.header.point_count);
      } else {
        const image=this.context.createImageData(width,height),pixels=image.data;
        for(let i=0;i<pixels.length;i+=4){pixels[i]=9;pixels[i+1]=15;pixels[i+2]=11;pixels[i+3]=255;}
        const depths=new Float32Array(width*height);depths.fill(Infinity);
        const data=this.frame.data,view=new DataView(data.buffer,data.byteOffset,data.byteLength);
        for(let i=0;i<this.frame.header.point_count;i++) {
          const offset=i*16,x=view.getFloat32(offset,true),y=-view.getFloat32(offset+4,true),z=-view.getFloat32(offset+8,true);
          const w=m[3]*x+m[7]*y+m[11]*z+m[15];if(w<=0)continue;
          const nx=(m[0]*x+m[4]*y+m[8]*z+m[12])/w,ny=(m[1]*x+m[5]*y+m[9]*z+m[13])/w;
          const nz=(m[2]*x+m[6]*y+m[10]*z+m[14])/w;
          if(Math.abs(nx)>1||Math.abs(ny)>1||nz < -1||nz>1)continue;
          const px=Math.min(width-1,Math.floor((nx+1)*.5*width)),py=Math.min(height-1,Math.floor((1-ny)*.5*height));
          const index=py*width+px;if(nz>=depths[index])continue;depths[index]=nz;
          const c=this.colorMode==='depth'?depthColor(-z/this.frame.header.max_range_m):data.subarray(offset+12,offset+15);
          for(let channel=0;channel<3;channel++)pixels[index*4+channel]=c[channel];
        }
        this.context.putImageData(image,0,0);
      }
    }
  }
  root.RGBPointCloud={Viewer,decode,matrix,multiply};
  if(typeof module!=='undefined'&&module.exports)module.exports=root.RGBPointCloud;
})(typeof window!=='undefined'?window:globalThis);
