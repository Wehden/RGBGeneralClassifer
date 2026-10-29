/* The image and table commit together, using one immutable scored snapshot. */
(function(root){
  const element=(tag,text,cls)=>{
    const node=document.createElement(tag);
    if(text!==undefined)node.textContent=text;
    if(cls)node.className=cls;
    return node;
  };
  function rowsTable(snapshot){
    const table=element('table'),head=element('thead'),header=element('tr');
    for(const text of ['Box','YOLOE object','Detection confidence','SigLIP2 material','Material score','Decision']){
      header.append(element('th',text));
    }
    head.append(header);table.append(head);
    const body=element('tbody');
    for(const row of snapshot.rows){
      const tr=element('tr'),m=row.material;
      const score=m.score_type==='supervised_material_head_score'
        ? `${m.score.toFixed(3)} · learned`
        : `${m.similarity.toFixed(3)} · similarity`;
      for(const text of [String(row.box_index+1),row.object.label,
          `${(row.object.confidence*100).toFixed(1)}%`,m.best_candidate,score,
          m.status==='identified'?`Identified: ${m.label}`:'Uncertain']){
        tr.append(element('td',text));
      }
      body.append(tr);
    }
    table.append(body);
    return table;
  }
  class Viewer{
    constructor(panel,content,status,options={}){
      this.panel=panel;this.content=content;this.status=status;
      this.fetch=options.fetch||root.fetch.bind(root);
      this.makeImage=options.makeImage||(()=>new Image());
      this.interval=options.interval||2000;
      this.generation=0;this.running=false;this.currentFrame=null;
      this.panel.ontoggle=()=>{if(this.panel.open)this.start();else this.stop();};
    }
    start(){
      if(this.running||!this.panel.open||this.panel.hidden)return;
      this.running=true;const generation=++this.generation;
      this.poll(generation);
    }
    stop(){
      this.running=false;this.generation++;clearTimeout(this.timer);
      this.abort?.abort();this.abort=null;
      if(this.pendingImage){
        this.cancelImage?.();
        this.cancelImage=null;
        this.pendingImage.onload=this.pendingImage.onerror=null;
        this.pendingImage.removeAttribute('src');this.pendingImage=null;
      }
    }
    select(enabled){
      this.panel.hidden=!enabled;this.stop();
      this.currentFrame=null;this.content.replaceChildren();
      this.status.textContent='Waiting for scored boxes…';
      if(enabled)this.start();
    }
    async poll(generation){
      const abort=new AbortController();this.abort=abort;
      // Bound requests so an unresponsive service cannot freeze polling.
      const deadline=setTimeout(()=>abort.abort(),5000);
      try{
        const response=await this.fetch('/api/camera/comparison',{signal:abort.signal,cache:'no-store'});
        if(!response.ok)throw new Error(`Request failed (${response.status})`);
        const snapshot=await response.json();
        if(!this.running||generation!==this.generation)return;
        if(!snapshot.rows?.length){
          this.content.replaceChildren();this.currentFrame=null;
          this.status.textContent=`Comparison ${snapshot.status} · waiting for scored boxes`;
        }else if(snapshot.frame_id===this.currentFrame){
          this.describe(snapshot);
        }else{
          await this.render(snapshot,generation);
        }
      }catch(error){
        if(this.running&&generation===this.generation)
          this.status.textContent=`Comparison unavailable · ${error.message}`;
      }finally{
        clearTimeout(deadline);
        if(this.running&&generation===this.generation)
          this.timer=setTimeout(()=>this.poll(generation),this.interval);
      }
    }
    describe(snapshot){
      this.status.textContent=`${snapshot.rows.length}/${snapshot.total_boxes} boxes scored · captured ${snapshot.age_s.toFixed(1)}s ago${snapshot.status==='stale'?' · stale snapshot':''}`;
    }
    async render(snapshot,generation){
      const image=this.makeImage();this.pendingImage=image;
      image.alt='RGB snapshot used by both model readings';
      // Only accept the local, frame-qualified image route.
      const expected='/api/camera/comparison/image.jpg?frame_id='+encodeURIComponent(snapshot.frame_id);
      if(snapshot.image_url!==expected)throw new Error('Unexpected comparison image');
      let deadline,loaded=false;
      try{
        await new Promise((resolve,reject)=>{
          this.cancelImage=()=>reject(new Error('Comparison stopped'));
          deadline=setTimeout(()=>reject(new Error('Snapshot image timed out')),5000);
          image.onload=()=>{loaded=true;resolve();};
          image.onerror=()=>reject(new Error('Snapshot image expired; refreshing'));
          image.src=expected;
        });
      }finally{
        clearTimeout(deadline);image.onload=image.onerror=null;
        if(this.pendingImage===image){this.pendingImage=null;this.cancelImage=null;}
        if(!loaded)image.removeAttribute('src');
      }
      if(!this.running||generation!==this.generation)return;
      const frame=element('div',undefined,'evidence-image comparison-image');frame.append(image);
      for(const row of snapshot.rows){
        const box=element('span',undefined,'box');
        box.style.top=`${row.box[0]*100}%`;box.style.left=`${row.box[1]*100}%`;
        box.style.height=`${(row.box[2]-row.box[0])*100}%`;box.style.width=`${(row.box[3]-row.box[1])*100}%`;
        box.append(element('span',String(row.box_index+1)));frame.append(box);
      }
      const table=element('div',undefined,'comparison-table');table.append(rowsTable(snapshot));
      this.content.replaceChildren(frame,table);
      this.currentFrame=snapshot.frame_id;
      this.describe(snapshot);
    }
  }
  root.RGBComparison={Viewer,rowsTable};
})(window);
