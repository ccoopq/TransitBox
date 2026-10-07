const fs=require('fs'),path=require('path'),assert=require('assert');
let JSDOM;try{({JSDOM}=require('jsdom'))}catch(_error){({JSDOM}=require('../../BusProject/UI/bus-reid-ui/node_modules/jsdom'))}
const directory=path.resolve(process.argv[2]||path.resolve(__dirname,'..'));
const html=fs.readFileSync(path.join(directory,'index.html'),'utf8');
const dom=new JSDOM(html,{url:'http://127.0.0.1:8787/',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window,d=w.document;w.requestAnimationFrame=()=>1;w.setInterval=()=>1;w.matchMedia=()=>({matches:false});w.structuredClone=o=>JSON.parse(JSON.stringify(o));
w.HTMLElement.prototype.scrollIntoView=function(){};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false;this.dispatchEvent(new w.Event('close'))};
Object.defineProperties(w.HTMLMediaElement.prototype,{readyState:{get(){return this._loaded?4:0}},duration:{get(){return this.id==='clip-video'?4:16000}},paused:{get(){return this._paused!==false}},currentTime:{get(){return this._time||0},set(v){this._time=v}},playbackRate:{get(){return this._rate||1},set(v){this._rate=v}}});
w.HTMLMediaElement.prototype.load=function(){this._loaded=!!this.src;if(this.onloadedmetadata)this.onloadedmetadata();this.dispatchEvent(new w.Event('loadeddata'))};
w.HTMLMediaElement.prototype.pause=function(){this._paused=true;this.dispatchEvent(new w.Event('pause'))};
w.HTMLMediaElement.prototype.play=async function(){this._paused=false;this.dispatchEvent(new w.Event('play'))};
const hlsInstances=[];
class MockHls{
 static Events={MEDIA_ATTACHED:'attached',ERROR:'error'};
 static isSupported(){return true}
 constructor(config){this.config=config;this.events={};hlsInstances.push(this)}
 on(event,callback){this.events[event]=callback}
 attachMedia(video){this.video=video;video._loaded=true;this.events.attached();if(video.onloadedmetadata)video.onloadedmetadata();video.dispatchEvent(new w.Event('loadeddata'))}
 loadSource(src){this.src=src}
 destroy(){this.destroyed=true}
}
w.Hls=MockHls;w.HTMLMediaElement.prototype.canPlayType=()=>'';
const overlayLabels={};
w.HTMLCanvasElement.prototype.getBoundingClientRect=()=>({width:640,height:360});
w.HTMLCanvasElement.prototype.getContext=function(){const id=this.id;return {clearRect(){overlayLabels[id]=[]},setLineDash(){},strokeRect(){},fillRect(){},measureText(s){return {width:s.length*7}},fillText(s){overlayLabels[id].push(s)}}};
w.URL.createObjectURL=()=>'/fake-download';w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=function(){};
const embedded=JSON.parse(d.getElementById('transitbox-data').textContent);
const hasData=Object.keys(embedded).length>0;
const manifest=hasData?JSON.parse(fs.readFileSync(path.join(directory,'runtime/reid_results.json'),'utf8')):{results:{}};
const ready=manifest.status==='complete'&&manifest.backbone==='transreid';
let fetchCount=0;
w.fetch=async()=>{fetchCount++;return {ok:true,json:async()=>manifest}};
const js=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].pop()[1];
new Function(js);w.eval(js);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 if(!hasData){
  assert.strictEqual(JSON.stringify(w.TransitBox.getData()),'{}');
  assert.strictEqual(w.TransitBox.getSource(),null);
  assert.strictEqual(fetchCount,0,'Empty frontend must not request private result files');
  assert.strictEqual(hlsInstances.length,0);
  assert.strictEqual(d.querySelectorAll('video[src],video[poster],img').length,0);
  assert(d.getElementById('video-error').textContent.includes('not included'));
  assert(d.getElementById('play-button').disabled);
  assert(d.getElementById('export-button').disabled);
  w.TransitBox.selectSource('C3_1');w.TransitBox.seekTo(1);
  w.dispatchEvent(new w.Event('resize'));
  d.getElementById('about-button').click();assert(d.getElementById('about-dialog').open);
  assert.strictEqual(w.TransitBox.getSource(),null);
  console.log('PASS: empty public UI, no media sources or private-data requests, safe controls.');
  dom.window.close();return;
 }
 assert.strictEqual(w.TransitBox.getSource(),'C3_1');assert(d.querySelectorAll('#records tr[data-clip]').length>0);
 assert(!/[\u4e00-\u9fff]/.test(d.body.textContent.replace(d.getElementById('transitbox-data').textContent,'').replace(js,'')),'Product text must be English');
 if(ready)assert(d.getElementById('records').textContent.includes('P000'),'Real TransReID identity was not merged');
 else{
  assert(d.getElementById('reid-progress').textContent.includes('TransReID'));
  assert(d.getElementById('records').textContent.includes('Awaiting TransReID'));
  assert(!w.TransitBox.getData().C3_1.clips.some(c=>c.reid),'Old backbone identities must not appear as TransReID results');
 }
 w.TransitBox.seekTo(0);assert.strictEqual(d.getElementById('metric-clips').textContent,'0clips');
 const scope=d.getElementById('record-scope');scope.value='all';scope.dispatchEvent(new w.Event('change'));
 const filter=d.getElementById('payment-filter');filter.value='Evade';filter.dispatchEvent(new w.Event('change'));
 const data=w.TransitBox.getData();
 const expectedReviews=data.C3_1.clips.filter(c=>(c.payment||c.reidPayment)==='Evade').length;
 assert(d.getElementById('pagination-info').textContent.includes(`${expectedReviews} records`));
 d.querySelector('#records tr[data-clip]').click();assert(d.getElementById('detail-dialog').open);assert(d.getElementById('detail-fields').textContent.includes('ReID identity'));
 d.getElementById('detail-dialog').close();
 const candidateRecord=Object.entries(manifest.results).find(([key,r])=>key.startsWith('C3_1')&&r.candidates?.length);
 if(candidateRecord){
  filter.value='all';filter.dispatchEvent(new w.Event('change'));
  d.getElementById('search').value=candidateRecord[0];d.getElementById('search').dispatchEvent(new w.Event('input'));
  d.querySelector('#records tr[data-clip]').click();
  assert(d.querySelectorAll('.candidate-card').length>0);
  assert(d.getElementById('detail-fields').textContent.includes('Boarding payment'));
  assert(d.getElementById('detail-fields').textContent.includes('Gallery entries'));
  assert(d.getElementById('reid-detail-note').textContent.includes('Nearest onboard boarding identity'));
  d.querySelector('.candidate-card').click();assert(d.getElementById('detail-dialog').open);
  d.getElementById('detail-dialog').close();d.getElementById('search').value='';d.getElementById('search').dispatchEvent(new w.Event('input'));
 }
 // Playback must reconstruct gallery size without future exits or duplicate clips.
 filter.value='all';
 for(const source of ['C3_1','C3_3']){
  w.TransitBox.selectSource(source);
  const clips=data[source].clips;
  const entry=clips.find(c=>c.galleryAction==='added');
  if(entry){
   w.TransitBox.seekTo(entry.end-.001);
   const earlier=clips.filter(c=>c.end<entry.end);
   const expectedBefore=earlier.filter(c=>c.galleryAction==='added').length-earlier.filter(c=>c.galleryAction==='removed').length;
   assert.strictEqual(Number(d.getElementById('metric-reid').textContent),expectedBefore);
   w.TransitBox.seekTo(entry.end+.001);
   const elapsed=clips.filter(c=>c.end<=entry.end+.001);
   assert.strictEqual(Number(d.getElementById('metric-reid').textContent),elapsed.filter(c=>c.galleryAction==='added').length-elapsed.filter(c=>c.galleryAction==='removed').length);
  }
  w.TransitBox.seekTo(data[source].duration);
  if(ready)assert.strictEqual(Number(d.getElementById('metric-reid').textContent),manifest.onboardBySource[source]);
  else assert.strictEqual(d.getElementById('metric-reid').textContent,'—','Unavailable model must not present occupancy as measured');
  assert.strictEqual(Number(d.getElementById('metric-payment').textContent),clips.filter(c=>c.payment).length,'Inherited boarding payments must not inflate charts');
  scope.value='onboard';scope.dispatchEvent(new w.Event('change'));
  assert.strictEqual(d.getElementById('record-count').textContent,`${manifest.onboardBySource[source]} gallery entries`);
  const expectedRemaining=new Set(clips.filter(c=>c.galleryAction==='added').map(c=>c.reid));
  for(const c of clips.filter(c=>c.galleryAction==='removed'))expectedRemaining.delete(c.reid);
  for(const row of d.querySelectorAll('#records tr[data-clip]')){
   const entry=clips.find(c=>c.id===row.dataset.clip);assert(expectedRemaining.has(entry.reid),'Removed passenger leaked into the onboard list');
  }
  scope.value='all';scope.dispatchEvent(new w.Event('change'));
  w.TransitBox.seekTo(0);assert.strictEqual(d.getElementById('metric-reid').textContent,ready?'0':'—');
 }
 w.TransitBox.selectSource('C3_1');
 filter.value='all';
 const skipped=data.C3_1.clips.find(c=>c.reidStatus==='skipped_inside');
 if(skipped){d.getElementById('search').value=skipped.id;d.getElementById('search').dispatchEvent(new w.Event('input'));assert(d.getElementById('records').textContent.includes('Not added to gallery'));assert(!d.getElementById('records').textContent.includes('Processing'));d.getElementById('search').value='';d.getElementById('search').dispatchEvent(new w.Event('input'))}
 // On-frame identities and inherited payment appear only after the relevant clip ends.
 const frame=data.C3_1.tracking.find(f=>data.C3_1.clips.some(c=>(!ready||c.reid)&&c.stop===f[1]&&c.end<=f[0]&&(c.trackIds||[c.trackId]).some(id=>f[2].some(o=>o[0]===id))));
 assert(frame,'Expected a real frame after a completed identity event');
 w.TransitBox.seekTo(frame[0]+.001);
 if(ready)assert(overlayLabels['main-overlay'].some(label=>/P\d{4}/.test(label)),'Overlay must include the completed ReID identity');
 else assert(!overlayLabels['main-overlay'].some(label=>/P\d{4}/.test(label)),'Unavailable model must not show old or invented identities');
 w.TransitBox.selectSource('C3_3');assert.strictEqual(w.TransitBox.getSource(),'C3_3');
 d.getElementById('dual-button').click();assert(d.getElementById('video-stage').classList.contains('dual'));
 assert(!/[\u4e00-\u9fff]/.test(d.getElementById('detail-dialog').textContent));
 if(data.C3_1.video.endsWith('.m3u8')){assert(hlsInstances.length>=2);assert(hlsInstances.some(player=>player.destroyed),'Source switching must release old HLS players');assert(hlsInstances.some(player=>player.src===data.C3_3.video));assert(hlsInstances.some(player=>player.config.startPosition>0),'HLS must load at the saved video time')}
 console.log(ready?'PASS: TransReID results, causal gallery playback, payment inheritance, identity overlay, filters, candidate navigation and video controls.':'PASS: explicit TransReID weight-unavailable state, no old/invented identities, preserved payment data, track overlay, English interface and video controls.');
 dom.window.close();
})().catch(error=>{console.error(error);process.exitCode=1;dom.window.close()});
