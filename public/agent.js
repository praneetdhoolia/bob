const $ = id => document.getElementById(id);
let token, state, stream, audioContext, audioNode, audioSource, videoUrl, events, frameTimer;
let voice = false, utteranceBuffer = '', lastTickSample, connected = false;
const speechQueue=[];
let activeSpeech=null;
const outputRows = new Map();
let memoryTarget=[],memoryDisplay=[],orbitTarget=[],orbitDisplay=[];
let audioBarLevels=[];
const audioHistory=[],audioWindowSeconds=5;
let visualFrameAt=0,visualFramePending=false,audioCursor=0,audioNewestSeconds=0,audioNewestAt=0;
let previousProjection, fullMatrixReceived = false, numericalRate, observationRate, projectionChange;
let lastObservation;
const sensorStats = new Map();
const themeMedia = matchMedia('(prefers-color-scheme: dark)');
let themePreference = ['system','light','dark'].includes(localStorage.getItem('bob.theme')) ? localStorage.getItem('bob.theme') : 'system';
const paletteCache={};
const palette = name => paletteCache[name] ||= getComputedStyle(document.documentElement).getPropertyValue('--'+name).trim();
// Commit each historical value once. Animation advances its horizontal clock,
// rather than blending an old array index toward a different sample's height.
class FlowTrace {
  constructor(){this.samples=[];this.recent=[];this.seconds=8;this.scale=2;this.cursor=null;this.arrivedAt=0;this.bufferSeconds=.15;}
  append(value,time,now){
    if(!Number.isFinite(value)||!Number.isFinite(time))return;
    if(this.samples.length&&time<this.samples.at(-1).time){this.samples=[];this.recent=[];this.cursor=null;this.scale=2;}
    if(this.samples.length&&time===this.samples.at(-1).time)return;
    this.recent.push(Math.max(0,value));if(this.recent.length>3)this.recent.shift();
    const filtered=this.recent.reduce((sum,x)=>sum+x,0)/this.recent.length;
    this.samples.push(Object.freeze({time,value:filtered}));
    while(this.samples.length>2&&this.samples[1].time<time-this.seconds-.4)this.samples.shift();
    while(this.samples.length>256)this.samples.shift();
    if(this.cursor===null)this.cursor=time-this.bufferSeconds;
    this.arrivedAt=now;
    // Keep the vertical range steady. Expand only for an actual new high.
    const needed=filtered*1000*1.2;
    if(needed>this.scale){const magnitude=10**Math.floor(Math.log10(needed));this.scale=[1,2,5,10].find(step=>step*magnitude>=needed)*magnitude;}
  }
  advance(now,dt,streaming){
    if(!streaming||!this.samples.length)return false;
    const latest=this.samples.at(-1).time;
    const target=Math.min(latest,latest-this.bufferSeconds+Math.min(.25,Math.max(0,(now-this.arrivedAt)/1000)));
    const previous=this.cursor;
    if(Math.abs(target-this.cursor)>.4)this.cursor=target;
    else{const next=this.cursor+dt;this.cursor=Math.max(this.cursor,Math.min(latest,next+(target-next)*(1-Math.exp(-dt/.3))));}
    return this.cursor!==previous;
  }
  points(left,right,top,bottom){
    return this.samples.map(sample=>({time:sample.time,
      x:left+(sample.time-this.cursor+this.seconds)/this.seconds*(right-left),
      y:bottom-sample.value*1000/this.scale*(bottom-top)}));
  }
}
const flowTrace = new FlowTrace();
function plotCanvas(id) {
  const canvas=$(id),ctx=canvas.getContext('2d'),w=canvas.clientWidth||240,h=canvas.clientHeight||100;
  const dpr=Math.min(3,devicePixelRatio||1);
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return {canvas,ctx,w,h};
}
function smoothCurve(ctx,points){if(!points.length)return;ctx.moveTo(points[0][0],points[0][1]);for(let i=1;i<points.length-1;i++)ctx.quadraticCurveTo(points[i][0],points[i][1],(points[i][0]+points[i+1][0])/2,(points[i][1]+points[i+1][1])/2);if(points.length>1)ctx.lineTo(...points.at(-1));}
function drawOrbit(values) {
  orbitTarget=Array.from(values);
  if(!orbitDisplay.length)orbitDisplay=orbitTarget.slice();
  requestVisualFrame();
}
function paintOrbit(values) {
  if($('agent-tile').classList.contains('activity-off'))return;
  const {ctx,w,h}=plotCanvas('state-orbit');
  const cx=w/2,cy=h/2,r=Math.min(78,cy-4),scale=Math.max(.0001,...values.map(Math.abs));
  for(let i=0;i<values.length;i+=2){const angle=i/values.length*Math.PI*2,v=values[i]/scale;
    const radius=r*.86+v*r*.12,x=cx+Math.cos(angle)*radius,y=cy+Math.sin(angle)*radius;
    ctx.globalAlpha=.65;ctx.fillStyle=palette(v>0?'green':'blue');
    ctx.beginPath();ctx.arc(x,y,Math.max(.8,Math.abs(v)*2),0,Math.PI*2);ctx.fill();}
}

function chartArea(ctx,w,h,max,axisLabel) {
  const left=34,right=w-5,top=8,bottom=h-21;
  ctx.save();ctx.globalAlpha=1;ctx.lineWidth=1;ctx.font='10px "Segoe UI",sans-serif';
  ctx.textAlign='right';ctx.textBaseline='middle';
  const label=v=>v===0?'0':Number(v.toPrecision(2)).toLocaleString(undefined,{maximumSignificantDigits:2,useGrouping:false});
  for(let i=0;i<3;i++){
    const y=top+(bottom-top)*i/2;
    ctx.strokeStyle=palette('line');ctx.beginPath();ctx.moveTo(left,y+.5);ctx.lineTo(right,y+.5);ctx.stroke();
    ctx.fillStyle=palette('muted');ctx.fillText(label(max*(1-i/2)),left-5,y);
  }
  ctx.textBaseline='bottom';ctx.textAlign='center';ctx.fillText(axisLabel,(left+right)/2,h-2);ctx.restore();
  return {left,right,top,bottom};
}
function drawFlow() {
  const {ctx,w,h}=plotCanvas('flow-chart'),max=flowTrace.scale;
  const {left,right,top,bottom}=chartArea(ctx,w,h,max,'Recent activity');
  const points=flowTrace.points(left,right,top,bottom);
  if(!points.length)return;
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
  ctx.strokeStyle=palette('blue');ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);
  for(let i=1;i<points.length;i++){
    const before=points[i-1],after=points[i],third=(after.x-before.x)/3;
    // Each completed segment depends only on its two committed endpoints.
    ctx.bezierCurveTo(before.x+third,before.y,after.x-third,after.y,after.x,after.y);
  }
  const end=points.at(-1),edge=Math.max(right,end.x);
  if(end.x<right)ctx.lineTo(right,end.y);
  ctx.stroke();ctx.lineTo(edge,bottom);ctx.lineTo(points[0].x,bottom);ctx.closePath();
  const fill=ctx.createLinearGradient(0,top,0,bottom);fill.addColorStop(0,palette('flow-fill'));fill.addColorStop(1,'transparent');ctx.fillStyle=fill;ctx.fill();
  ctx.restore();
}

function startGalaxy() {
  const host=$('galaxy'),canvas=document.createElement('canvas');host.append(canvas);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const gl=canvas.getContext('webgl',{alpha:false,antialias:false,powerPreference:'low-power'});
  let phase=0,last=0,scheduled=false,ready=true;
  const schedule=()=>{if(!scheduled&&!document.hidden){scheduled=true;requestAnimationFrame(draw);}};
  const resize=()=>{const dpr=Math.min(1.25,devicePixelRatio||1);canvas.width=Math.round(innerWidth*dpr);canvas.height=Math.round(innerHeight*dpr);};
  resize();
  let seed=813;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const count=7000,data=new Float32Array(count*4);
  for(let i=0;i<count;i++){
    const category=random(),r=category<.2?Math.sqrt(random())*.25:Math.pow(random(),.8)*1.5;
    const angle=category<.65?random()*Math.PI*2:(i%2)*Math.PI-Math.log(Math.max(.16,r))*2.+(random()-.5)*(.6+r*.5);
    data.set([r,angle,1+random()*3.5,.03+random()*.18],i*4);
  }
  if(!gl){
    host.replaceChildren();const fallback=document.createElement('canvas');host.append(fallback);
    const context=fallback.getContext('2d');let previous=0,queued=false;
    const request=()=>{if(!queued&&!document.hidden){queued=true;requestAnimationFrame(frame);}};
    function frame(now){
      queued=false;if(document.hidden)return;
      if(previous&&!reduced.matches)phase+=Math.min(.1,(now-previous)/1000);previous=now;
      fallback.width=innerWidth;fallback.height=innerHeight;context.fillStyle=palette('base');context.fillRect(0,0,innerWidth,innerHeight);
      const scale=innerHeight/2;
      for(let i=0;i<count;i+=3){const [r,a,size,light]=data.subarray(i*4,i*4+4),angle=a+phase*.014;
        const x=innerWidth/2+Math.cos(angle)*r*scale,y=innerHeight/2+Math.sin(angle)*r*scale*.55;
        context.fillStyle=`rgba(230,227,220,${light})`;context.beginPath();context.arc(x,y,size*.5,0,Math.PI*2);context.fill();}
      host.dataset.simulationSeconds=phase.toFixed(3);host.dataset.renderer='canvas';
      if(!reduced.matches)request();
    }
    const refresh=()=>{previous=0;request();};
    addEventListener('resize',refresh);reduced.addEventListener('change',refresh);document.addEventListener('visibilitychange',refresh);refresh();return refresh;
  }
  function program(vertex,fragment){
    const shaders=[[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]].map(([kind,source])=>{
      const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));
      return shader;
    });
    const result=gl.createProgram();shaders.forEach(shader=>gl.attachShader(result,shader));gl.linkProgram(result);
    if(!gl.getProgramParameter(result,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(result));
    shaders.forEach(shader=>gl.deleteShader(shader));return result;
  }
  const disk=program(`
    attribute vec2 position;
    varying vec2 uv;
    void main(){uv=position*.5+.5;gl_Position=vec4(position,0.,1.);}
  `,`
    precision highp float;
    varying vec2 uv;
    uniform vec2 resolution;
    uniform float time;
    uniform float lightMode;
    float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float noise(vec2 p){
      vec2 cell=floor(p),f=fract(p);f=f*f*(3.-2.*f);
      return mix(mix(hash(cell),hash(cell+vec2(1.,0.)),f.x),mix(hash(cell+vec2(0.,1.)),hash(cell+1.),f.x),f.y);
    }
    float cloud(vec2 p){
      float sum=0.,weight=.5;
      for(int i=0;i<5;i++){sum+=noise(p)*weight;p=mat2(.8,-.6,.6,.8)*p*2.03+17.;weight*=.5;}
      return sum;
    }
    void main(){
      vec2 sky=(uv-.5)*vec2(resolution.x/resolution.y,1.)*2.;
      // A fixed camera views an inclined disk: no camera roll or full-screen spin.
      vec2 p=vec2(sky.x,sky.y/.55);
      float a=time*.014;
      p=mat2(cos(a),-sin(a),sin(a),cos(a))*p;
      float r=length(p);
      float winding=atan(p.y,p.x)+log(max(r,.16))*2.;
      float arm=pow(.5+.5*cos(winding*2.),5.);
      float haze=cloud(p*8.);
      float filaments=cloud(p*37.+5.);
      float density=exp(-r*1.5)*(.22+arm*.7)*(.15+haze*1.3);
      float dust=pow(cloud(p*19.+13.),3.)*(.2+arm)*exp(-r*1.1);
      float core=exp(-r*r*24.);
      vec3 nebula=mix(vec3(.47,.49,.54),vec3(.7,.64,.53),exp(-r*3.));
      vec3 light=nebula*density*1.25+vec3(.86,.8,.68)*core*1.05;
      light*=1.-dust*.75;
      light+=vec3(.33,.37,.44)*filaments*arm*exp(-r*2.)*.2;
      vec2 cells=sky*150.;vec2 cell=floor(cells),f=fract(cells)-.5;
      float star=pow(max(0.,1.-length(f)*3.),8.)*step(.985,hash(cell));
      vec3 dark=vec3(.065)+min(light,vec3(.43))+vec3(star*.32);
      vec3 pale=vec3(.5)+light*.38+vec3(star*.12);
      gl_FragColor=vec4(mix(dark,pale,lightMode),1.);
    }
  `);
  const stars=program(`
    attribute vec4 star;
    uniform vec2 resolution;
    uniform float time;
    varying float strength;
    varying float warmth;
    void main(){
      float angle=star.y+time*.014;
      vec2 p=vec2(cos(angle)*star.x,sin(angle)*star.x*.55);
      gl_Position=vec4(p.x*resolution.y/resolution.x,p.y,0.,1.);
      gl_PointSize=star.z*min(1.5,resolution.y/700.);
      strength=star.w;warmth=sin(star.y*7.)*.5+.5;
    }
  `,`
    precision mediump float;
    varying float strength;
    varying float warmth;
    uniform float lightMode;
    void main(){
      float radius=length(gl_PointCoord-.5)*2.;
      float glow=exp(-radius*radius*4.5)*(1.-smoothstep(.6,1.,radius));
      vec3 color=mix(vec3(.66,.73,.83),vec3(.86,.77,.64),warmth);
      gl_FragColor=vec4(color*glow*strength*(1.-lightMode*.65),1.);
    }
  `);
  const quad=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,quad);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  const particles=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,particles);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);
  function uniforms(p){
    gl.uniform2f(gl.getUniformLocation(p,'resolution'),canvas.width,canvas.height);
    gl.uniform1f(gl.getUniformLocation(p,'time'),phase);
    gl.uniform1f(gl.getUniformLocation(p,'lightMode'),document.documentElement.dataset.theme==='light'?1:0);
  }
  function draw(now){
    scheduled=false;
    if(document.hidden||!ready){last=now;return;}
    if(!reduced.matches&&last)phase+=Math.min(.1,(now-last)/1000);
    last=now;host.dataset.simulationSeconds=phase.toFixed(3);host.dataset.renderer='webgl';
    gl.viewport(0,0,canvas.width,canvas.height);gl.disable(gl.BLEND);gl.useProgram(disk);uniforms(disk);
    gl.bindBuffer(gl.ARRAY_BUFFER,quad);const pos=gl.getAttribLocation(disk,'position');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE);gl.useProgram(stars);uniforms(stars);
    gl.bindBuffer(gl.ARRAY_BUFFER,particles);const attr=gl.getAttribLocation(stars,'star');gl.enableVertexAttribArray(attr);gl.vertexAttribPointer(attr,4,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.POINTS,0,count);
    if(!reduced.matches)schedule();
  }
  const refresh=()=>{last=0;resize();schedule();};
  addEventListener('resize',refresh);reduced.addEventListener('change',refresh);document.addEventListener('visibilitychange',refresh);
  canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();ready=false;});
  refresh();return refresh;
}
// A decorative GPU failure must not stop media, chat or status controls.
let refreshGalaxy=()=>{};
try{refreshGalaxy=startGalaxy();}catch{$('galaxy').replaceChildren();}
function scrollEdges(element,horizontal=false) {
  const update=()=>{
    if(horizontal){
      element.style.setProperty('--fade-left',element.scrollLeft>2?'12px':'0px');
      element.style.setProperty('--fade-right',element.scrollWidth-element.clientWidth-element.scrollLeft>2?'18px':'0px');
    }else{
      element.style.setProperty('--fade-top',element.scrollTop>2?'16px':'0px');
      element.style.setProperty('--fade-bottom',element.scrollHeight-element.clientHeight-element.scrollTop>2?'22px':'0px');
    }
  };
  element.addEventListener('scroll',update,{passive:true});
  new ResizeObserver(update).observe(element);
  new MutationObserver(update).observe(element,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden']});
  update();
}
scrollEdges($('state-panel'));scrollEdges($('messages'));
scrollEdges(document.querySelector('.sensor-wrap'),true);

function redrawPlots() {
  drawMemory();paintOrbit(orbitDisplay);drawFlow();drawAudio();requestVisualFrame();
}
function applyTheme() {
  document.documentElement.dataset.theme=themePreference==='system'?(themeMedia.matches?'dark':'light'):themePreference;
  for(const name in paletteCache)delete paletteCache[name];
  const name=themePreference[0].toUpperCase()+themePreference.slice(1), next={system:'Light',light:'Dark',dark:'System'}[themePreference];
  $('theme').querySelector('use').setAttribute('href','#i-'+{system:'desktop',light:'sun',dark:'moon'}[themePreference]);
  $('theme').setAttribute('aria-label',`Theme: ${name}. Switch to ${next}`);
  $('theme').title=`Theme: ${name}`;
  localStorage.setItem('bob.theme',themePreference);
  refreshGalaxy();redrawPlots();
}
$('theme').addEventListener('click',()=>{themePreference={system:'light',light:'dark',dark:'system'}[themePreference];applyTheme();});
themeMedia.addEventListener('change',()=>{if(themePreference==='system')applyTheme();});

function initializeResize(id, type) {
  const handle=$(id),root=document.documentElement;
  const storage='bob.'+type;
  const stored=Number(localStorage.getItem(storage));
  let value=type==='video-share'?(stored>=25&&stored<=65?stored:50):(stored>=260&&stored<=560?stored:320);
  function set(next) {
    if(type==='video-share'){
      const width=$('participant-resizer').parentElement.clientWidth;
      const lower=Math.max(25,220/width*100),upper=Math.min(65,(width-372)/width*100);
      value=Math.max(lower,Math.min(Math.max(lower,upper),next));
      root.style.setProperty('--video-share',value+'%');
    }else{
      value=Math.max(260,Math.min(560,next,Math.max(260,$('chat-resizer').parentElement.clientWidth-450)));
      root.style.setProperty('--chat-width',value+'px');
    }
    handle.setAttribute('aria-valuenow',Math.round(value));
    handle.setAttribute('aria-valuetext',type==='video-share'?Math.round(value)+' percent video width':Math.round(value)+' pixels chat width');
    localStorage.setItem(storage,String(value));
  }
  set(value);
  handle.addEventListener('pointerdown',event=>{
    if(event.button!==0)return;
    event.preventDefault();handle.setPointerCapture(event.pointerId);document.body.classList.add('resizing');
    const start=event.clientX,initial=value,width=handle.parentElement.clientWidth;
    const move=e=>set(type==='video-share'?initial+(e.clientX-start)/width*100:initial-(e.clientX-start));
    const end=()=>{document.body.classList.remove('resizing');handle.removeEventListener('pointermove',move);handle.removeEventListener('lostpointercapture',end);redrawPlots();};
    handle.addEventListener('pointermove',move);handle.addEventListener('lostpointercapture',end);
  });
  handle.addEventListener('keydown',event=>{
    const step=type==='video-share'?2:20;
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();
    const next=event.key==='Home'?Number(handle.getAttribute('aria-valuemin')):event.key==='End'?Number(handle.getAttribute('aria-valuemax')):value+(event.key==='ArrowRight'?1:-1)*step*(type==='video-share'?1:-1);
    set(next);redrawPlots();
  });
  return ()=>set(value);
}
const fitParticipants=initializeResize('participant-resizer','video-share');
initializeResize('chat-resizer','chat-width');
new ResizeObserver(()=>{if(innerWidth>1100)fitParticipants();redrawPlots();}).observe($('agent-tile'));
function elapsed(seconds) {
  const s=Math.max(0,Math.floor(seconds||0));
  return [Math.floor(s/3600),Math.floor(s/60)%60,s%60].map(x=>String(x).padStart(2,'0')).join(':');
}
const decimal=(digits=1,unit='')=>value=>value.toLocaleString(undefined,{minimumFractionDigits:digits,maximumFractionDigits:digits})+unit;
const count=value=>Math.round(value).toLocaleString();
const gib=value=>value/1024**3;
const sensorDefinitions=[
  ['Runtime','elapsed','Active time',v=>v.physicalSeconds,elapsed,false],
  ['Runtime','updates','Numerical updates',()=>numericalRate,decimal(1,' Hz')],
  ['Runtime','observations-rate','Observations',()=>observationRate,decimal(1,' Hz')],
  ['Runtime','observation-ms','Observation duration',v=>v.latestObservationMs,decimal(1,' ms')],
  ['Runtime','encoder-ms','Input encoding',v=>v.encoderMs,decimal(1,' ms')],
  ['Runtime','interval','Observation interval',v=>v.inputIntervalMs,decimal(1,' ms')],
  ['Runtime','late','Late observations · total',v=>v.deadlineMisses,count,false],
  ['Runtime','ticks','Numerical updates · total',v=>v.ticks,count,false],
  ['Runtime','observations','Observations · total',v=>v.observations,count,false],
  ['Inputs','frames','Video frames · received',v=>v.framesReceived,count,false],
  ['Inputs','superseded','Video frames · superseded',v=>v.supersededFrames,count,false],
  ['Inputs','rejected-frames','Video frames · rejected',v=>v.rejectedFrames,count,false],
  ['Inputs','video-age','Last video age',v=>v.videoAgeSeconds,decimal(2,' s')],
  ['Inputs','frame-queue','Video frames · queued',v=>v.queues?.frames,count],
  ['Inputs','audio-received','Audio samples · received',v=>v.audioSamplesReceived,count,false],
  ['Inputs','audio-rejected','Audio samples · rejected',v=>v.rejectedAudioSamples,count,false],
  ['Inputs','audio-queue','Audio samples · queued',v=>v.queues?.audioSamples,count],
  ['Inputs','text-queue','Text messages · queued',v=>v.queues?.text,count],
  ['Inputs','audio-rms','Audio amplitude · RMS',v=>v.audioRms,decimal(4)],
  ['Inputs','audio-db','Audio level',v=>v.audioRms===undefined?null:20*Math.log10(Math.max(.000001,v.audioRms)),decimal(1,' dBFS')],
  ['Inputs','text-rows','Text tokens · latest input',v=>v.latestModalities?.textRows,count],
  ['Inputs','rows','Encoded input rows',v=>v.encodedRows,count],
  ['Numerical state','rms','State amplitude · RMS',v=>v.matrixRms,decimal(4)],
  ['Numerical state','change','Projection change · RMS',()=>projectionChange,decimal(4)],
  ['Numerical state','range-low','Lowest coordinate',v=>v.matrixRange?.[0],decimal(4)],
  ['Numerical state','range-high','Highest coordinate',v=>v.matrixRange?.[1],decimal(4)],
  ['Numerical state','matrix-size','Retained state',v=>v.matrixBytes/1024,decimal(0,' KiB'),false],
  ['Numerical state','parameters','Learned parameters',v=>v.parameters,count,false],
  ['Memory and compute','ram-used','System RAM · used',v=>v.memory&&gib(v.memory.systemUsedBytes),decimal(2,' GiB')],
  ['Memory and compute','ram-free','System RAM · available',v=>v.memory&&gib(v.memory.systemAvailableBytes),decimal(2,' GiB')],
  ['Memory and compute','ram-total','System RAM · capacity',v=>v.memory&&gib(v.memory.systemTotalBytes),decimal(1,' GiB'),false],
  ['Memory and compute','ram-percent','System RAM · usage',v=>v.memory&&v.memory.systemUsedBytes/v.memory.systemTotalBytes*100,decimal(1,' %')],
  ['Memory and compute','agent-ram','Agent resident RAM',v=>v.memory&&gib(v.memory.agentRssBytes),decimal(2,' GiB')],
  ['Memory and compute','system-cpu','System CPU',v=>v.compute?.systemCpuPercent,decimal(1,' %')],
  ['Memory and compute','agent-cpu','Agent CPU · whole system',v=>v.compute?.agentCpuPercent,decimal(1,' %')],
  ['Memory and compute','gpu-allocated','Agent tensor VRAM',v=>v.gpuAllocatedBytes===undefined?null:gib(v.gpuAllocatedBytes),decimal(2,' GiB')],
  ['Memory and compute','gpu-reserved','Agent reserved VRAM',v=>v.gpuReservedBytes===undefined?null:gib(v.gpuReservedBytes),decimal(2,' GiB')],
  ['Memory and compute','gpu-total','GPU VRAM · capacity',v=>v.compute?.gpuTotalBytes?gib(v.compute.gpuTotalBytes):null,decimal(1,' GiB'),false],
  ['Output and maintenance','bytes','Emitted bytes · total',v=>v.emittedBytes,count,false],
  ['Output and maintenance','probability','Next symbol probability',v=>v.translationConfidence===undefined?null:v.translationConfidence*100,decimal(1,' %')],
  ['Output and maintenance','urge','Urge · share of onset threshold',v=>v.urge===undefined?null:v.urge/v.urgeThreshold*100,decimal(0,' %')],
  ['Output and maintenance','readiness','Readiness · non-blank probability',v=>v.readiness===undefined?null:v.readiness*100,decimal(1,' %')],
  ['Output and maintenance','surprise','Sensory surprise',v=>v.surprise,decimal(2,' σ')],
  ['Output and maintenance','releases','Stimulus releases · total',v=>v.urgeReleases,count,false],
  ['Output and maintenance','cache-resets','Encoder cache resets · total',v=>v.encoderCacheResets,count,false],
  ['Output and maintenance','training','Training update',v=>v.weightsUpdate,count,false],
];

const sensorCells=new Map(),sensorGroupRows=new Map(),sensorGroupButtons=new Map();
let collapsedGroups={};
try{collapsedGroups=JSON.parse(localStorage.getItem('bob.sensorGroups')||'{}');}catch{}
let sensorGroup;
function setSensorGroup(group,open){
  for(const row of sensorGroupRows.get(group)||[])row.hidden=!open;
  sensorGroupButtons.get(group).setAttribute('aria-expanded',String(open));
  collapsedGroups[group]=!open;localStorage.setItem('bob.sensorGroups',JSON.stringify(collapsedGroups));
}
for(const [group,id,label] of sensorDefinitions){
  if(group!==sensorGroup){
    const row=document.createElement('tr');row.className='sensor-group';const cell=document.createElement('th');cell.colSpan=5;
    const button=document.createElement('button');button.type='button';button.setAttribute('aria-expanded','true');
    const icon=document.createElementNS('http://www.w3.org/2000/svg','svg');icon.setAttribute('viewBox','0 0 24 24');icon.setAttribute('aria-hidden','true');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d','m5 9 7 7 7-7');icon.append(path);
    button.append(icon,document.createTextNode(group));button.addEventListener('click',()=>setSensorGroup(group,button.getAttribute('aria-expanded')!=='true'));
    cell.append(button);row.append(cell);$('sensor-rows').append(row);sensorGroup=group;
    sensorGroupRows.set(group,[]);sensorGroupButtons.set(group,button);
  }
  const row=document.createElement('tr');row.dataset.sensor=id;row.id='sensor-'+id;
  const labelCell=document.createElement('th');labelCell.scope='row';labelCell.textContent=label;labelCell.title=label;row.append(labelCell);
  const cells=Array.from({length:4},()=>{const cell=document.createElement('td');cell.textContent='—';row.append(cell);return cell;});
  sensorCells.set(id,cells);sensorGroupRows.get(group).push(row);$('sensor-rows').append(row);
}
for(const[group,rows]of sensorGroupRows){sensorGroupButtons.get(group).setAttribute('aria-controls',rows.map(row=>row.id).join(' '));setSensorGroup(group,!collapsedGroups[group]);}
function updateSensors(value, sample) {
  for(const [,id,,read,format,statistics=true] of sensorDefinitions){
    const number=read(value),cells=sensorCells.get(id);
    if(!Number.isFinite(number)){cells[0].textContent='—';continue;}
    cells[0].textContent=format(number);
    if(!statistics)continue;
    let summary=sensorStats.get(id);
    if(sample){summary=summary?{min:Math.min(summary.min,number),max:Math.max(summary.max,number),sum:summary.sum+number,n:summary.n+1}:{min:number,max:number,sum:number,n:1};sensorStats.set(id,summary);}
    if(summary)[summary.min,summary.max,summary.sum/summary.n].forEach((x,i)=>cells[i+1].textContent=format(x));
  }
}
for (const id of ['camera','microphone','speaker','agent-mute']) {
  const slash = document.createElementNS('http://www.w3.org/2000/svg','path');
  slash.setAttribute('class','slash');slash.setAttribute('d','m3 3 18 18');
  $(id).querySelector('svg').append(slash);
}

async function api(path, body, retry = true) {
  const response = await fetch('/api/' + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {'Content-Type':'application/json', 'X-Bob-Token':token || ''},
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  const value = await response.json();
  if (response.status === 403 && retry && path !== 'session') {
    token = (await api('session', undefined, false)).token;
    return api(path, body, false);
  }
  if (!response.ok) throw Error(value.error || `Request failed (${response.status})`);
  return value;
}

function notice(text = '') { $('error').textContent = text; $('error').hidden = !text; }
function setChat(open) {
  $('chat-panel').hidden = !open;
  $('chat-toggle').setAttribute('aria-expanded', String(open));
  $('chat-toggle').setAttribute('aria-label', open ? 'Close chat' : 'Open chat');
  document.body.classList.toggle('chat-open', open);
  $('chat-resizer').hidden=!open;
  requestAnimationFrame(()=>{fitParticipants();redrawPlots();});
  if (open) { $('unread').hidden = true; $('text').focus(); }
  else if ($('chat-panel').contains(document.activeElement) || document.activeElement === document.body) $('chat-toggle').focus();
}
function setActivity(open) {
  $('state-panel').hidden = !open;
  $('agent-tile').classList.toggle('activity-off', !open);
  $('state-toggle').setAttribute('aria-expanded', String(open));
  $('state-toggle').setAttribute('aria-label', open ? 'Hide activity' : 'Show activity');
  localStorage.setItem('bob.activity', String(open));
  requestAnimationFrame(redrawPlots);
  if(open){visualFrameAt=0;requestVisualFrame();}
}
function bubble(row) {
  $('placeholder')?.remove();
  const div = document.createElement('div');
  div.className = 'message ' + (row.source === 'user' ? 'user' : '');
  const label = document.createElement('small'), text = document.createElement('span');
  label.textContent = row.source === 'user' ? 'YOU' : 'BOB'; text.textContent = row.text;
  div.append(label, text); $('messages').append(div);
  while($('messages').children.length>100){
    const oldest=$('messages').firstElementChild;
    for(const [id,node] of outputRows)if(node.parentElement===oldest)outputRows.delete(id);
    oldest.remove();
  }
  $('messages').scrollTop = $('messages').scrollHeight;
  if ($('chat-panel').hidden) $('unread').hidden = false;
  return text;
}
function output(row) {
  let text = outputRows.get(row.id);
  if (!text) { text = bubble(row); outputRows.set(row.id, text); }
  text.textContent = row.text;
  $('messages').scrollTop = $('messages').scrollHeight;
  if (row.ended && row.text) $('announce').textContent = 'Bob: ' + row.text;
  if (voice && row.part) utteranceBuffer += row.part;
  if (voice && (row.ended || /[.!?]\s*$/.test(utteranceBuffer))) {
    speak(utteranceBuffer); utteranceBuffer = '';
  }
}
function speak(text) {
  if (!voice || !text.trim() || !window.speechSynthesis) return;
  speechQueue.push(text.slice(0,2048));
  while(speechQueue.length>6)speechQueue.shift();
  pumpSpeech();
}
function pumpSpeech() {
  if(activeSpeech||!voice||!speechQueue.length)return;
  const speech = new SpeechSynthesisUtterance(speechQueue.shift());
  activeSpeech=speech;
  speech.onstart = () => $('face').classList.add('speaking');
  speech.onend = speech.onerror = () => {
    if(activeSpeech!==speech)return;
    activeSpeech=null;$('face').classList.remove('speaking');pumpSpeech();
  };
  speechSynthesis.speak(speech);
}
function cancelSpeech(){speechQueue.length=0;activeSpeech=null;utteranceBuffer='';window.speechSynthesis?.cancel();$('face').classList.remove('speaking');}
function toggleVoice() {
  voice = !voice; utteranceBuffer = '';
  $('speaker').setAttribute('aria-pressed', String(voice));
  $('agent-mute').setAttribute('aria-pressed', String(!voice));
  $('agent-mute').title = voice ? 'Mute agent' : 'Unmute agent';
  if (!voice) cancelSpeech();
}
function fitFocus() {
  const stage = $('video').parentElement, media = !$('video').hidden ? $('video') : $('processed');
  const width = media.videoWidth || media.naturalWidth, height = media.videoHeight || media.naturalHeight;
  if (!width || !height) return;
  const scale = Math.max(stage.clientWidth / width, stage.clientHeight / height);
  Object.assign($('focus').style, {left:(stage.clientWidth-width*scale)/2+'px',
    top:(stage.clientHeight-height*scale)/2+'px', width:width*scale+'px', height:height*scale+'px'});
}
new ResizeObserver(fitFocus).observe($('video').parentElement);
$('video').addEventListener('loadedmetadata', fitFocus);
$('processed').addEventListener('load', fitFocus);



function matrix(values) {
  memoryTarget=[];
  for(let bin=0;bin<32;bin++){
    let square=0;const start=Math.floor(bin*values.length/32),end=Math.floor((bin+1)*values.length/32);
    for(let i=start;i<end;i++)square+=values[i]**2;
    memoryTarget.push(Math.sqrt(square/Math.max(1,end-start)));
  }
  if(!memoryDisplay.length)memoryDisplay=memoryTarget.slice();
  requestVisualFrame();
}
function drawMemory() {
  const {ctx,w,h}=plotCanvas('matrix'),energy=memoryDisplay,max=Math.max(.0001,...energy)*1.08;
  const {left,right,top,bottom}=chartArea(ctx,w,h,max,'Memory groups');
  const points=energy.map((v,i)=>[left+i/31*(right-left),bottom-v/max*(bottom-top)]);
  if(!points.length)return;
  ctx.beginPath();smoothCurve(ctx,points);ctx.lineTo(right,bottom);ctx.lineTo(left,bottom);ctx.closePath();
  const fill=ctx.createLinearGradient(0,top,0,bottom);fill.addColorStop(0,palette('memory-fill'));fill.addColorStop(1,'transparent');ctx.fillStyle=fill;ctx.fill();
  ctx.beginPath();smoothCurve(ctx,points);ctx.strokeStyle=palette('green');ctx.lineWidth=2;ctx.stroke();
}

function waveform(rms,envelope) {
  const time=state?.physicalSeconds||0,now=performance.now();
  if(audioHistory.length&&time<audioHistory.at(-1).time-.1)audioHistory.length=0;
  if(!audioHistory.length)audioCursor=time;
  if(!audioHistory.length||time>audioHistory.at(-1).time+.001){
    for(let half=0;half<2;half++){
      const part=envelope?.slice(half*20,(half+1)*20);
      const amplitude=part?.length===20?Math.sqrt(part.reduce((sum,x)=>sum+x*x,0)/20):rms||0;
      audioHistory.push({time:time-.05+half*.05,value:Math.max(0,Math.min(1,amplitude)),display:0});
    }
  }
  while(audioHistory.length&&audioHistory[0].time<time-audioWindowSeconds-.5)audioHistory.shift();
  audioNewestSeconds=time;audioNewestAt=now;requestVisualFrame();
}
function requestVisualFrame(){
  if(!visualFramePending&&!document.hidden&&!$('state-panel').hidden){
    visualFramePending=true;requestAnimationFrame(animateVisuals);
  }
}
function animateVisuals(now){
  visualFramePending=false;
  const dt=Math.min(.1,Math.max(.001,(now-visualFrameAt)/1000));visualFrameAt=now;
  const streaming=state?.active&&connected&&now-audioNewestAt<400;
  if(streaming){
    const target=audioNewestSeconds+Math.min(.25,(now-audioNewestAt)/1000);
    if(Math.abs(target-audioCursor)>.4)audioCursor=target;
    else{audioCursor+=dt;audioCursor+=(target-audioCursor)*(1-Math.exp(-dt/.3));}
  }
  let moving=false;
  for(const sample of audioHistory){
    sample.display+=(sample.value-sample.display)*(1-Math.exp(-dt/.06));
    if(Math.abs(sample.value-sample.display)>.00002)moving=true;
  }
  const interpolate=(display,target,tau=.07)=>{
    const blend=1-Math.exp(-dt/tau);
    while(display.length<target.length)display.push(display.at(-1)||0);
    display.length=target.length;
    for(let i=0;i<target.length;i++){
      display[i]+=(target[i]-display[i])*blend;
      if(Math.abs(target[i]-display[i])>.000001)moving=true;
    }
  };
  interpolate(memoryDisplay,memoryTarget);interpolate(orbitDisplay,orbitTarget);
  if(flowTrace.advance(now,dt,streaming))moving=true;
  drawMemory();drawFlow();paintOrbit(orbitDisplay);
  if(drawAudio(dt))moving=true;
  if(streaming||moving)requestVisualFrame();
}
function audioBars(left,right){
  const count=Math.ceil(Math.min(100,Math.max(32,Math.floor((right-left)/6)))/2),step=(right-left)/count;
  return Array.from({length:count},(_,i)=>({x:left+(i+.5)*step,width:step*.65,time:audioCursor-audioWindowSeconds+(i+.5)/count*audioWindowSeconds}));
}
function audioAmplitudeAt(time){
  if(!audioHistory.length||time<audioHistory[0].time)return 0;
  for(let i=1;i<audioHistory.length;i++){
    const next=audioHistory[i],previous=audioHistory[i-1];
    if(time<=next.time){
      const fraction=Math.max(0,Math.min(1,(time-previous.time)/Math.max(.000001,next.time-previous.time)));
      return previous.display+(next.display-previous.display)*fraction;
    }
  }
  return audioHistory.at(-1).display;
}
function drawAudio(dt=0) {
  const {canvas,ctx,w,h}=plotCanvas('audio-wave'),left=42,right=w-7,top=8,bottom=h-8,centre=(top+bottom)/2;
  canvas.dataset.windowSeconds=String(audioWindowSeconds);
  const level=rms=>Math.max(0,Math.min(1,(20*Math.log10(Math.max(.000001,rms))+60)/60));
  const amplitude=(bottom-top)/2;
  ctx.font='10px "Segoe UI",sans-serif';ctx.fillStyle=palette('muted');ctx.textAlign='right';ctx.textBaseline='middle';
  for(const[db,y]of [[0,top],[-30,(top+centre)/2],[-60,centre]]){
    ctx.fillText(String(db),left-6,y);ctx.strokeStyle=palette('line');ctx.beginPath();ctx.moveTo(left,y+.5);ctx.lineTo(right,y+.5);ctx.stroke();
  }
  ctx.save();ctx.beginPath();ctx.rect(left,top-1,right-left,bottom-top+2);ctx.clip();ctx.fillStyle=palette('red');
  const bars=audioBars(left,right);
  if(audioBarLevels.length!==bars.length){
    const previous=audioBarLevels;
    audioBarLevels=bars.map((_,i)=>previous[Math.round(i/Math.max(1,bars.length-1)*(previous.length-1))]||0);
  }
  let moving=false;
  for(let i=0;i<bars.length;i++){
    const bar=bars[i],target=level(audioAmplitudeAt(bar.time)),tau=target>audioBarLevels[i] ? .12 : .22;
    audioBarLevels[i]+=(target-audioBarLevels[i])*(1-Math.exp(-dt/tau));
    if(Math.abs(target-audioBarLevels[i])>.0001)moving=true;
    const height=Math.max(1,audioBarLevels[i]*amplitude*2);
    ctx.beginPath();ctx.roundRect(bar.x-bar.width/2,centre-height/2,bar.width,height,Math.min(1.5,bar.width/2));ctx.fill();
  }
  ctx.restore();
  return moving;
}
document.addEventListener('visibilitychange',()=>{visualFrameAt=0;if(!document.hidden)requestVisualFrame();});
function render(value) {
  state = value;
  $('presence').textContent = value.ready ? (value.active ? 'Live' : 'Paused') : value.error ? 'Failed' : 'Loading';
  $('pause').setAttribute('aria-label', value.active ? 'Pause agent' : 'Resume agent');
  $('pause').title = value.active ? 'Pause agent' : 'Resume agent';
  const sample=value.observations!==lastObservation;
  if(sample && lastTickSample) {
    const seconds=value.physicalSeconds-lastTickSample.seconds;
    if(seconds>.05){numericalRate=Math.max(0,(value.ticks-lastTickSample.ticks)/seconds);observationRate=Math.max(0,(value.observations-lastTickSample.observations)/seconds);}
  }
  if(sample)lastTickSample={ticks:value.ticks,seconds:value.physicalSeconds,observations:value.observations};
  if (sample && value.matrixProjection) {
    if(!fullMatrixReceived)matrix(value.matrixProjection);
    drawOrbit(value.matrixProjection);
    const difference=previousProjection?Math.sqrt(value.matrixProjection.reduce((sum,x,i)=>sum+(x-previousProjection[i])**2,0)/value.matrixProjection.length):0;
    previousProjection=value.matrixProjection.slice();flowTrace.append(difference,value.physicalSeconds,performance.now());
    projectionChange=difference;requestVisualFrame();
  }
  if (value.memory?.systemTotalBytes) {
    const total = value.memory.systemTotalBytes / 1024**3, used = value.memory.systemUsedBytes / 1024**3;
    $('ram-label').textContent = `${used.toFixed(1)} / ${total.toFixed(1)} GiB`;
    $('ram-meter').max = total; $('ram-meter').value = used;
    $('ram-label').parentElement.title = `System RAM; agent ${(value.memory.agentRssBytes/1024**3).toFixed(2)} GiB`;
  }
  if(!value.active)waveform(0);else if(sample)waveform(value.audioRms || 0,value.audioEnvelope);
  updateSensors(value,sample);
  lastObservation=value.observations;
  if (value.error) notice(value.error);
  $('focus').replaceChildren();
  for (const box of value.focus || []) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
    for (const key of ['x','y','width','height']) rect.setAttribute(key, box[key]);
    $('focus').append(rect);
  }
  if (value.latestModalities?.videoFrames && !stream && !videoUrl) {
    $('processed').src = '/api/frame?t='+value.observations;
    $('processed').hidden = false; $('empty-stage').hidden = true;
  }
  $('stage-label').textContent = ({quiet:'Video input', recorded:'Test recording', camera:'Camera', video:'Recording'})[value.source] || 'Video input';
  $('video-status').textContent = value.videoAgeSeconds > 1 && !$('processed').hidden ? 'Last observed frame' : '';
}

// media-controls-begin
// Video (camera or file frames) and audio (microphone, audio file or the video
// file's soundtrack) are independent pipelines. Only one audio source feeds the
// agent at a time; changing video never silently stops the microphone.
let audioKind = null, elementAudio = null;
async function stopAudio() {
  if (audioNode) { audioNode.disconnect(); audioNode = null; }
  if (audioSource) { audioSource.disconnect(); audioSource = null; }
  // A media element can be bound to one Web Audio source for its lifetime, so
  // its context is suspended for reuse rather than closed.
  if (audioContext) { if (audioContext === elementAudio?.context) await audioContext.suspend(); else await audioContext.close(); audioContext = null; }
  audioKind = null;
  $('microphone').setAttribute('aria-pressed','false');
}
async function stopVideo() {
  clearInterval(frameTimer); frameTimer = null;
  if (audioKind === 'video') await stopAudio();
  if (stream) { stream.getTracks().forEach(track=>track.stop()); stream = null; }
  const video = $('video'); video.pause(); video.srcObject = null;
  video.removeAttribute('src'); video.load();
  if (videoUrl) { URL.revokeObjectURL(videoUrl); videoUrl = null; }
  video.hidden = true; $('camera').setAttribute('aria-pressed','false');
}
async function stopMedia() { await stopVideo(); await stopAudio(); }
async function rawSampler(source, kind, context = new AudioContext()) {
  audioContext = context; audioKind = kind;
  await audioContext.audioWorklet.addModule('/audio-worklet.js');
  audioNode = new AudioWorkletNode(audioContext,'liquid-raw-audio');
  audioNode.port.onmessage = async event => {
    if (!connected || !state?.active || !state?.ready) return;
    try {
      const bytes = new Uint8Array(event.data.buffer); let data = '';
      for (const byte of bytes) data += String.fromCharCode(byte);
      await api('audio', {sampleRate:16000, pcm:btoa(data)});
    } catch (error) { notice(error.message); }
  };
  const mute = audioContext.createGain(); mute.gain.value = 0;
  audioNode.connect(mute).connect(audioContext.destination);
  audioSource = source(audioContext); audioSource.connect(audioNode); await audioContext.resume();
}
function captureFrames() {
  const canvas = document.createElement('canvas'); canvas.width = 448;
  const context = canvas.getContext('2d'); let sending = false;
  frameTimer = setInterval(async()=>{
    const video = $('video'); if (sending || !connected || !state?.active || !state?.ready || !video.videoWidth || video.paused) return;
    sending = true;
    try {
      canvas.height = Math.round(canvas.width*video.videoHeight/video.videoWidth);
      context.drawImage(video,0,0,canvas.width,canvas.height);
      await api('frame',{image:canvas.toDataURL('image/jpeg',.8).split(',')[1]});
    } catch (error) { notice(error.message); } finally { sending = false; }
  },100);
}
async function camera() {
  if (stream) { await stopVideo(); await api('source',{source:'quiet'}); return; }
  await stopVideo();
  stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
  const video = $('video'); video.muted = true; video.srcObject = stream; video.hidden = false;
  $('processed').hidden = true; $('empty-stage').hidden = true;
  await video.play(); await api('source',{source:'camera'});
  $('camera').setAttribute('aria-pressed','true'); captureFrames();
}
async function microphone() {
  if (audioKind === 'mic') { await stopAudio(); return; }
  await stopAudio();
  const input = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
  await rawSampler(context=>{
    const source = context.createMediaStreamSource(input), disconnect = source.disconnect.bind(source);
    source.disconnect = ()=>{input.getTracks().forEach(track=>track.stop()); disconnect();};
    return source;
  }, 'mic');
  $('microphone').setAttribute('aria-pressed','true');
}
async function playVideoFile(file) {
  await stopMedia();
  const video = $('video');
  videoUrl = URL.createObjectURL(file); video.src = videoUrl; video.loop = true;
  // Unmuted so its soundtrack reaches the graph; the graph's zero gain keeps it silent.
  video.muted = false; video.hidden = false; $('processed').hidden = true; $('empty-stage').hidden = true;
  if (!elementAudio) { const context = new AudioContext(); elementAudio = {context, source:context.createMediaElementSource(video)}; }
  await rawSampler(()=>elementAudio.source, 'video', elementAudio.context);
  await video.play(); await api('source',{source:'video'}); captureFrames();
}
async function playAudioFile(file) {
  await stopAudio(); const data = await file.arrayBuffer();
  await rawSampler(context=>{
    const source = context.createBufferSource();
    context.decodeAudioData(data).then(buffer=>{source.buffer=buffer;source.start();})
      .catch(()=>{notice('This audio file could not be decoded.');if(audioSource===source)stopAudio();});
    source.onended = ()=>{ if (audioSource === source) stopAudio(); };
    return source;
  }, 'file');
}
// media-controls-end
function action(id, fn) {
  $(id).addEventListener('click', async()=>{
    const button = $(id); button.disabled = true;
    try { notice(); await fn(); }
    catch (error) {
      notice(error.name === 'NotAllowedError' ? 'Allow camera or microphone access in your browser.' :
        error.name === 'NotReadableError' ? 'The device is unavailable or in use by another app.' : error.message);
    } finally { button.disabled = false; }
  });
}
action('camera', camera); action('microphone', microphone);
action('speaker', toggleVoice); action('agent-mute', toggleVoice);
action('chat-toggle', ()=>setChat($('chat-panel').hidden)); action('chat-close', ()=>setChat(false));
action('state-toggle', ()=>setActivity($('state-panel').hidden)); action('state-close', ()=>setActivity(false));
action('more-toggle', ()=>{const open=$('more-menu').hidden; $('more-menu').hidden=!open; $('more-toggle').setAttribute('aria-expanded',String(open));});
action('recorded',async()=>{await stopMedia();await api('source',{source:'recorded'});$('more-menu').hidden=true;});
action('quiet',async()=>{await stopMedia();await api('source',{source:'quiet'});$('processed').hidden=true;$('empty-stage').hidden=false;$('more-menu').hidden=true;});
action('pause',async()=>{if(state?.active){await stopMedia();cancelSpeech();}await api('active',{active:!state?.active});render(await api('status'));});
$('video-file').addEventListener('change',async event=>{
  try { const file=event.target.files[0]; if(file) await playVideoFile(file); } catch(error){notice(error.message);}
});
$('audio-file').addEventListener('change',async event=>{
  try { const file=event.target.files[0]; if(file) await playAudioFile(file); } catch(error){notice(error.message);}
});
$('composer').addEventListener('submit',async event=>{
  event.preventDefault();const text=$('text').value;if(!text.trim())return;
  try{await api('message',{text});$('text').value='';}catch(error){notice(error.message);}
});
document.addEventListener('keydown',event=>{
  if(event.key!=='Escape')return;
  if(!$('more-menu').hidden){$('more-menu').hidden=true;$('more-toggle').setAttribute('aria-expanded','false');$('more-toggle').focus();}
  else if(!$('chat-panel').hidden)setChat(false);
});
document.addEventListener('click',event=>{if(!event.target.closest('.more-wrap'))$('more-menu').hidden=true;});
let connectDelay=500;
async function connect() {
  try {
    token=(await api('session')).token;const value=await api('status');render(value);connected=true;
    for(const row of value.messages||[])bubble(row);
    events=new EventSource('/api/events?after='+value.lastEvent);
    events.onmessage=event=>{
      connected=true;
      const row=JSON.parse(event.data);
      if(row.type==='sample'||row.type==='snapshot')render(row.data);
      else if(row.type==='message')bubble(row.data);
      else if(row.type==='output')output(row.data);
      else if(row.type==='error'||row.type==='input-error')notice(row.data.error);
    };
    events.onerror=()=>{
      connected=false;waveform(0);$('presence').textContent='Reconnecting';
      // EventSource stops for good after an HTTP error (for example 403 or 429).
      if(events.readyState===EventSource.CLOSED){events.close();setTimeout(connect,connectDelay);connectDelay=Math.min(connectDelay*2,5000);}
    };
    connectDelay=500;
  }catch(error){
    $('presence').textContent='Waiting for agent';
    setTimeout(connect,connectDelay);connectDelay=Math.min(connectDelay*2,5000);
  }
}
window.addEventListener('pagehide',()=>{
  clearInterval(frameTimer);if(stream)stream.getTracks().forEach(track=>track.stop());
  audioSource?.disconnect();audioContext?.close();events?.close();
});
setActivity(localStorage.getItem('bob.activity') !== 'false');
applyTheme();
connect();
setInterval(async()=>{if(!state?.ready||document.hidden||!connected)return;try{
  const value=await api('state');const bytes=Uint8Array.from(atob(value.matrixFloat32),x=>x.charCodeAt(0));
  matrix(new Float32Array(bytes.buffer));fullMatrixReceived=true;
}catch{}},1250);
