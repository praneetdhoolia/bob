const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
// Run the served media-control block against fake browser devices. No real
// camera, microphone or audio output is touched.
const source=fs.readFileSync('public/agent.js','utf8');
const start=source.indexOf('// media-controls-begin'),end=source.indexOf('// media-controls-end');
assert.ok(start>=0&&end>start);

function environment(){
  const elements={},calls=[],bound=new Set();
  const element=id=>elements[id]||=Object.assign({id,attributes:{},hidden:true,paused:true,muted:false,
    setAttribute(name,value){this.attributes[name]=value;},removeAttribute(name){delete this[name];},
    pause(){this.paused=true;},load(){},async play(){this.paused=false;}});
  class Track{constructor(kind){this.kind=kind;this.stopped=false;}stop(){this.stopped=true;}}
  class Node{connect(next){return next||this;}disconnect(){this.disconnected=true;}}
  class AudioContext{
    constructor(){this.state='running';this.destination=new Node();this.audioWorklet={addModule:async()=>{}};contexts.push(this);}
    createGain(){return Object.assign(new Node(),{gain:{value:1}});}
    createMediaStreamSource(input){return Object.assign(new Node(),{input});}
    createMediaElementSource(media){
      if(bound.has(media))throw Object.assign(Error('already connected'),{name:'InvalidStateError'});
      bound.add(media);return new Node();
    }
    createBufferSource(){return Object.assign(new Node(),{start(){}});}
    decodeAudioData(){return new Promise(()=>{});}
    async resume(){this.state='running';}async suspend(){this.state='suspended';}async close(){this.state='closed';}
  }
  const contexts=[];
  const context={
    $:element,console,
    document:{createElement:()=>({getContext:()=>({drawImage(){}}),toDataURL:()=>'data:image/jpeg;base64,AA=='})},
    navigator:{mediaDevices:{getUserMedia:async request=>({request,tracks:[new Track(request.video?'video':'audio')],getTracks(){return this.tracks;}})}},
    AudioContext,AudioWorkletNode:class extends Node{constructor(){super();this.port={};}},
    URL:{createObjectURL:()=>'blob:test',revokeObjectURL(){}},
    setInterval:()=>1,clearInterval(){},btoa:()=>'',
    api:async(path,body)=>{calls.push([path,body]);return {};},notice(){},
    connected:true,state:{active:true,ready:true},
  };
  vm.createContext(context);
  vm.runInContext('let stream,audioContext,audioNode,audioSource,videoUrl,frameTimer;\n'+source.slice(start,end)+
    '\nthis.controls={camera,microphone,playVideoFile,playAudioFile,stopMedia,get audioKind(){return audioKind;},get stream(){return stream;}};',context);
  return {controls:context.controls,element,calls,contexts};
}

test('camera on and off leaves an active microphone running',async()=>{
  const {controls,element}=environment();
  await controls.microphone();
  assert.equal(controls.audioKind,'mic');
  await controls.camera();
  assert.equal(controls.audioKind,'mic');assert.equal(element('camera').attributes['aria-pressed'],'true');
  await controls.camera();
  assert.equal(controls.audioKind,'mic');assert.equal(element('microphone').attributes['aria-pressed'],'true');
});

test('a second local video reuses its single media-element audio source',async()=>{
  const {controls,contexts}=environment();
  await controls.playVideoFile({});
  await controls.playVideoFile({});
  assert.equal(controls.audioKind,'video');
  assert.equal(contexts.length,1);assert.equal(contexts[0].state,'running');
});

test('video soundtrack is unmuted for the graph and camera preview is muted',async()=>{
  const {controls,element}=environment();
  await controls.playVideoFile({});assert.equal(element('video').muted,false);
  await controls.camera();assert.equal(element('video').muted,true);
});

test('microphone during file playback starts the microphone instead of stopping',async()=>{
  const {controls,contexts}=environment();
  await controls.playAudioFile({arrayBuffer:async()=>new ArrayBuffer(8)});
  assert.equal(controls.audioKind,'file');
  await controls.microphone();
  assert.equal(controls.audioKind,'mic');assert.equal(contexts[0].state,'closed');
  await controls.microphone();
  assert.equal(controls.audioKind,null);
});
