const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
function capture(rate,frequency) {
  let Processor;const packets=[];
  const context={sampleRate:rate,Float32Array,Float64Array,Math,Number,
    AudioWorkletProcessor:class{constructor(){this.port={postMessage:value=>packets.push(Array.from(value))};}},
    registerProcessor:(_,value)=>{Processor=value;}};
  vm.runInNewContext(fs.readFileSync('public/audio-worklet.js','utf8'),context);
  const processor=new Processor();
  for(let start=0;start<rate;start+=128) {
    const input=Float32Array.from({length:Math.min(128,rate-start)},(_,i)=>Math.sin(2*Math.PI*frequency*(start+i)/rate));
    const output=new Float32Array(input.length).fill(1);processor.process([[input]],[[output]]);
    assert.ok(output.every(value=>value===0));
  }
  return packets.flat();
}
function rms(values){const tail=values.slice(1600);return Math.sqrt(tail.reduce((sum,x)=>sum+x*x,0)/tail.length);}
test('native 44.1/48 kHz both emit exactly ten complete 100 ms packets',()=>{
  for(const rate of [44100,48000])assert.equal(capture(rate,1000).length,16000);
});
test('speech passband preserved while 12 kHz alias is suppressed',()=>{
  const pass=rms(capture(48000,4000)),stop=rms(capture(48000,12000));
  assert.ok(Math.abs(pass-Math.SQRT1_2)<.02);assert.ok(stop/pass<.01);
});
