const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
// Execute the actual pure graph model in the served script, without a DOM.
const source=fs.readFileSync('public/agent.js','utf8');
const start=source.indexOf('class FlowTrace {'),end=source.indexOf('\nconst flowTrace =',start);
assert.ok(start>=0&&end>start);
const FlowTrace=vm.runInNewContext(source.slice(start,end)+'\nFlowTrace;');
test('committed historical heights survive new samples, scrolling and pruning',()=>{
  const trace=new FlowTrace();
  for(let i=0;i<90;i++)trace.append(.001+.0004*Math.sin(i),i/10,i*100);
  const time=trace.samples[5].time,record=trace.samples[5];
  const before=trace.points(34,300,8,100).find(point=>point.time===time);
  for(let i=90;i<94;i++){
    trace.append(.001+.0004*Math.sin(i),i/10,i*100);
    for(let frame=0;frame<6;frame++)trace.advance(i*100+frame*16.7,1/60,true);
    const after=trace.points(34,300,8,100).find(point=>point.time===time);
    assert.ok(after);assert.equal(after.y,before.y);assert.ok(after.x<before.x);
    assert.strictEqual(trace.samples.find(point=>point.time===time),record);
  }
});
test('causal average is committed once and history stays bounded',()=>{
  const trace=new FlowTrace();
  for(let i=0;i<300;i++)trace.append(i%2*.001,i*.001,i);
  assert.ok(trace.samples.length<=256);assert.ok(Object.isFrozen(trace.samples[0]));
  const sample=trace.samples.at(-1),count=trace.samples.length;
  trace.append(1,sample.time,400);assert.equal(trace.samples.length,count);
  assert.strictEqual(trace.samples.at(-1),sample);
});
test('axis does not breathe when old peaks leave or rendering is paused',()=>{
  const trace=new FlowTrace();trace.append(.01,0,0);const scale=trace.scale;
  for(let i=1;i<200;i++)trace.append(.0001,i*.1,i*100);
  assert.equal(trace.scale,scale);
  const cursor=trace.cursor;trace.advance(25000,.1,false);assert.equal(trace.cursor,cursor);
  const time=trace.samples.at(-1).time;trace.append(NaN,30,30000);assert.equal(trace.samples.at(-1).time,time);
});
test('buffered live edge reveals known segments without extrapolating the signal',()=>{
  const trace=new FlowTrace();trace.append(.001,10,1000);trace.append(.0015,10.1,1100);
  trace.advance(1100,1/60,true);const before=trace.cursor;
  for(let frame=0;frame<20;frame++)trace.advance(1100+frame*16.7,1/60,true);
  assert.ok(trace.cursor>before);assert.ok(trace.cursor<=10.1);
  assert.ok(trace.points(0,800,0,100).at(-1).x>=800);
});
