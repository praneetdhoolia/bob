// Band-limit before conversion, with a fractional clock for native 44.1 kHz.
class LiquidRawAudio extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer=new Float32Array(1600); this.used=0;
    this.ratio=sampleRate/16000; this.position=0; this.nextOutput=0;
    this.previous=0; this.ring=new Float64Array(63); this.head=0;
    this.taps=new Float64Array(63);
    const cutoff=.45*Math.min(sampleRate,16000)/sampleRate;
    let sum=0;
    for(let i=0;i<63;i++) {
      const offset=i-31;
      const sinc=offset===0?2*cutoff:Math.sin(2*Math.PI*cutoff*offset)/(Math.PI*offset);
      this.taps[i]=sinc*(.54-.46*Math.cos(2*Math.PI*i/62)); sum+=this.taps[i];
    }
    for(let i=0;i<63;i++)this.taps[i]/=sum;
  }
  process(inputs,outputs) {
    const input=inputs[0]?.[0];
    if(input)for(const sample of input) {
      this.ring[this.head]=Number.isFinite(sample)?sample:0;
      let filtered=0;
      for(let tap=0;tap<63;tap++)filtered+=this.taps[tap]*this.ring[(this.head-tap+63)%63];
      this.head=(this.head+1)%63;
      while(this.nextOutput<=this.position+1e-8) {
        const fraction=Math.max(0,Math.min(1,this.nextOutput-(this.position-1)));
        this.buffer[this.used++]=this.previous+(filtered-this.previous)*fraction;
        this.nextOutput+=this.ratio;
        if(this.used===1600){this.port.postMessage(this.buffer);this.buffer=new Float32Array(1600);this.used=0;}
      }
      this.previous=filtered;this.position++;
    }
    for(const output of outputs)for(const channel of output)channel.fill(0);
    return true;
  }
}
registerProcessor('liquid-raw-audio',LiquidRawAudio);
