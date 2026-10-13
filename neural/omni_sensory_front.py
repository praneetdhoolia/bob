"""Native raw sensory embeddings without materializing the 8B language trunk.

The native codec and pretrained numerical encoders are retained. A lightweight
collector receives their embeddings; no ASR/caption/backbone decision runs.
"""
import importlib
import json
from pathlib import Path
import time
import types
import torch


class OmniSensoryFront:
    architecture='native-raw-sensory-embeddings-no-language-trunk-v1'

    def __init__(self,path='.runtime/minicpm-omni',chunk_ms=100,device='cuda'):
        from accelerate import init_empty_weights
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        path=Path(path);self.device=device;self.chunk_ms=chunk_ms
        config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True)
        config._attn_implementation='sdpa'
        config.init_vision=True;config.init_audio=True;config.init_tts=False
        cls=get_class_from_dynamic_module('modeling_minicpmo.MiniCPMO',str(path),local_files_only=True)
        # Only metadata is allocated for the language trunk. Materialize the
        # exact frontend and text embedding tensors from the pinned checkpoint.
        with init_empty_weights():
            self.model=cls(config)
        index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        components={'vpm':self.model.vpm,'resampler':self.model.resampler,
            'apm':self.model.apm,'audio_projection_layer':self.model.audio_projection_layer,
            'llm.model.embed_tokens':self.model.llm.get_input_embeddings()}
        for prefix,module in components.items():
            keys=[key for key in index if key.startswith(prefix+'.')]
            state={}
            for shard in sorted(set(index[key] for key in keys)):
                with safe_open(path/shard,framework='pt',device='cpu') as handle:
                    for key in keys:
                        if index[key]==shard:
                            state[key[len(prefix)+1:]]=handle.get_tensor(key).to(device=device,dtype=torch.bfloat16)
            module.load_state_dict(state,strict=True,assign=True)
            module.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
            del state
        self.rows=[]
        original=self.model.init_tts
        self.model.init_tts=types.MethodType(lambda self,*a,**k:None,self.model)
        try:self.duplex=self.model.as_duplex(generate_audio=False,chunk_ms=chunk_ms,
            first_chunk_ms=chunk_ms+35,cnn_redundancy_ms=20)
        finally:self.model.init_tts=original
        def collect(decoder,embeds,return_logits=False):
            self.rows.append(embeds.detach().reshape(-1,embeds.shape[-1]))
            return (None,None) if return_logits else None
        self.duplex.decoder.feed=types.MethodType(collect,self.duplex.decoder)
        self.reset()

    def reset(self):
        self.model.reset_session(reset_token2wav_cache=False)
        self.rows=[]
        self.duplex.prepare(prefix_system_prompt='Streaming Omni Conversation.',ref_audio=None,prompt_wav_path=None)
        self.initial_tokens=torch.cat(self.rows,0).detach()
        self.rows=[];self.steps=0;self.started=time.perf_counter()

    @torch.inference_mode()
    def step(self,audio=None,frames=None,texts=None):
        self.rows=[];started=time.perf_counter()
        prefill=self.duplex.streaming_prefill(audio_waveform=audio,frame_list=frames or [],
            text_list=texts or [],max_slice_nums=1,batch_vision_feed=True)
        if not prefill['success'] and texts:
            ids=self.duplex.tokenizer.encode(''.join(texts),add_special_tokens=False)
            if ids:self.duplex.decoder.feed(self.duplex.decoder.embed_token(torch.tensor(ids,device=self.device)))
        tokens=torch.cat(self.rows,0) if self.rows else torch.empty(0,self.model.embed_dim,device=self.device,dtype=torch.bfloat16)
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        self.steps+=1
        self.last={'prefill':prefill,'seconds':time.perf_counter()-started,'tokens':len(tokens),
            'audioSamples':0 if audio is None else len(audio),'frames':len(frames or []),'texts':len(texts or []),
            'audioEncoderCacheActive':self.model.audio_past_key_values is not None,
            'allocatedGpuBytes':torch.cuda.memory_allocated() if self.device.startswith('cuda') else 0,
            'elapsedPhysicalSeconds':time.perf_counter()-self.started,'step':self.steps}
        # No decoder generation is called. Keep schema bookkeeping bounded.
        self.duplex.prefill_schema_tokens=self.duplex.prefill_schema_tokens[-32:]
        return tokens,self.last

    def profile(self):
        real={};meta=0
        for parameter in self.model.parameters():
            if parameter.device.type=='meta':meta+=parameter.numel()
            else:
                key=str(parameter.device)+'/'+str(parameter.dtype)
                real[key]=real.get(key,0)+parameter.numel()*parameter.element_size()
        return {'materializedParameterBytes':real,'unmaterializedLanguageParameters':meta,
            'backboneForwardCalls':0,'embeddingDimension':self.model.embed_dim}
