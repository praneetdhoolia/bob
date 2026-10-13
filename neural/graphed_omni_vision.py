"""Replay native encoder operations with fresh pixels and fixed geometry.

Cache executable graphs and geometry only, never image features. Native eager
outputs validate each graph before use. The single-camera path is supported;
other shapes beyond the bounded graph budget keep the native eager path.
"""
import time
import types
import torch


class GraphedOmniVision:
    def __init__(self,model,max_graphs=2,fast_attention=False):
        self.model=model;self.original=model.get_vision_embedding
        self.max_graphs=max_graphs;self.graphs={};self.events=[]
        self.fast_attention=fast_attention;self.last_patch_attention=None
        if fast_attention:
            for block in model.vpm.encoder.layers:
                attention=block.self_attn;original=attention.forward
                def fused(module,hidden_states,attention_mask=None,output_attentions=False,_original=original):
                    if output_attentions:return _original(hidden_states,attention_mask,output_attentions)
                    batch,length,_=hidden_states.shape
                    shape=(batch,length,module.num_heads,module.head_dim)
                    query=module.q_proj(hidden_states).view(shape).transpose(1,2)
                    key=module.k_proj(hidden_states).view(shape).transpose(1,2)
                    value=module.v_proj(hidden_states).view(shape).transpose(1,2)
                    output=torch.nn.functional.scaled_dot_product_attention(query,key,value,
                        attn_mask=attention_mask,dropout_p=0.,is_causal=False,scale=module.scale)
                    output=output.transpose(1,2).reshape(batch,length,module.embed_dim)
                    return module.out_proj(output),None
                attention.forward=types.MethodType(fused,attention)
        model.get_vision_embedding=types.MethodType(lambda _,data:self.encode(data),model)

    def pixels(self,data):
        values=[image.flatten(end_dim=1).permute(1,0) for image in data['pixel_values'][0]]
        padded=torch.nn.utils.rnn.pad_sequence(values,batch_first=True,padding_value=0.)
        batch,length,_=padded.shape
        return padded.permute(0,2,1).reshape(batch,3,-1,length).to(self.model.llm.model.embed_tokens.weight.dtype)

    @torch.inference_mode()
    def encode(self,data):
        self.last_patch_attention=None
        if len(data['pixel_values'])!=1 or len(data['pixel_values'][0])!=1:
            return self.original(data)
        sizes=torch.vstack(data['tgt_sizes']).detach().cpu().tolist()
        pixels=self.pixels(data)
        signature=(tuple(pixels.shape),tuple(tuple(size) for size in sizes))
        if signature not in self.graphs:
            if len(self.graphs)>=self.max_graphs:return self.original(data)
            self.graphs[signature]=self.capture(data,pixels,sizes)
        record=self.graphs[signature]
        record['input'].copy_(pixels)
        record['graph'].replay()
        self.last_patch_attention=(record['attention'],signature[1][0])
        # The surrounding native collector copies these output rows before
        # another replay. Do not keep this borrowed output as image memory.
        return [record['output']]

    @torch.inference_mode()
    def capture(self,data,pixels,sizes):
        started=time.perf_counter();positions=[]
        hook=self.model.vpm.embeddings.position_embedding.register_forward_pre_hook(
            lambda module,args:positions.append(args[0].detach().clone()))
        try:reference=self.original(data)[0].clone()
        finally:hook.remove()
        assert len(positions)==1
        h,w=map(int,sizes[0]);static=pixels.clone();position_ids=positions[0].clone()
        assert position_ids.shape[1]==h*w
        resampler=self.model.resampler;vpm=self.model.vpm
        pos=resampler.pos_embed[:h,:w].reshape(h*w,-1).to(static.dtype)[:,None].contiguous()
        mask=torch.zeros((1,h*w),device=static.device,dtype=torch.bool)
        def forward():
            patch=vpm.embeddings.patch_embedding(static)
            embedding=patch.flatten(2).transpose(1,2)+vpm.embeddings.position_embedding(position_ids)
            hidden=vpm.encoder(inputs_embeds=embedding,attention_mask=None,
                output_attentions=False,output_hidden_states=False,return_dict=True).last_hidden_state
            hidden=vpm.post_layernorm(hidden)
            value=resampler.ln_kv(resampler.kv_proj(hidden)).permute(1,0,2)
            query=resampler.ln_q(resampler.query).unsqueeze(1)
            out,attention=resampler.attn(query,value+pos,value,key_padding_mask=mask)
            return resampler.ln_post(out.permute(1,0,2))@resampler.proj,attention
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):test,_=forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        difference=float((test-reference).abs().max())
        cosine=float(torch.nn.functional.cosine_similarity(test.float().flatten(),reference.float().flatten(),dim=0))
        if cosine<.99999:raise RuntimeError(f'Fixed-geometry encoder differs from native reference: cosine={cosine}, max={difference}')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output,attention=forward()
        graph.replay();torch.cuda.synchronize()
        graph_cosine=float(torch.nn.functional.cosine_similarity(output.float().flatten(),reference.float().flatten(),dim=0))
        if graph_cosine<.99999:raise RuntimeError('Captured encoder differs from native reference')
        self.events.append({'shape':list(static.shape),'grid':[h,w],'setupSeconds':time.perf_counter()-started,
            'manualMaxDifference':difference,'manualCosine':cosine,'graphCosine':graph_cosine})
        # CUDA graphs keep addresses, not Python ownership. Every tensor
        # created outside capture must remain alive for every future replay.
        return {'input':static,'graph':graph,'output':output,'attention':attention,
                'positionIds':position_ids,'positionEmbedding':pos,'paddingMask':mask}
