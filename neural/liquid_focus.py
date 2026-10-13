"""Approximate byte sensitivity transported to actual visual patches."""
import numpy as np
import torch


def rectangles(heat):
    if heat.ndim!=2 or not heat.size or not np.isfinite(heat).all():return []
    spread=float(heat.max()-heat.min())
    if float(heat.max())<=0 or spread<=max(1e-12,float(abs(heat.max()))*1e-6):return []
    height,width=heat.shape
    threshold=max(float(np.quantile(heat,.9)),float(heat.min())+spread*1e-6)
    active=heat>=threshold;seen=set();components=[]
    for y,x in zip(*np.where(active)):
        if (y,x) in seen:continue
        stack=[(y,x)];seen.add((y,x));points=[]
        while stack:
            cy,cx=stack.pop();points.append((cy,cx))
            for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                point=(cy+dy,cx+dx)
                if 0<=point[0]<height and 0<=point[1]<width and active[point] and point not in seen:
                    seen.add(point);stack.append(point)
        components.append(points)
    components.sort(key=lambda points:sum(float(heat[y,x]) for y,x in points),reverse=True)
    return [{'x':min(x for y,x in points)/width,'y':min(y for y,x in points)/height,
             'width':(max(x for y,x in points)-min(x for y,x in points)+1)/width,
             'height':(max(y for y,x in points)-min(y for y,x in points)+1)/height} for points in components[:3]]


def measure_focus(core,previous,tokens,interval,code,patch_attention):
    with torch.inference_mode(False),torch.enable_grad():
        inputs=tokens.detach().cpu().float().clone().requires_grad_(True)
        state=core.observe(previous.detach().float().clone(),inputs,interval)
        objective=core.logits(state)[0,code]
        gradient=torch.autograd.grad(objective,inputs)[0][2:66]
    weights,grid=patch_attention
    query=gradient.square().sum(-1).sqrt().detach().numpy()
    query/=max(float(query.sum()),1e-20)
    heat=query@weights.detach().float().cpu().numpy()[0]
    return rectangles(heat.reshape(tuple(map(int,grid))))
