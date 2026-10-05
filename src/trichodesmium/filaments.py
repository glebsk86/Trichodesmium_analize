"""Associate tangent-compatible fragments without inventing tissue pixels."""
from dataclasses import replace
import numpy as np
from scipy.ndimage import distance_transform_edt
from .tubular import TubeSettings, extract_spines, sample


def associate_labels(labels,settings=TubeSettings()):
    tips=[]
    for ident in np.unique(labels):
        if ident<=0:continue
        yy,xx=np.nonzero(labels==ident)
        y0,x0=max(0,int(yy.min())-1),max(0,int(xx.min())-1)
        y1,x1=min(labels.shape[0],int(yy.max())+2),min(labels.shape[1],int(xx.max())+2)
        mask=labels[y0:y1,x0:x1]==ident
        paths,_,_=extract_spines(mask,replace(settings,min_spine_length_px=25.))
        distance=distance_transform_edt(mask).astype(np.float32)
        for path in paths:
            if len(path)<6:continue
            width=max(2.,float(np.median(sample(distance,path)))*2)
            for points in (path,path[::-1]):
                outward=points[0]-points[5];outward/=max(np.linalg.norm(outward),1e-9)
                tips.append((int(ident),points[0]+[x0,y0],outward,width))
    options=[]
    for i,(ia,a,ua,wa) in enumerate(tips):
        for j in range(i+1,len(tips)):
            ib,b,ub,wb=tips[j]
            if ia==ib:continue
            delta=b-a;gap=float(np.linalg.norm(delta))
            if not 2<gap<=settings.max_gap_px or min(wa,wb)/max(wa,wb)<.65:continue
            direction=delta/gap
            alignment=min(float(ua@direction),float(ub@-direction),float(ua@-ub))
            if alignment<.90:continue
            options.append((gap/(alignment**4),i,j,gap))
    # A tip with two nearly equal continuations is an unresolved junction.
    choices={i:sorted((o[0],o[2] if o[1]==i else o[1]) for o in options if i in o[1:3])
             for i in range(len(tips))}
    parent={int(i):int(i) for i in np.unique(labels) if i>0}
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    used=set();links=[]
    for score,i,j,gap in sorted(options):
        if i in used or j in used:continue
        if any(len(choices[k])>1 and choices[k][1][0]<choices[k][0][0]*1.20 for k in (i,j)):continue
        ia,ib=root(tips[i][0]),root(tips[j][0])
        if ia==ib:continue
        parent[max(ia,ib)]=min(ia,ib);used.update((i,j))
        links.append(dict(source_ids=[tips[i][0],tips[j][0]],gap_px=gap,
                          endpoints_xy=[tips[i][1].tolist(),tips[j][1].tolist()],
                          notice='Tangent-compatible association; hidden bridge is an estimate.'))
    result=labels.copy()
    for i in parent:result[labels==i]=root(i)
    return result,links
