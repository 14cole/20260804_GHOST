"""Fixed piecewise-linear physical shapes; refinement never rounds corners."""
import numpy as np

CASES=('rectangle','reentrant','acute','gap','dielectric','mixed','sheet')


def segment(name,vertices,count,kind=2,ibc=0,pos=0,closed=True):
    v=np.asarray(vertices,float)
    if closed:
        area=np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
        if area > 0:v=v[::-1]
        v=np.vstack((v,v[0]))
    lengths=np.linalg.norm(np.diff(v,axis=0),axis=1)
    counts=np.maximum(1,np.floor(int(count)*lengths/lengths.sum()).astype(int))
    while counts.sum()<count:
        counts[np.argmax(lengths/counts)]+=1
    pairs=[]
    for a,b,n in zip(v[:-1],v[1:],counts):
        points=np.linspace(a,b,int(n)+1)
        pairs.extend(dict(x1=float(p[0]),y1=float(p[1]),x2=float(q[0]),y2=float(q[1]))
                     for p,q in zip(points[:-1],points[1:]))
    return dict(name=name,seg_type=kind,properties=list(map(str,(kind,1,ibc,pos,0))),point_pairs=pairs)


def fixture(case,count=192):
    rectangle=[[-.06,-.025],[-.06,.025],[.06,.025],[.06,-.025]]
    vertices=rectangle
    if case=='reentrant':
        vertices=[[-.06,-.04],[.06,-.04],[.06,-.018],[-.025,-.018],[-.025,.018],
                  [.06,.018],[.06,.04],[-.06,.04]]
    if case=='acute':vertices=[[-.07,0],[.065,-.023],[.044,.012],[-.012,.029]]
    if case not in CASES:raise ValueError(case)
    result=dict(segments=[],ibcs=[],dielectrics=[])
    if case=='gap':
        result['segments']=[segment('lower',np.asarray(rectangle)+[0,-.0255],count//2),
                            segment('upper',np.asarray(rectangle)+[.007,.0255],count-count//2)]
    elif case=='sheet':
        result['segments']=[segment('bent sheet',[[-.065,-.015],[0,0],[.05,.04]],count,1,ibc=1,closed=False)]
        result['ibcs']=[['1','constant','75','0','0','0']]
    elif case in ('dielectric','mixed'):
        result['segments']=[segment('dielectric',vertices,count if case=='dielectric' else count//2,3,pos=1)]
        result['dielectrics']=[['1','3','-.1','1','0']]
        if case=='mixed':result['segments'].append(segment('PEC',np.asarray(rectangle)*.5+[.11,.03],count-count//2))
    else:result['segments']=[segment(case,vertices,count)]
    return result
