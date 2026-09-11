"""Capture actual assembled systems for repeated algebra experiments.

Disk writes and extra reference solves invalidate capture timing as a solver
benchmark. Use audit_profile.py for the current public solve timing instead.
"""
from common import *
import argparse
import inspect
import pickle
from unittest.mock import patch
import scipy.linalg as la
from scipy.sparse import coo_matrix, save_npz
import boundary_fields

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind')
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--frequency', type=float, default=.6)
    args = parser.parse_args()
    folder = HERE/'systems'/('{}-n{}-f{}'.format(args.kind,args.count,args.frequency))
    folder.mkdir(parents=True, exist_ok=True)
    original = boundary_fields.solve_fields
    signature = inspect.signature(original)
    count = [0]
    def capture(*a, **kw):
        bound = signature.bind(*a, **kw)
        bound.apply_defaults()
        p = bound.arguments
        index = count[0]; count[0] += 1
        prefix = folder/('TE' if index == 0 else 'TM')
        mesh, matrix, rhs = p['mesh'], p['matrix'], p['rhs_builder']
        n, d = len(mesh.nodes), len(matrix)
        b = rhs(np.asarray(p['angles']))
        xy = p['coordinates']
        if xy is None:
            xy = np.zeros((n, 2))
            for e in mesh.elements:
                xy[e.node_ids[0]], xy[e.node_ids[1]] = e.p0, e.p1
            xy = np.tile(xy, (d//n, 1))
        center = (xy.min(axis=0)+xy.max(axis=0))/2
        radius = np.linalg.norm(xy-center,axis=1).max()
        angular_order = int(np.ceil(abs(p['k0'])*radius + 28))
        sample_angles = np.arange(2*angular_order+1)*360/(2*angular_order+1)
        modes_rhs = rhs(sample_angles)
        for name, value in (('a',matrix),('b',b),('xy',xy),('angles',p['angles']),('harmonic_b',modes_rhs),('center',center)):
            np.save(str(prefix)+'-'+name+'.npy',value)
        x = la.solve(matrix,b,check_finite=False)
        np.save(str(prefix)+'-x.npy',x)
        rows, cols, values = [], [], []
        for start in range(0,d,64):
            stop = min(d,start+64)
            basis = np.zeros((d,stop-start))
            basis[np.arange(start,stop),np.arange(stop-start)] = 1
            mapped = basis[:n] if p['density_builder'] is None else p['density_builder'](basis)
            rr,cc = np.nonzero(mapped)
            rows.extend(rr);cols.extend(cc+start);values.extend(mapped[rr,cc])
        save_npz(str(prefix)+'-projection.npz',coo_matrix((values,(rows,cols)),shape=(n,d)).tocsr())
        with open(str(prefix)+'-mesh.pkl','wb') as handle:
            pickle.dump(mesh,handle)
        metadata = dict(label=p['label'],k0=p['k0'],n=d,nnodes=n,potential=p['potential'],
            second_potential=p['second_potential'],element_mask=None if p['element_mask'] is None else np.asarray(p['element_mask']).tolist(),
            angular_order=angular_order,center=center.tolist(),radius=radius,frequency=args.frequency,kind=args.kind)
        Path(str(prefix)+'-meta.json').write_text(json.dumps(metadata))
        print(prefix.name,d,'captured',flush=True)
        return original(*a,**kw)
    with patch.object(boundary_fields,'solve_fields',capture):
        result = public_solve(args.kind,args.count,args.frequency)
    (folder/'public-result.json').write_text(json.dumps(result))

if __name__ == '__main__':
    main()
