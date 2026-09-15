import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT)]
import numpy as np
from scipy.integrate import quad,quad_vec
from ghost_backend.twod.pulse.coefficients import green,_self_integral

k=2*np.pi*1e9/299792458.0            # 1 GHz
lengths=np.linspace(1e-3,3e-3,400)   # 400 distinct panel lengths -> 400 cache misses

def old(k,length):
    value,_=quad_vec(lambda t:green(k,np.array([length*t]),False)[0][0],
                     0.,.5,epsabs=2e-13,epsrel=2e-12)
    return 2*length*value

def new(k,length):
    def component(part):
        return quad(lambda t:part(green(k,np.array([length*t]),False)[0][0]),
                    0.,.5,epsabs=2e-13,epsrel=2e-12,limit=200)[0]
    return 2*length*complex(component(lambda z:z.real),component(lambda z:z.imag))

for name,fn in (('quad_vec (old)',old),('quad real+imag (new)',new)):
    fn(k,1e-3)
    t=time.perf_counter()
    values=[fn(complex(k),float(L)) for L in lengths]
    print('%-22s %6.3f s for %d distinct lengths  (%.2f ms each)'%(
        name,time.perf_counter()-t,len(lengths),1e3*(time.perf_counter()-t)/len(lengths)))
    if name.startswith('quad_vec'):reference=values
print('max relative difference: %.2e'%max(abs(a-b)/abs(a) for a,b in zip(reference,values)))

# how many integrand evaluations does each need?
calls=[0]
def counted(t,length=1e-3):
    calls[0]+=1
    return green(complex(k),np.array([length*t]),False)[0][0]
quad_vec(counted,0.,.5,epsabs=2e-13,epsrel=2e-12);a=calls[0];calls[0]=0
quad(lambda t:counted(t).real,0.,.5,epsabs=2e-13,epsrel=2e-12,limit=200)
quad(lambda t:counted(t).imag,0.,.5,epsabs=2e-13,epsrel=2e-12,limit=200)
print('integrand evaluations: quad_vec %d, quad real+imag %d'%(a,calls[0]))
