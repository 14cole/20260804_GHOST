from common import *
import numpy as np
from scipy.integrate import dblquad
from ghost_backend.twod.polynomial_quadrature import log_moments
from ghost_backend.twod.basis import values,derivative_matrix
from ghost_backend.twod import operators as ops

# 1. log_moments against numeric double integration (degree 1 basis)
d=1
num=np.zeros((d+1,d+1))
for i in range(d+1):
    for j in range(d+1):
        f=lambda y,x: values(np.array([x]),d)[0,i]*values(np.array([y]),d)[0,j]*np.log(abs(x-y)+1e-300)
        num[i,j]=dblquad(f,0,1,0,1,epsabs=1e-12)[0]
print('log_moments(1) analytic\n',log_moments(1))
print('numeric\n',num,'  max diff %.2e'%np.max(abs(num-log_moments(1))))

# 2. closed-form self single-layer block vs adaptive quadrature
from ghost_backend.twod.geometry import LinearMesh,LinearNode,LinearElement
import inspect
print(inspect.signature(LinearElement.__init__) if hasattr(LinearElement,'__init__') else '')
