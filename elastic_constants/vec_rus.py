import numpy as np
from scipy import linalg as LA
from time import time
t0 = time()
N = 1

class ElasticSolid:

    # instance attributes
    def __init__(self, elasticConstants, mass, dimensions, order):
        
        self.mass = mass # mass of the sample
        self.rho = mass / np.prod(dimensions)

        self.order = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.Lx = dimensions[0]/2 # sample dimensions divided by 2
        self.Ly = dimensions[1]/2 # they are divided by two because we will only integrate over half of the sample later
        self.Lz = dimensions[2]/2

        
        # initialize elastic tensor
        self.ctens = np.zeros([3, 3, 3, 3], dtype= np.double)
        self.ctens[0,0,0,0] = elasticConstants['c11']
        self.ctens[1,1,1,1] = elasticConstants['c22']
        self.ctens[2,2,2,2] = elasticConstants['c33']
        self.ctens[0,0,1,1] = ctens[1,1,0,0] = elasticConstants['c12']
        self.ctens[2,2,0,0] = ctens[0,0,2,2] = elasticConstants['c13']
        self.ctens[1,1,2,2] = ctens[2,2,1,1] = elasticConstants['c23']
        self.ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = elasticConstants['c44']
        self.ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = elasticConstants['c55']
        self.ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = elasticConstants['c66']

NT = int((N+1)*(N+2)*(N+3)/6)
m, xD, yD, zD = .045, .145, .201, .302
V = np.array([xD/2, yD/2, zD/2])
rho = m / (8*np.prod(V))
C = np.zeros([3, 3, 3, 3], dtype= np.double)
compress, shear, ax = 3.2161990, 1.2499627, 1.0350101
for k in range(3): C[k,k,k,k] = compress
for (i,j) in [(0,1), (0,2), (1,2)]: C[i,j,i,j]=C[j,i,j,i]=C[i,j,j,i]=C[j,i,i,j]=shear
for (i,j) in [(0,1), (0,2), (1,2)]: C[i,i,j,j]=C[j,j,i,i]=ax

lookUp = {(1, 1, 1) : 0, (1, 1, -1) : 1, (1, -1, 1) : 2, (-1, 1, 1) : 3, (1, -1, -1): 4, (-1, 1, -1) : 5, (-1, -1, 1) : 6, (-1, -1, -1) : 7}

basis, idx, block = np.zeros((NT, 3)), 0, [[],[],[],[],[],[],[],[]]
for k in range(N+1):
    for l in range(N+1):
        for m in range(N+1):
            if k+l+m > N: continue
            else:
                basis[idx] = np.array([k,l,m])
                for ii in range(3):
                    block[lookUp[tuple((-1,-1,-1)**(basis[idx] + np.roll([1,0,0], ii)))]].append(ii*NT + idx)
                idx += 1
E, I = np.zeros((3,idx,3,idx), dtype= np.double), np.zeros((3,idx,3,idx), dtype= np.double)
M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])

def E_int(i, j):
    ps = basis[i] + basis[j] + 1.
    if np.any(ps%2==0): return 0.
    return 8*rho *np.prod(V**ps / ps)
def G_int(i, j, k, l):
    if not basis[i][k]*basis[j][l]: return 0
    ps = basis[i] + basis[j] + 1. - M[k,l]
    if np.any(ps%2==0): return 0.
    return 8*basis[i][k]*basis[j][l]*np.prod(V**ps / ps)
for x in range(3*idx):
    i, k = x%3, x%idx
    for y in range(x, 3*idx):
        j, l = y%3, y%idx
        if i==j: E[i,k,j,l]=E[j,l,i,k]=E_int(k,l)
        I[i,k,j,l]=I[j,l,i,k]=G_int(k,l,i,j)
E = E.reshape(3*idx,3*idx)
print("MAKE EI: ", time() - t0)

t0 = time()
G = np.swapaxes(np.tensordot(C, I, axes= ([1,3],[0,2])), 2, 1).reshape(3*idx, 3*idx)
w = np.array([])
for ii in range(8): w = np.concatenate((w, LA.eigh(G[np.ix_(block[ii], block[ii])], E[np.ix_(block[ii], block[ii])], eigvals_only=True)))
print(np.sqrt(np.absolute(np.sort(w))[:30])/(2*np.pi))
print("EIG SOLVE: ", time() - t0)

for _ in range(10):
    C = np.zeros([3, 3, 3, 3], dtype= np.double)
    compress, shear, ax = np.random.random()+3, np.random.random()+1, np.random.random()+1
    for k in range(3): C[k,k,k,k] = compress
    for (i,j) in [(0,1), (0,2), (1,2)]: C[i,j,i,j]=C[j,i,j,i]=C[i,j,j,i]=C[j,i,i,j]=shear
    for (i,j) in [(0,1), (0,2), (1,2)]: C[i,i,j,j]=C[j,j,i,i]=ax
    t0 = time()
    G = np.swapaxes(np.tensordot(C, I, axes= ([1,3],[0,2])), 2, 1).reshape(3*idx, 3*idx)
    w = np.array([])
    for ii in range(8): w = np.concatenate((w, LA.eigh(G[np.ix_(block[ii], block[ii])], E[np.ix_(block[ii], block[ii])], eigvals_only=True)))
    print(np.sqrt(np.absolute(np.sort(w))[:30])/(2*np.pi))
    print("EIG SOLVE: ", time() - t0)



