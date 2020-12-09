import numpy as np 
from numpy import array
from scipy import linalg as LA

N = 6
V = (1.4, 1.2, 1.9)
rho = 1.

C = np.zeros([3, 3, 3, 3])
compress, shear, ax = 5., 1. , .5
for k in range(3):
    C[k,k,k,k] = compress
C[1,2,1,2]=C[2,1,1,2]=C[1,2,2,1]=shear
C[1,0,1,0]=C[0,1,1,0]=C[1,0,0,1]=shear
C[0,2,0,2]=C[2,0,0,2]=C[0,2,2,0]=shear
C[0,0,1,1]=C[1,1,0,0]=ax
C[0,0,2,2]=C[2,2,0,0]=ax
C[2,2,1,1]=C[1,1,2,2]=ax

basis = {}
idx = 0
for l in range(N+1):
    for m in range(N+1):
        for n in range(N+1):
            if l + m + n > N: continue
            for ii in range(3):
                basis[idx] = [l, m, n, ii]
                idx += 1
            
E = np.zeros([idx, idx])
G = np.zeros([idx, idx])

for i in range(idx):
    for j in range(i, idx):
        b1, b2 = basis[i], basis[j]
        pow_sum =array([b1[0]+b2[0], b1[1]+b2[1], b1[2]+b2[2]])
        vol_int = (V[0]**(pow_sum[0]+1) + V[1]**(pow_sum[1]+1) + V[2]**(pow_sum[2]+1))/((pow_sum[0]+1)*(pow_sum[1]+1)*(pow_sum[2]+1))
        if b1[-1]==b2[-1]: 
            E[i, j] = rho * vol_int
            if i != j: E[j,i] = E[i,j]
        tmp = 0
        for k in range(3):
            if pow_sum[k] == 1: tmp += 0
            else: tmp += b1[k]*b2[k] * (pow_sum[k] + 1) * C[b1[-1], k, b2[-1], k] / (V[k]**2 * (pow_sum[k] - 1))
            for l in range(3):
                if l == k or pow_sum[l]==0 or pow_sum[k]==0: pass
                else: tmp += b1[k]*b2[l]*(pow_sum[k]+1)*(pow_sum[l]+1)*C[b1[-1], k, b2[-1], l] / (V[k]*V[l]*pow_sum[k]*pow_sum[l])
        G[i, j] = vol_int * tmp 
        if i != j: G[j, i] = G[i,j]

w = LA.eigh(G, E, eigvals_only= True)

print (w)