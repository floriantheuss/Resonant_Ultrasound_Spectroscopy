import numpy as np
from scipy import linalg as LA
from time import time
t0 = time()
N = 1

class ElasticSolid:

    # instance attributes
    def __init__(self, initElasticConstants_dict, mass, dimensions, order, nb_freq, method='differential_evolution'):
        # elastic constants should be a dictionary
        # mass should just be a number
        # dimensions should be a numpy array
        # CrystalStructure should be a string
        
        self.mass       = mass # mass of the sample
        self.rho        = mass / np.prod(dimensions)

        self.order      = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N          = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.Vol         = dimensions/2 # sample dimensions divided by 2

        self.elasticConstants_dict = initElasticConstants_dict

        self.nb_freq = nb_freq

        
        self.method      = method # "shgo", "differential_evolution", "leastsq"
        self.pars        = Parameters()
        for param_name, param_range in self.elasticConstants_dict.items():
            self.pars.add(param_name, value = self.init_member[param_name], min = param_range[0], max = param_range[-1])
        



        # create basis
        lookUp = {(1, 1, 1) : 0, (1, 1, -1) : 1, (1, -1, 1) : 2, (-1, 1, 1) : 3, (1, -1, -1): 4, (-1, 1, -1) : 5, (-1, -1, 1) : 6, (-1, -1, -1) : 7}

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block = [[],[],[],[],[],[],[],[]]
        for k in range(self.order+1):
            for l in range(self.order+1):
                for m in range(self.order+1):
                    if k+l+m > self.order: continue
                    else:
                        self.basis[self.idx] = np.array([k,l,m])
                        for ii in range(3):
                            self.block[lookUp[tuple((-1,-1,-1)**(self.basis[self.idx] + np.roll([1,0,0], ii)))]].append(ii*self.N + self.idx)
                        self.idx += 1
    


            


    def elastic_tensor (self, pars):
        ctens = np.zeros([3,3,3,3])

        if len(pars) == 3:                      # cubic
            c11 = c22 = c33 = pars['c11']
            c12 = c13 = c23 = pars['c12']
            c44 = c44 = c66 = pars['c44']

        elif len(pars) == 5:                    # hexagonal
            c11 = c22       = pars['c11']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = (pars['c11']-pars['c12'])/2
        
        elif len(pars) == 6:                    # tetragonal
            c11 = c22       = pars['c11']
            c33             = pars['c33']
            c12             = pars['c12']
            c13 = c23       = pars['c13']
            c44 = c55       = pars['c44']
            c66             = pars['c66']
        
        elif len(pars) == 9:                    # orthorhombic
            c11             = pars['c11']
            c22             = pars['c22']
            c33             = pars['c33']
            c12             = pars['c12']
            c13             = pars['c13']
            c23             = pars['c23']
            c44             = pars['c44']
            c55             = pars['c55']
            c66             = pars['c66']
        
        else:
            print ('You have not given a valid Crystal Structure')

        ctens[0,0,0,0] = elasticConstants['c11']
        ctens[1,1,1,1] = elasticConstants['c22']
        ctens[2,2,2,2] = elasticConstants['c33']
        ctens[0,0,1,1] = ctens[1,1,0,0] = elasticConstants['c12']
        ctens[2,2,0,0] = ctens[0,0,2,2] = elasticConstants['c13']
        ctens[1,1,2,2] = ctens[2,2,1,1] = elasticConstants['c23']
        ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = elasticConstants['c44']
        ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = elasticConstants['c55']
        ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = elasticConstants['c66']

        return ctens



    def E_int (self, i, j):       # calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod(self.Vol**ps / ps)

    def G_int (self, i, j, k, l): # calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*basis[i][k]*basis[j][l]*np.prod(self.Vol**ps / ps)

    def E_mat (self):
        Etens = np.zeros((3,idx,3,idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = x%3, x%self.idx
            for y in range(x, 3*self.idx):
                j, l = y%3, y%self.idx
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)*self.rho
        
        Emat = Etens.reshape(3*idx,3*idx)
        return Emat

    def I_tens (self):
        Itens = np.zeros((3,idx,3,idx), dtype= np.double)
        for x in range(3*idx):
            i, k = x%3, x%idx
            for y in range(x, 3*idx):
                j, l = y%3, y%idx
                Itens[i,k,j,l]=Itens[j,l,i,k]=self.G_int(k,l,i,j)
        return Itens

    def G_mat (self, pars):
        C = self.elastic_tensor(pars)
        Gtens = np.tensordot(C, self.I_tens(), axes= ([1,3],[0,2]))
        Gmat = np.swapaxes(Gtens, 2, 1).reshape(3*self.idx, 3*self.idx)
        return Gmat

    def resonance_frequencies (self, pars):
        w = np.array([])
        for ii in range(8): 
            w = np.concatenate((w, LA.eigh(self.G_mat(pars)[np.ix_(self.block[ii], self.block[ii])], self.E_mat()[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
        f = np.sqrt(np.absolute(np.sort(w))[:self.nb_freq])/(2*np.pi)
        return f
