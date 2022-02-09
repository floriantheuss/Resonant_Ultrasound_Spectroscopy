import numpy as np
from scipy import linalg
from copy import deepcopy
from rus.elastic_constants import ElasticConstants
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSRPR(ElasticConstants):
    def __init__(self, cij_dict, symmetry,
                 mass, dimensions,
                 order,
                 nb_freq=1,
                 angle_x=0, angle_y=0, angle_z=0,
                 init=False, use_quadrants=True):
        """
        cij_dict: a dictionary of elastic constants in GPa
        mass: a number in kg
        dimensions: numpy array of x, y, z lengths in m
        order: integer - highest order polynomial used to express basis functions
        nb_freq: number of frequencies to display
        method: fitting method
        use_quadrants: if True, uses symmetry arguments of the elastic tensor to simplify and speed up eigenvalue solver;
                        only gives correct result if crystal symmetry is orthorhombic or higher;
                        if symmetry is e.g. rhombohedral, use use_quadrants=False
                        (use_quadrants=True ignores all terms c14, c15, c16, c24, c25, ... in the elastic tensor)
        """
        super().__init__(cij_dict,
                         symmetry=symmetry,
                         angle_x=angle_x, angle_y=angle_y, angle_z=angle_z)

        self.mass       = mass # mass of the sample
        self.dimensions = np.array(dimensions) # in meters
        self.density    = mass / np.prod(self.dimensions)


        self.order      = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N          = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.cij_dict = deepcopy(cij_dict)
        self._nb_freq = nb_freq
        self.freqs    = None

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]
        self.Emat  = None
        self.Itens = None

        self.use_quadrants = use_quadrants

        if init == True:
            self.initialize()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _get_nb_freq(self):
        return self._nb_freq
    def _set_nb_freq(self, nb_freq):
        self._nb_freq = nb_freq
    nb_freq = property(_get_nb_freq, _set_nb_freq)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def initialize(self):
        # create basis and sort it based on its parity;
        # for details see Arkady's paper;
        # this is done here in __init__ because we only need to is once and it is the "long" part of the calculation
        self.idx    =  0
        self.block  = [[],[],[],[],[],[],[],[]]
        self.Emat  = None
        self.Itens = None

        lookUp = {(1, 1, 1) : 0,
                   (1, 1, -1) : 1,
                    (1, -1, 1) : 2,
                     (-1, 1, 1) : 3,
                      (1, -1, -1): 4,
                       (-1, 1, -1) : 5,
                        (-1, -1, 1) : 6,
                         (-1, -1, -1) : 7}
        for k in range(self.order+1):
            for l in range(self.order+1):
                for m in range(self.order+1):
                    if k+l+m <= self.order:
                        self.basis[self.idx] = np.array([k,l,m])
                        for ii in range(3):
                            self.block[lookUp[tuple((-1,-1,-1)**(self.basis[self.idx] + np.roll([1,0,0], ii)))]].append(ii*self.N + self.idx)
                        self.idx += 1
        self.Emat = self.E_mat()
        self.Itens = self.I_tens()

    def copy_object(self, rpr_object):
        self.block = rpr_object.block
        self.idx   = rpr_object.idx
        self.Emat  = rpr_object.Emat
        self.Itens = rpr_object.Itens

    def E_int(self, i, j):
        """
        calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        """
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod((self.dimensions/2)**ps / ps)


    def G_int(self, i, j, k, l):
        """
        calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        """
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*self.basis[i][k]*self.basis[j][l]*np.prod((self.dimensions/2)**ps / ps)


    def E_mat(self):
        """
        put the integrals from E_int in a matrix
        Emat is the kinetic energy matrix from Arkady's paper
        """
        Etens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)*self.density
        Emat = Etens.reshape(3*self.idx,3*self.idx)
        return Emat


    def I_tens(self):
        """
        put the integrals from G_int in a tensor;
        it is the tensor in the potential energy matrix in Arkady's paper;
        i.e. it is the the potential energy matrix without the elastic tensor;
        """
        Itens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                Itens[i,k,j,l]=Itens[j,l,i,k]=self.G_int(k,l,i,j)
        return Itens


    def G_mat(self):
        """
        get potential energy matrix;
        this is a separate step because I_tens is independent of elastic constants, but only dependent on geometry;
        it is also the slow part of the calculation but only has to be done once this way
        """
        Gtens = np.tensordot(self.cijkl*1e9, self.Itens, axes= ([1,3],[0,2]))
        Gmat = np.swapaxes(Gtens, 2, 1).reshape(3*self.idx, 3*self.idx)
        return Gmat


    def compute_resonances(self, eigvals_only=True):
        """
        calculates resonance frequencies in MHz;
        pars: dictionary of elastic constants
        nb_freq: number of elastic constants to be displayed
        eigvals_only (True/False): gets only eigenvalues (i.e. resonance frequencies) or also gives eigenvectors (the latter is important when we want to calculate derivatives)
        """
        Gmat = self.G_mat()
        if eigvals_only==True:
            if self.use_quadrants==True:
                w = np.array([])
                for ii in range(8):
                    w = np.concatenate((w, linalg.eigh(Gmat[np.ix_(self.block[ii], self.block[ii])], self.Emat[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
                self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            else:
                w = linalg.eigh(Gmat, self.Emat, eigvals_only=True)
                self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            return self.freqs
        else:
            w, a = linalg.eigh(Gmat, self.Emat)
            a = a.transpose()[np.argsort(w)][6:self.nb_freq+6]
            self.freqs = np.sqrt(np.absolute(np.sort(w))[6:self.nb_freq+6])/(2*np.pi) * 1e-6 # resonance frequencies in MHz
            return self.freqs, a




    def log_derivatives_analytical(self, return_freqs=False):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f), following Arkady's paper
        """

        f, a = self.compute_resonances(eigvals_only=False)
        derivative_matrix = np.zeros((self.nb_freq, len(self.cij_dict)))
        ii = 0
        cij_dict_original = deepcopy(self.cij_dict)

        for direction in sorted(cij_dict_original):
            value = cij_dict_original[direction]
            Cderivative_dict = {key: 0 for key in cij_dict_original}
            # Cderivative_dict = {'c11': 0,'c22': 0, 'c33': 0, 'c13': 0, 'c23': 0, 'c12': 0, 'c44': 0, 'c55': 0, 'c66': 0}
            Cderivative_dict[direction] = 1
            self.cij_dict = Cderivative_dict

            Gmat_derivative = self.G_mat()
            for idx, res in enumerate(f):
                derivative_matrix[idx, ii] = np.matmul(a[idx].T, np.matmul(Gmat_derivative, a[idx]) ) / (res**2) * value
            ii += 1
        log_derivative = np.zeros((self.nb_freq, len(self.cij_dict)))
        for idx, der in enumerate(derivative_matrix):
            log_derivative[idx] = der / sum(der)

        self.cij_dict = cij_dict_original

        if return_freqs == True:
            return log_derivative, f
        elif return_freqs == False:
            return log_derivative



    def print_logarithmic_derivative(self, print_frequencies=True):
        print ('start taking derivatives ...')
        if self.Emat is None:
            self.initialize()

        log_der, freqs_calc = self.log_derivatives_analytical(return_freqs=True)

        cij_all = deepcopy(sorted(self.cij_dict))
        cij = np.array([name for name in cij_all if name[0]=='c'])
        template = ""
        for i, _ in enumerate(cij):
            template += "{" + str(i) + ":<13}"
        header = ['2 x logarithmic derivative (2 x dlnf / dlnc)']+(len(cij)-1)*['']
        der_text = template.format(*header) + '\n'
        der_text = der_text + template.format(*cij) + '\n'
        der_text = der_text + '-'*13*len(cij) + '\n'

        for ii in np.arange(self.nb_freq):
            text = [str(round(log_der[ii,j], 6)) for j in np.arange(len(cij))]
            der_text = der_text + template.format(*text) + '\n'

        if print_frequencies == True:
            freq_text = ''
            freq_template = "{0:<10}{1:<13}"
            freq_text += freq_template.format(*['idx', 'freq calc']) + '\n'
            freq_text += freq_template.format(*['', '(MHz)']) + '\n'
            freq_text += '-'*23 + '\n'
            for ii, f in enumerate(freqs_calc):
                freq_text += freq_template.format(*[int(ii), round(f, 4)]) + '\n'

            total_text = ''
            for ii in np.arange(len(der_text.split('\n'))):
                total_text = total_text + freq_text.split('\n')[ii] + der_text.split('\n')[ii] + '\n'
        else:
            total_text = der_text

        return total_text