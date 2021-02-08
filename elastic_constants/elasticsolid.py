import numpy as np
from scipy import linalg as LA
from time import time
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import copy
from multiprocessing import cpu_count, Pool

class ElasticSolid:

    # instance attributes
    def __init__(self, initElasticConstants_dict, mass, dimensions, order, nb_freq, method='differential_evolution'):
        """
        initElasticConstants_dict: a dictionary of elastic constants in Pa
        mass: a number in kg
        dimensions: numpy array of x, y, z lengths in m
        order: integer - highest order polynomial used to express basis functions
        nb_freq: number of frequencies to display
        method: fitting method
        """
        
        self.mass       = mass # mass of the sample
        self.rho        = mass / np.prod(dimensions)

        self.order      = order # order of the highest polynomial used to calculate the resonacne frequencies
        self.N          = int((order+1)*(order+2)*(order+3)/6) # this is the number of basis functions

        self.Vol         = dimensions/2 # sample dimensions divided by 2

        self.elasticConstants_dict = copy.deepcopy(initElasticConstants_dict)

        self.nb_freq = nb_freq

        
        self.method      = method # "shgo", "differential_evolution", "leastsq"
        self.pars        = Parameters()
        # for param_name, param_range in self.elasticConstants_dict.items():
        #     self.pars.add(param_name, value = self.elasticConstants_dict[param_name])#, min = param_range[0], max = param_range[-1])
        



        # create basis and sort it based on its parity;
        # for details see Arkady's paper;
        # this is done here in __init__ because we only need to is once and it is the "long" part of the calculation
        lookUp = {(1, 1, 1) : 0, (1, 1, -1) : 1, (1, -1, 1) : 2, (-1, 1, 1) : 3, (1, -1, -1): 4, (-1, 1, -1) : 5, (-1, -1, 1) : 6, (-1, -1, -1) : 7}

        self.basis  = np.zeros((self.N, 3))
        self.idx    =  0
        self.block = [[],[],[],[],[],[],[],[]]
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
    


            


    def elastic_tensor (self, pars):
        """
        returns the elastic tensor from given elastic constants in pars
        (a dictionary of elastic constants)
        based on the length of pars it decides what crystal structure we the sample has
        """
        ctens = np.zeros([3,3,3,3])

        if len(pars) == 3:                      # cubic
            c11 = c22 = c33 = pars['c11']
            c12 = c13 = c23 = pars['c12']
            c44 = c55 = c66 = pars['c44']

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

        ctens[0,0,0,0] = c11
        ctens[1,1,1,1] = c22
        ctens[2,2,2,2] = c33
        ctens[0,0,1,1] = ctens[1,1,0,0] = c12
        ctens[2,2,0,0] = ctens[0,0,2,2] = c13
        ctens[1,1,2,2] = ctens[2,2,1,1] = c23
        ctens[0,1,0,1] = ctens[1,0,0,1] = ctens[0,1,1,0] = ctens[1,0,1,0] = c44
        ctens[0,2,0,2] = ctens[2,0,0,2] = ctens[0,2,2,0] = ctens[2,0,2,0] = c55
        ctens[1,2,1,2] = ctens[2,1,2,1] = ctens[2,1,1,2] = ctens[1,2,2,1] = c66

        return ctens



    def E_int (self, i, j):
        """
        calculates integral for kinetic energy matrix, i.e. the product of two basis functions
        """
        ps = self.basis[i] + self.basis[j] + 1.
        if np.any(ps%2==0): return 0.
        return 8*np.prod(self.Vol**ps / ps)

    def G_int (self, i, j, k, l):
        """
        calculates the integral for potential energy matrix, i.e. the product of the derivatives of two basis functions
        """
        M = np.array([[[2.,0.,0.],[1.,1.,0.],[1.,0.,1.]],[[1.,1.,0.],[0.,2.,0.],[0.,1.,1.]],[[1.,0.,1.],[0.,1.,1.],[0.,0.,2.]]])
        if not self.basis[i][k]*self.basis[j][l]: return 0
        ps = self.basis[i] + self.basis[j] + 1. - M[k,l]
        if np.any(ps%2==0): return 0.
        return 8*self.basis[i][k]*self.basis[j][l]*np.prod(self.Vol**ps / ps)

    def E_mat (self):
        """
        put the integrals from E_int in a matrix
        Emat is the kinetic energy matrix from Arkady's paper
        """
        Etens = np.zeros((3,self.idx,3,self.idx), dtype= np.double)
        for x in range(3*self.idx):
            i, k = int(x/self.idx), x%self.idx
            for y in range(x, 3*self.idx):
                j, l = int(y/self.idx), y%self.idx
                if i==j: Etens[i,k,j,l]=Etens[j,l,i,k]=self.E_int(k,l)*self.rho
        
        Emat = Etens.reshape(3*self.idx,3*self.idx)
        return Emat

    def I_tens (self):
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

    def G_mat (self, pars):
        """
        get potential energy matrix;
        this is a separate step because I_tens is independent of elastic constants, but only dependent on geometry;
        it is also the slow part of the calculation but only has to be done once this way
        """
        C = self.elastic_tensor(pars)
        Gtens = np.tensordot(C, self.Itens, axes= ([1,3],[0,2]))
        Gmat = np.swapaxes(Gtens, 2, 1).reshape(3*self.idx, 3*self.idx)
        return Gmat

    def resonance_frequencies (self, pars=None, nb_freq=None, eigvals_only=True):
        """
        calculates resonance frequencies in Hz;
        pars: dictionary of elastic constants
        nb_freq: number of elastic constants to be displayed
        eigvals_only (True/False): gets only eigenvalues (i.e. resonance frequencies) or also gives eigenvectors (the latter is important when we want to calculate derivatives)
        """

        
        if nb_freq==None:
            nb_freq = self.nb_freq   
        if pars is None:
            pars = self.elasticConstants_dict

        Gmat = self.G_mat(pars)
        if eigvals_only==True:
            w = np.array([])
            for ii in range(8): 
                w = np.concatenate((w, LA.eigh(Gmat[np.ix_(self.block[ii], self.block[ii])], self.Emat[np.ix_(self.block[ii], self.block[ii])], eigvals_only=True)))
            f = np.sqrt(np.absolute(np.sort(w))[6:nb_freq+6])/(2*np.pi) # resonance frequencies in Hz
            return f
        else:
            w, a = LA.eigh(Gmat, self.Emat)
            a = a.transpose()[np.argsort(w)][6:nb_freq+6]
            f = np.sqrt(np.absolute(np.sort(w))[6:nb_freq+6])/(2*np.pi) 
            return f, a



    def log_derivatives_analytical (self, pars):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f), following Arkady's paper
        """
        f, a = self.resonance_frequencies(pars, eigvals_only=False)
        derivative_matrix = np.zeros((self.nb_freq, len(pars)))
        ii = 0
        for direction, value in pars.items():
            Cderivative_dict = {key: 0 for key in pars}
            Cderivative_dict[direction] = 1
            Gmat_derivative = self.G_mat(Cderivative_dict)
            for idx, res in enumerate(f):
                derivative_matrix[idx, ii] = np.matmul(a[idx], np.matmul(Gmat_derivative, a[idx]) ) / res * 2*np.pi * value
            ii += 1
        log_derivative = np.zeros((self.nb_freq, len(pars)))
        for idx, der in enumerate(derivative_matrix):
            log_derivative[idx] = der / sum(der)
        
        # print the logarithmic derivatives of each frequency
        formats = "{0:<15}{1:<15}"
        k = 2
        for _ in log_derivative[0]:
            formats = formats + '{' + str(k) + ':<15}'
            k+=1
        print ('-----------------------------------------------------------------------')
        print ('-----------------------------------------------------------------------')
        print ('2 x LOGARITHMIC DERIVATIVES')
        print ('-----------------------------------------------------------------------')
        print (formats.format('f [MHz]','dlnf/dlnc11','dlnf/dlnc12','dlnf/dlnc44','SUM') )
        for idx, line in enumerate(log_derivative):
            text = [str(round(f[idx]/1e6,6))] + [str(round(d, 6)) for d in line] + [str(round(sum(line),7))]
            print ( formats.format(*text) )
        print ('-----------------------------------------------------------------------')
        print ('-----------------------------------------------------------------------')

        return log_derivative




    def log_derivatives_numerical (self, pars=None, dc=1e5, N=5, Rsquared_threshold=1e-5, parallel=False, nb_workers=None ):
        """
        calculating logarithmic derivatives of the resonance frequencies with respect to elastic constants,
        i.e. (df/dc)*(c/f), by calculating the resonance frequencies for slowly varying elastic constants
        variables: pars (dictionary of elastic constants), dc, N
        The derivative is calculated by computing resonance frequencies for N different elastic cosntants centered around the value given in pars and spaced by dc.
        A line is then fitted through these points and the slope is extracted as the derivative.
        """
        if pars is None:
            pars = self.elasticConstants_dict
        if nb_workers is None:
            nb_workers = min( [int(cpu_count()/2), N] )


        # calculate the resonance frequencies for the "true" elastic constants
        freq_result = self.resonance_frequencies(pars=pars)


        if parallel == True:
                print("# of available cores: ", cpu_count())
                pool = Pool(processes=nb_workers)
                print("--- Pool initialized with ", nb_workers, " workers ---")
        

        fit_results_dict = {}
        Rsquared_matrix = np.zeros([len(freq_result), len(pars)])
        log_derivative_matrix = np.zeros([len(freq_result), len(pars)])
        # take derivatives with respect to all elastic constants
        ii = 0
        for elastic_constant in pars:
            print ('start taking derivative with respect to ', elastic_constant)
            t1 = time()
            # create an array of elastic constants centered around the "true" value
            c_result = pars[elastic_constant]
            c_derivative_array = np.linspace(c_result-N/2*dc, c_result+N/2*dc, N)
            elasticConstants_derivative_dict = copy.deepcopy(pars)
            # calculate the resonance frequencies for all elastic constants in c_test and store them in Ctest

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # # this calculates all the necessary sets of resonance frequencies for the derivative in parallel
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            elasticConstants_derivative_array = []
            # here we are creating an array where each element is a dictionary of a full set
            # of elastic constants. Hopefully, we can give this array to a pool.map
            for c in c_derivative_array:
                elasticConstants_derivative_dict[elastic_constant] = c
                # copy.deepcopy actually makes a copy of the dictionary instead of just creating a new pointer to the same location
                elasticConstants_derivative_array.append(copy.deepcopy(elasticConstants_derivative_dict))
            
            if parallel == True:
                # print("# of available cores: ", cpu_count())
                # pool = Pool(processes=nb_workers)
                freq_derivative_matrix = pool.map(self.resonance_frequencies, elasticConstants_derivative_array) - np.array([freq_result for _ in np.arange(N)])
                freq_derivative_matrix = np.transpose( np.array( freq_derivative_matrix ) )
            else:
                freq_derivative_matrix = np.zeros([len(freq_result), N])
                for idx, parameter_set in enumerate(elasticConstants_derivative_array):
                    freq_derivative_matrix[:, idx] = self.resonance_frequencies(pars=parameter_set) - freq_result


            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # # this calculates all the necessary sets of resonance frequencies for the derivative in series
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # for idx, c in enumerate(c_derivative_array):
            #     elasticConstants_derivative_dict[elastic_constant] = c
            #     # note we don't actually save the resonance frequencies, but we shift them by the values at the "true" elastic constants;
            #     # this is done because within the elastic constants in c_test the frequencies change only very little compared to their absolute value,
            #     # thus this shift is important to get a good fit later
            #     freq_derivative_matrix[:,idx] = self.resonance_frequencies(pars=elasticConstants_derivative_dict)-freq_result
            
            # shift array of elastic constants to be centered around zero, for similar argument made for the shift of resonance frequencies
            c_derivative_array = c_derivative_array - c_result
            
            fit_matrix = np.zeros([len(freq_result), N])
            # here we fit a straight line to the resonance frequency vs elastic costants for all resonances
            for idx, freq_derivative_array in enumerate(freq_derivative_matrix):
                # popt, pcov = curve_fit(line, Ctest, freq, p0=[1e-7, 0])
                slope, y_intercept = np.polyfit(c_derivative_array, freq_derivative_array, 1)
                log_derivative_matrix[idx, ii] = 2 * slope * pars[elastic_constant]/freq_result[idx]

                ## check if data really lies on a line
                # offset.append(popt[1])
                current_fit = slope*c_derivative_array + y_intercept
                fit_matrix[idx,:] = current_fit
                # calculate R^2;
                # this is a value judging how well the data is described by a straight line
                SStot = sum( (freq_derivative_array - np.mean(freq_derivative_array))**2 )
                SSres = sum( (freq_derivative_array - current_fit)**2 )
                Rsquared = 1 - SSres/SStot
                Rsquared_matrix[idx, ii] = Rsquared
                # we want a really good fit!
                # R^2 = 1 would be perfect
                if abs(1-Rsquared) > Rsquared_threshold:
                    # if these two fits differ by too much, just print the below line and plot that particular data
                    print ('not sure if data is a straight line ', elastic_constant, ' ', freq_result[idx]/1e6, ' MHz')
                    plt.figure()
                    plt.plot(c_derivative_array/1e3, freq_derivative_array, 'o')
                    plt.plot(c_derivative_array/1e3, current_fit)
                    plt.title(elastic_constant +'; f = ' + str(round(freq_result[idx]/1e6, 3)) + ' MHz; $R^2$ = ' + str(round(Rsquared, 7)))
                    plt.xlabel('$\\Delta c$ [kPa]')
                    plt.ylabel('$\\Delta f$ [Hz]')
                    plt.show()
                
            # store all fit results in this dictionary, just in case you need to look at this at some point later
            fit_results_dict[elastic_constant] = {
                'freq_test': freq_derivative_matrix,
                'c_test': c_derivative_array,
                'fit': fit_matrix,
                'Rsquared': Rsquared_matrix
            }

            print ('derivative with respect to ', elastic_constant, ' done in ', round(time()-t1, 4), ' s')
            ii += 1

        if parallel == True:
            pool.terminate()

        # print the logarithmic derivatives of each frequency
        formats = "{0:<15}{1:<15}"
        k = 2
        for _ in log_derivative_matrix[0]:
            formats = formats + '{' + str(k) + ':<15}'
            k+=1
        print ('-----------------------------------------------------------------------')
        print ('-----------------------------------------------------------------------')
        print ('2 x LOGARITHMIC DERIVATIVES')
        print ('-----------------------------------------------------------------------')
        print (formats.format('f [MHz]','dlnf/dlnc11','dlnf/dlnc12','dlnf/dlnc44','SUM') )
        for idx, line in enumerate(log_derivative_matrix):
            text = [str(round(freq_result[idx]/1e6,6))] + [str(round(d, 6)) for d in line] + [str(round(sum(line),7))]
            print ( formats.format(*text) )
        print ('-----------------------------------------------------------------------')
        print ('-----------------------------------------------------------------------')
    
        return (log_derivative_matrix)






if __name__ == '__main__':

    order = 12
    mass = 0.045e-3
    dimensions = np.array([0.145e-2, 0.201e-2, 0.302e-2])

    initElasticConstants_dict = {
        'c11': 321.61990e9,
        'c12': 103.50101e9,
        'c44': 124.99627e9
        }

    nb_freq = 30
    
    t0 = time()
    print ('initialize the class ...')
    srtio3 = ElasticSolid(initElasticConstants_dict, mass, dimensions, order, nb_freq, method='differential_evolution')
    print ('class initialized in ', round(time()-t0, 4), ' s')

    f = srtio3.resonance_frequencies()
    print(f/1e6)

    dern, fit_results = srtio3.log_derivatives_numerical(pars=initElasticConstants_dict)
    dera = srtio3.log_derivatives_analytical(pars=initElasticConstants_dict)
    

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # plot the fit results for the derivative >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # plt.figure()
    # for line in np.transpose(derivatives):
    #     plt.plot(np.arange(len(line)), line, 'o')#, label=key)
    # # plt.legend()

  
    # # test derivatives
    # fig, ax = plt.subplots(1,1)
    # elc = 'c12'

    
    # freqtest = fit_results[elc]['freq_test']
    # Ctest = fit_results[elc]['c_test']
    # # slope = fit_results[elc]['slope']
    # # offset = fit_results[elc]['offset']
    # # sigma = fit_results[elc]['error']
    # Rsq = fit_results[elc]['Rsquared'][:,1]
    # fit = fit_results[elc]['fit']
    # # for ii in np.arange(len(offset)):
    # #     fit.append(  slope[ii]*Ctest + offset[ii]  )

    
    # # SSres = sum( ( freqtest[0] - fit[0] )**2 )
    # # SStot = sum( ( freqtest[0] - np.mean(freqtest[0]) )**2 )
    # # ax.set_title(str(1-SSres/SStot))
    # ax.set_title(str(Rsq[0]))
    # plot = ax.scatter(Ctest, freqtest[0])
    # plotf, = ax.plot(Ctest, fit[0], color='red')
    # # plots, = ax.plot(Ctest, sigma[0], color='green')

    # # set slider which goes between different resonances
    # slider_axis = plt.axes([0.15, .92, 0.2, 0.03], facecolor='lightgrey')
    # slider = Slider(slider_axis, 'Resonance', 0, len(freqtest)-1, valinit=0, valstep=1, color='red')
    # slider.label.set_size(15)

    # def update(val):
    #     plot.set_offsets(np.array(list(zip(Ctest, freqtest[val]))))
    #     plotf.set_ydata(fit[val])
    #     # plots.set_ydata(sigma[val])
    #     mean = np.mean(freqtest[val])
    #     dev = max(freqtest[val]) - mean
    #     ax.set_ylim(mean-1.1*dev, mean+1.1*dev)
    #     # SSres = sum( ( freqtest[val] - fit[val] )**2 )
    #     # SStot = sum( ( freqtest[val] - np.mean(freqtest[val]) )**2 )
    #     # ax.set_title(str(1-SSres/SStot))
    #     ax.set_title(str(Rsq[val]))
    #     fig.canvas.draw_idle()


    # slider.on_changed(update)



    # plt.show()