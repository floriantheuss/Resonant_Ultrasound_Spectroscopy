from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
from elastic_solid.elasticsolid import ElasticSolid
import numpy as np
from time import time


# Mn3.1Sn0.98
order = 16              # highest order basis polynomial
nb_freq = 0             # number of frequencies included in fit (if 0, all resonances in file are used)
nb_missing_freq = 5     # maximum number of missing resonances
# maxiter = 100
method = 'differential_evolution' # fit method: 'differential_evolution' and 'leastsq' are allowed
# method = 'leastsq'

# freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\test\\Mn3.1Sn0.98.txt"
freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\examples\\Mn3.1Sn0.89_2010A_in.txt"
mass = 0.00855e-3   # mass in kg
dimensions = np.array([0.935e-3, 1.010e-3, 1.231e-3]) # dimensions of sample in m
             # first value is length along [100], second along [010], and third is along [001]

# initial elastic constants in Pa
initElasticConstants_dict = {
    'c11': 119.854e9,
    # 'c66': 46e9,
    'c12': 28.058e9,
    'c13': 14.273e9,
    'c33': 142.826e9,
    'c44': 44.006e9
    }
        
ElasticConstants_bounds = {
    'c11': [110e9, 140e9],
    # 'c66': [30e9, 60e9],
    'c12': [20e9, 40e9],
    'c13': [0, 30e9],
    'c33': [125e9, 155e9],
    'c44': [30e9, 60e9]
    }
        
ElasticConstants_vary = {
    'c11': True,
    # 'c66': True,
    'c12': True,
    'c13': True,
    'c33': True,
    'c44': True
    }

    


t0 = time()
print ('initialize the class ...')
rus = ElasticSolid(initElasticConstants_dict, ElasticConstants_bounds, ElasticConstants_vary, mass, dimensions, order, nb_freq, method, freqs_file, nb_missing_freq)#, maxiter=maxiter)
print ('class initialized in ', round(time()-t0, 4), ' s')


rus.fit()