from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
from elastic_solid.elasticsolid import ElasticSolid
import numpy as np
from time import time


# Mn3.1Sn0.98
order = 16
nb_freq = 0
nb_missing_freq = 5
maxiter = 100
method = 'differential_evolution'
# method = 'leastsq'

# freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\test\\Mn3.1Sn0.98.txt"
freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\examples\\Mn3.1Sn0.89_new.txt"
mass = 0.00855e-3
dimensions = np.array([0.935e-3, 1.010e-3, 1.231e-3])

initElasticConstants_dict = {
    'c11': 123.568e9,
    # 'c66': 45e9,
    'c12': 33.712e9,
    'c13': 18.007e9,
    'c33': 142.772e9,
    'c44': 42.446e9
    }
        
ElasticConstants_bounds = {
    'c11': [110e9, 140e9],
    # 'c66': [30e9, 60e9],
    'c12': [20e9, 50e9],
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
srtio3 = ElasticSolid(initElasticConstants_dict, ElasticConstants_bounds, ElasticConstants_vary, mass, dimensions, order, nb_freq, method, freqs_file, nb_missing_freq)#, maxiter=maxiter)
print ('class initialized in ', round(time()-t0, 4), ' s')


srtio3.fit()