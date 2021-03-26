from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
from elastic_solid.elasticsolid import ElasticSolid
import numpy as np
from time import time


# Mn3.1Sn0.98
order = 16
nb_freq = 0
nb_missing_freq = 5
maxiter = 100
# method = 'differential_evolution'
method = 'leastsq'

freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\examples\\Mn3Ge_in.txt"
mass = 0.0089e-3
dimensions = np.array([0.911e-3, 1.020e-3, 1.305e-3])

initElasticConstants_dict = {
    'c11': 138.8e9,
    # 'c66': 45e9,
    'c12': 42e9,
    'c13': 14.3e9,
    'c33': 194.6e9,
    'c44': 45.2e9
    }
        
ElasticConstants_bounds = {
    'c11': [120e9, 150e9],
    # 'c66': [30e9, 60e9],
    'c12': [25e9, 55e9],
    'c13': [0, 30e9],
    'c33': [180e9, 210e9],
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