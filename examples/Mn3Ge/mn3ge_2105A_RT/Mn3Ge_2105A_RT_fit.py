from numpy.lib.function_base import angle
from rus.rus_fitting import RUSFitting
from rus.rus_comsol import RUSComsol
from rus.rus_rpr import RUSRPR
import sys
import os
import numpy as np


order = 16
nb_freq = 75
nb_missing_freq = 5
maxiter = 1000
method = 'differential_evolution'
# method = 'leastsq'

print (os.getcwd())
freqs_file = "examples/Mn3Ge/mn3ge_2105A_RT/data_for_fit.dat"

mass = 0.0089e-3/(0.911*1.020*1.305)*(0.869*1.010*1.193)
dimensions = np.array([0.869e-3, 1.010e-3, 1.193e-3]) 


# elastic constants init in GPa
elastic_dict = {
    'c11': 138.8,
    # 'c66': 45,
    'c12': 42,
    'c13': 14.3,
    'c33': 194.6,
    'c44': 45.2
    }

# bounds for fit; el const in GPa, angles in degrees        
bounds_dict = {
    'c11': [110, 160],
    # 'c66': [30, 60],
    'c12': [15, 65],
    'c13': [-10, 40],
    'c33': [170, 220],
    'c44': [20, 70]
    }
        


    

print ('initialize rus object')
rus_object = RUSRPR(cij_dict=elastic_dict, symmetry="hexagonal", 
                                angle_x=0, angle_y=0, angle_z=0,
                                dimensions=dimensions,
                                mass = mass,
                                order = order,
                                nb_freq=nb_freq, use_quadrants=True)
rus_object.initialize()
print ('initialized ...')

fit = RUSFitting(rus_object, bounds_dict,
                 freqs_file=freqs_file, nb_freqs=nb_freq, nb_missing=nb_missing_freq, method=method, polish=False, updating='deferred')

fit.run_fit(print_derivatives=False)