from numpy.lib.function_base import angle
from rus.rus_fitting import RUSFitting
from rus.rus_comsol import RUSComsol
from rus.rus_rpr import RUSRPR
import sys
import os
import numpy as np


order = 16
nb_freq = 'all'
nb_missing_freq = 5
maxiter = 1000
# method = 'differential_evolution'
method = 'leastsq'

print (os.getcwd())
freqs_file = "examples/Mn3Ge/new_Mn3Ge_2001B_in_for_fit.dat"

mass = 0.0089e-3 #2001B
dimensions = np.array([0.911e-3, 1.020e-3, 1.305e-3]) #2001B


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
    'c11': [120, 150],
    # 'c66': [30, 60],
    'c12': [25, 55],
    'c13': [0, 30],
    'c33': [180, 210],
    'c44': [30, 60]
    }
        


    

print ('initialize rus object')
rus_object = RUSRPR(cij_dict=elastic_dict, symmetry="hexagonal", 
                                angle_x=0, angle_y=0, angle_z=0,
                                dimensions=dimensions,
                                mass = mass,
                                order = order,
                                nb_freq=100, use_quadrants=True)
rus_object.initialize()
print ('initialized ...')

fit = RUSFitting(rus_object, bounds_dict,
                 freqs_file=freqs_file, nb_freqs=nb_freq, nb_missing=nb_missing_freq, method=method)

fit.run_fit(print_derivatives=True)