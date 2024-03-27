#%%
# Import packages
from rus_comsol.rus_fitting import RUSFitting
from rus_comsol.rus_comsol import RUSComsol
from rus_comsol.rus_rpr import RUSRPR
import sys
import os
import numpy as np
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# get current directory
print ('this is the current working directory')
print (os.getcwd())

#%%
## Initial values
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


mass = 0.0089e-3 # kg
dimensions = np.array([0.911e-3, 1.020e-3, 1.305e-3]) # m
density = mass/np.prod(dimensions)
print (density)

#%%
# Initialize RUS object

# rus_object = RUSRPR(cij_dict=elastic_dict, symmetry="rhombohedral", angle_z=-90,
#                         dimensions=dimensions,
#                         mass = mass,
#                         order = 16,
#                         nb_freq=20, use_quadrants=False)


rus_object = RUSComsol(cij_dict=elastic_dict, symmetry="hexagonal",
                       density = density,
                       mph_file="Mn3Ge_2104C_no_incl.mph",
                       mesh=3,nb_freq=30)


#%%
# rus_object.start_comsol()
# res = rus_object.compute_resonances()
# print(res)
#%%
# Initialize Fit Object

fitObject = RUSFitting(rus_object=rus_object, bounds_dict=bounds_dict,
                        freqs_file='RT_exp_data_for_fit.dat',
                        nb_freqs='all',
                        nb_workers=30, nb_max_missing=5,
                       mutation=0.9, crossing=0.7, population=15,
                       polish=False)


#%%
# Run fit

rus_object = fitObject.run_fit()
fitObject.run_fit(print_derivatives=False)


#%%
# Stop Comsol

rus_object.stop_comsol()
