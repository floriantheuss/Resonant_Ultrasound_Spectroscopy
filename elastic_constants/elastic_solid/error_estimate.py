import numpy as np
from time import time
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import json

from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
from elastic_solid.elasticsolid import ElasticSolid


#########################################################################################################################################################
#########################################################################################################################################################
# input
#########################################################################################################################################################
#########################################################################################################################################################

order = 10             # highest order basis polynomial
nb_freq = 0             # number of frequencies included in fit (if 0, all resonances in file are used)
nb_missing_freq = 5     # maximum number of missing resonances
crystal_structure = 'hexagonal'
reference_temperature = 460


freqs_file = "C:\\Users\\Florian\\Box Sync\\Code\\Resonant_Ultrasound_Spectroscopy\\elastic_constants\\examples\\Mn3.1Sn0.89_in.txt"
folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\good_data"
mass = 0.00855e-3   # mass in kg

dimensions = [
    np.array([0.935e-3, 1.010e-3, 1.231e-3]),
    np.array([0.935e-3, 1.010e-3, 1.231e-3+5e-6]),
    np.array([0.935e-3, 1.010e-3+5e-6, 1.231e-3]),
    np.array([0.935e-3+5e-6, 1.010e-3, 1.231e-3])
    ] # dimensions of sample in m

# initial elastic constants in Pa
initElasticConstants_dict = {
    'c11': 119.854e9,
    # 'c66': 45e9,
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



#########################################################################################################################################################
#########################################################################################################################################################
# calculation
#########################################################################################################################################################
#########################################################################################################################################################

#########################################################################################################################################################
# function definitions
#########################################################################################################################################################

def fit (initial_conditions, bounds, vary, mass, dimensions, order):
    t0 = time()
    print ('initialize the class ...')
    rus = ElasticSolid(initial_conditions, bounds, vary, mass, dimensions, order, nb_freq, method='leastsq', freqs_file=freqs_file, nb_missing_res=nb_missing_freq)
    print ('class initialized in ', round(time()-t0, 4), ' s')

    rus.call = 0
    out = minimize(rus.residual_function, rus.params, method=rus.method)

    high_T_C = {}
    for c, value in out.params.items():
        high_T_C[c] = value.value
    log_der = rus.log_derivatives_analytical (high_T_C, rus.nb_freq+rus.nb_missing_res+10)

    fht_exp = rus.freqs_data / 1e6
    fht_calc = rus.resonance_frequencies (high_T_C, rus.nb_freq, eigvals_only=True) / 1e6

    return high_T_C, fht_exp, fht_calc, log_der


def temp_dep (initial_conditions, bounds, vary, mass, dimensions, order):

    high_T_C, fht_exp, fht_calc, log_der = fit (initial_conditions, bounds, vary, mass, dimensions, order)

    analysis = ElasticConstantsTemperatureDependence(folder_path, freqs_file, crystal_structure, high_T_C, reference_temperature)
    Tint, fint, gint = analysis.interpolate()
    C_irrep, dC_irrep, T = analysis.get_irreps (fint, Tint, fit_results=[fht_exp, fht_calc, log_der])

    # analysis.plot_irreps (dC_irrep, T, '$\\Delta \\mathrm{c}$ (GPa) ')

    return C_irrep, dC_irrep, T



#########################################################################################################################################################
# run loops
#########################################################################################################################################################

C = []
dC = []
T = []

for dimension in dimensions:
    c, dc, t = temp_dep (initElasticConstants_dict, ElasticConstants_bounds, ElasticConstants_vary, mass, dimension, order)
    C.append(c)
    dC.append(dc)
    T.append(t)

Creal = C[0]
dCreal = dC[0]
Treal = T[0]
Cerror = {}

for irrep in Creal:
    error = []
    for idx in np.arange(len(C))[1:]:
        error.append(np.abs(dCreal[irrep] - dC[idx][irrep]))
    error = np.transpose(error)
    maxerror = []
    for row in error:
        maxerror.append(max(row))
    Cerror[irrep] = maxerror



irreps = ['A1g1', 'A1g2', 'A1g3']
plt.figure()
for irrep in irreps:
    plt.fill_between(Treal, dCreal[irrep]-Cerror[irrep], dCreal[irrep]+Cerror[irrep], alpha=0.3)
    plt.plot(Treal, dCreal[irrep])

irreps = ['E1g', 'E2g']
plt.figure()
for irrep in irreps:
    plt.fill_between(Treal, dCreal[irrep]-Cerror[irrep], dCreal[irrep]+Cerror[irrep], alpha=0.3)
    plt.plot(Treal, dCreal[irrep])

plt.show()