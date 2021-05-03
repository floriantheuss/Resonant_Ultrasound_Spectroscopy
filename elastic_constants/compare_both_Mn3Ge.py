import numpy as np
import matplotlib.pyplot as plt
import json



# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# specify file paths
# Mn3Ge
Mn3Ge1 = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\temp_dependent_data_with_Labview\\improved_setup\\good_data\\irreps_with_T_incl_error.json"
Mn3Ge2 = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2103B\\good_data\\irreps_with_T_incl_error.json"



# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# import elastic constants with errors
T = {}
error = {}
elastic_constants = {}
with open(Mn3Ge1) as json_file:
    data = json.load(json_file)
    T['Mn3Ge1'] = np.array(data['Temperature'])
    elastic_constants['Mn3Ge1'] = data['elastic constants']
    error['Mn3Ge1'] = data['error']

with open(Mn3Ge2) as json_file:
    data = json.load(json_file)
    T['Mn3Ge2'] = np.array(data['Temperature'])
    elastic_constants['Mn3Ge2'] = data['elastic constants']
    error['Mn3Ge2'] = data['error']

c_reference = {key:value[-1] for key,value in elastic_constants['Mn3Ge2'].items()}


dc_dict = {}
for crystal, el_const in elastic_constants.items():
    elastic_constants_intermediate_dict = {}
    for irrep, values in el_const.items():
        elastic_constants[crystal][irrep] = np.array( elastic_constants[crystal][irrep] )
        dc = (np.array(values) - c_reference[irrep]) / 1e9
        elastic_constants_intermediate_dict[irrep] = dc
    dc_dict[crystal] = elastic_constants_intermediate_dict



# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# plot irreps

t0 = 380
index1 = (abs(T['Mn3Ge1']-t0) < .05)
c1 = {key:value[index1][0] for key, value in dc_dict['Mn3Ge1'].items()}
index2 = (abs(T['Mn3Ge2']-t0) < .05)
c2 = {key:value[index2][0] for key, value in dc_dict['Mn3Ge2'].items()}



irreps = ['A1g1', 'A1g2', 'A1g3']
plt.figure()
for i, irrep in enumerate(irreps):
    temp = T['Mn3Ge1']
    dat = dc_dict['Mn3Ge1'][irrep]
    err = np.array(error['Mn3Ge1'][irrep])/1e9
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3)
    plt.plot(temp, dat, label=irrep)

    mask = T['Mn3Ge2'] > 370.45
    temp = T['Mn3Ge2'][mask]
    dat = (dc_dict['Mn3Ge2'][irrep]-c2[irrep]+c1[irrep])[mask]
    err = np.array(error['Mn3Ge2'][irrep])/1e9
    err = err[mask]
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3)
    plt.plot(temp, dat)

    plt.legend()
    plt.xlabel('T (K)')
    plt.ylabel('$\\Delta$c (GPa)')
    

irreps = ['E1g', 'E2g']
plt.figure()
for i, irrep in enumerate(irreps):
    temp = T['Mn3Ge1']
    dat = dc_dict['Mn3Ge1'][irrep]
    err = np.array(error['Mn3Ge1'][irrep])/1e9
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3)
    plt.plot(temp, dat, label=irrep)

    mask = T['Mn3Ge2'] > 370.45
    temp = T['Mn3Ge2'][mask]
    dat = (dc_dict['Mn3Ge2'][irrep]-c2[irrep]+c1[irrep])[mask]
    err = np.array(error['Mn3Ge2'][irrep])/1e9
    err = err[mask]
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3)
    plt.plot(temp, dat)

    plt.legend()
    plt.xlabel('T (K)')
    plt.ylabel('$\\Delta$c (GPa)')

plt.show()