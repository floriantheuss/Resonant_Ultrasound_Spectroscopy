import numpy as np
import matplotlib.pyplot as plt
import json



# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# specify file paths
# Mn3Ge
Mn3Ge = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out_elastic_constants.json'
Mn3Ge_error = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out_elastic_constants_error.json'
# Mn3.1Sn0.89
Mn3Sn = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out_elastic_constants.json'
Mn3Sn_error = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out_elastic_constants_error.json'


# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# import elastic constants with errors
T = {}
error = {}
elastic_constants = {}
with open(Mn3Ge) as json_file:
    data = json.load(json_file)
    T['Mn3Ge'] = data['temperature']
    elastic_constants['Mn3Ge'] = data['elastic constants']
with open(Mn3Ge_error) as json_file:
    data = json.load(json_file)
    error['Mn3Ge'] = data
with open(Mn3Sn) as json_file:
    data = json.load(json_file)
    T['Mn3Sn'] = data['temperature']
    elastic_constants['Mn3Sn'] = data['elastic constants']
with open(Mn3Sn_error) as json_file:
    data = json.load(json_file)
    error['Mn3Sn'] = data

dc_dict = {}
for crystal, el_const in elastic_constants.items():
    elastic_constants_intermediate_dict = {}
    for irrep, values in el_const.items():
        dc = (np.array(values) - np.array(values)[-1]) / 1e9
        elastic_constants_intermediate_dict[irrep] = dc
        error[crystal][irrep] = np.array(error[crystal][irrep]) / 1e9
    dc_dict[crystal] = elastic_constants_intermediate_dict






# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# plot irreps
irreps = ['A1g1', 'A1g2', 'A1g3']
plt.figure()
for irrep in irreps:
    plt.fill_between(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep]-error['Mn3Ge'][irrep], dc_dict['Mn3Ge'][irrep]+error['Mn3Ge'][irrep], alpha=0.3)
    plt.plot(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep])

irreps = ['E1g', 'E2g']
plt.figure()
for irrep in irreps:
    plt.fill_between(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep]-error['Mn3Ge'][irrep], dc_dict['Mn3Ge'][irrep]+error['Mn3Ge'][irrep], alpha=0.3)
    plt.plot(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep])

plt.show()