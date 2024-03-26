import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import odr
from scipy.interpolate import interp1d



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


######################################################################################################
######################################################################################################
# match the elastic constants of both samples
######################################################################################################
######################################################################################################
absolute_el_const = {'A1g1':92.423, 'A1g2':194.724, 'A1g3':17.387, 'E1g':44.404, 'E2g':47.650}
Tmin = max(T['Mn3Ge1'])
Tmax = min(T['Mn3Ge2'])
Tfit = np.linspace(Tmin, Tmax, 500)
def missmatch (p, x):
    return p[0] + p[1]*x

combined = {'Temperature':None, 'elastic constants': {}, 'error': {}}
for irrep in el_const:
    mn3ge1 = interp1d(T['Mn3Ge1'], dc_dict['Mn3Ge1'][irrep]) (Tfit)
    mn3ge2 = interp1d(T['Mn3Ge2'], dc_dict['Mn3Ge2'][irrep]) (Tfit)
    data = odr.RealData(Tfit, mn3ge1-mn3ge2)
    initial_guess = [0,0]
    fix = [1,0]
    model = odr.Model(missmatch)
    fit = odr.ODR(data, model, beta0=initial_guess, ifixb=fix)
    out = fit.run()
    popt = out.beta
    print (popt)

    irrep_combined = np.array(list(dc_dict['Mn3Ge1'][irrep]) + list(dc_dict['Mn3Ge2'][irrep]+missmatch(popt, T['Mn3Ge2']))) + absolute_el_const[irrep]
    error_combined = np.array(list(error['Mn3Ge1'][irrep]) + list(error['Mn3Ge2'][irrep])) / 1e9
    T_combined     = np.array(list(T['Mn3Ge1']) + list(T['Mn3Ge2']))

    irrep_combined = list(irrep_combined[np.argsort(T_combined)])
    error_combined = list(error_combined[np.argsort(T_combined)])
    T_combined     = list(np.sort(T_combined))

    combined['Temperature']              = T_combined
    combined['elastic constants'][irrep] = irrep_combined
    combined['error'][irrep]             = error_combined

save_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B_2103B_combined.json"
with open(save_path, 'w') as f:
            json.dump(combined, f, indent=4)




# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# plot irreps
fs = 17
ls = 15



t0 = 380
index1 = (abs(T['Mn3Ge1']-t0) < .05)
c1 = {key:value[index1][0] for key, value in dc_dict['Mn3Ge1'].items()}
index2 = (abs(T['Mn3Ge2']-t0) < .05)
c2 = {key:value[index2][0] for key, value in dc_dict['Mn3Ge2'].items()}



irreps = ['A1g1', 'A1g2', 'A1g3']
colors = ['midnightblue', 'purple', 'green']
plt.figure()
for i, irrep in enumerate(irreps):
    # temp = T['Mn3Ge1']
    temp = combined['Temperature']
    # dat = dc_dict['Mn3Ge1'][irrep]
    dat = np.array(combined['elastic constants'][irrep])
    # err = np.array(error['Mn3Ge1'][irrep])/1e9
    err = np.array(combined['error'][irrep])
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3, facecolor=colors[i])
    plt.plot(temp, dat, label=irrep, color=colors[i])

    mask = T['Mn3Ge2'] > 370.45
    temp = T['Mn3Ge2'][mask]
    dat = (dc_dict['Mn3Ge2'][irrep]-c2[irrep]+c1[irrep])[mask]
    err = np.array(error['Mn3Ge2'][irrep])/1e9
    err = err[mask]
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3, facecolor=colors[i])
    plt.plot(temp, dat, '--', color=colors[i])

    plt.legend(fontsize=ls)
    plt.text(415, -10, 'solid: 2001B\ndashed: 2103B', fontsize=ls)
    plt.xlabel('T (K)', fontsize=fs)
    plt.ylabel('$\\Delta$c (GPa)', fontsize=fs)
    plt.tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')
    

irreps = ['E1g', 'E2g']
plt.figure()
for i, irrep in enumerate(irreps):
    temp = T['Mn3Ge1']
    dat = dc_dict['Mn3Ge1'][irrep]
    err = np.array(error['Mn3Ge1'][irrep])/1e9
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3, facecolor=colors[i])
    plt.plot(temp, dat, label=irrep, color=colors[i])

    mask = T['Mn3Ge2'] > 370.45
    temp = T['Mn3Ge2'][mask]
    dat = (dc_dict['Mn3Ge2'][irrep]-c2[irrep]+c1[irrep])[mask]
    err = np.array(error['Mn3Ge2'][irrep])/1e9
    err = err[mask]
    plt.fill_between(temp, dat-err, dat+err, alpha=0.3, facecolor=colors[i])
    plt.plot(temp, dat, '--', color=colors[i])

    plt.legend(fontsize=ls)
    plt.text(415, -4, 'solid: 2001B\ndashed: 2103B', fontsize=ls)
    plt.xlabel('T (K)', fontsize=fs)
    plt.ylabel('$\\Delta$c (GPa)', fontsize=fs)
    plt.tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')

plt.show()