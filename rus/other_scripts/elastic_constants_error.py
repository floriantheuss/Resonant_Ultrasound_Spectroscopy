import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
import json


# Mn3Ge
filepath = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out_elastic_constants.json'
folder = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error'
# Mn3.1Sn0.89
filepath = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out_elastic_constants.json'
folder = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\5um_error'

with open(filepath) as json_file:
    data_mean = json.load(json_file)
c_mean = data_mean['elastic constants']
dc_mean = {key:np.array(item)-np.array(item)[-1] for key, item in c_mean.items()}
T = data_mean['temperature']



data_files = os.listdir(folder)
data_files = [i for i in data_files if i[-5:]=='.json']
filenames = [folder + '\\' + i for i in data_files]


error = {key:np.array(item)*0 for key, item in c_mean.items() }
for error_file in filenames:
    with open(error_file) as json_file:
        c_error = json.load(json_file)['elastic constants']
        dc_error = {key:np.array(item)-np.array(item)[-1] for key, item in c_error.items()}
    
    for irrep, values in dc_mean.items():
        new_error = np.abs( np.array(values) - np.array(dc_error[irrep]) )
        previous_error = error[irrep]
        total_error = np.array( [ max([new_error[i], previous_error[i]]) for i in np.arange(len(T)) ] )
        error[irrep] = list(total_error)



save_path = filepath[:-5] + '_error.json'
with open(save_path, 'w') as f:
            json.dump(error, f, indent=4)



irreps = ['A1g1', 'A1g2', 'A1g3']
plt.figure()
for irrep in irreps:
    plt.fill_between(T, dc_mean[irrep]-error[irrep], dc_mean[irrep]+error[irrep], alpha=0.3)
    plt.plot(T, dc_mean[irrep])

irreps = ['E1g', 'E2g']
plt.figure()
for irrep in irreps:
    plt.fill_between(T, dc_mean[irrep]-error[irrep], dc_mean[irrep]+error[irrep], alpha=0.3)
    plt.plot(T, dc_mean[irrep])

plt.show()