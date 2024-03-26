from rus.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
import os


# folder_path = 'C:/Users/Florian/Box Sync/Projects/UTe2/RUS/UTe2_2203B/Red Pitaya_temperature_dependence/nice_sweeps/var_any-idx_any-overlap'
folder_path = 'C:/Users/Florian/Box Sync/Projects/UTe2/RUS/UTe2_2203B/Red Pitaya_temperature_dependence/nice_sweeps'
fit_path = 'examples/UTe2/UTe2_2203B/fit_report_UTe2_2203B_comsol_super_mesh_low_T_stokes_xyz_basis_18_152_resonances_first_3_resonances_zero_weight_with_inclusions.txt'

crystal_structure = 'orthorhombic'

reference_temperature = 1.7


# mean
high_T_el_const = {
    'c11': 89.757e9,
    'c12': 25.276e9,
    'c13': 40.987e9,
    'c22': 143.935e9,
    'c23': 31.839e9,
    'c33': 95.988e9,
    'c44': 28.099e9,
    'c55': 53.402e9,
    'c66': 30.594e9
}


# manual_indices = [1,2,4,5,7,8,9,11,13,14,15,20,21,22,29,33,34]

filenames = os.listdir(folder_path)
filenames = np.array([folder_path+'/'+i for i in filenames if i.endswith('.dat')])


# def get_consecutive_subset(array, subset_length):
#     subsets_list = []
#     l1 = array.shape[0]
#     for ii in np.arange(l1-subset_length+1):
#         subset = array[ii:ii+subset_length]
#         subsets_list.append(subset)
#     return np.array(subsets_list)

# def skip (array, num):
#     temp = get_consecutive_subset(np.arange(array.shape[0]), num)
#     temp_inv = []
#     for idx, t in enumerate(temp):
#         temp_inv.append(np.delete(array, np.array(t)))
#     return np.array(temp_inv)

# A = np.arange(filenames.shape[0])

# subsets = get_consecutive_subset(A, 32)
# index_list = []
# for subset in subsets:
#     idx = skip(subset,2)
#     index_list.append(idx)
# index_list = np.array(index_list)

# index_list = index_list.flatten().reshape((index_list.shape[0]*index_list.shape[1], index_list.shape[2]))

# good_idx = []
# for indices in index_list:
#     UTe2 = ElasticConstantsTemperatureDependence(filenames[indices], fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear')
#     Tint, fint, _ = UTe2.interpolate()
#     C_irrep, dC_irrep, T = UTe2.get_irreps(fint, Tint)

#     fit1    = polynomial.polyfit(T[T<1.45], C_irrep['A1g1'][T<1.45], 1)
#     fit2    = polynomial.polyfit(T[T>1.65], C_irrep['A1g1'][T>1.65], 1)
#     diff1   = polynomial.polyval(1.5, fit2) - polynomial.polyval(1.5, fit1)

#     fit1    = polynomial.polyfit(T[T<1.45], C_irrep['A1g2'][T<1.45], 1)
#     fit2    = polynomial.polyfit(T[T>1.65], C_irrep['A1g2'][T>1.65], 1)
#     diff2   = polynomial.polyval(1.5, fit2) - polynomial.polyval(1.5, fit1)

#     fit1    = polynomial.polyfit(T[T<1.45], C_irrep['A1g3'][T<1.45], 1)
#     fit2    = polynomial.polyfit(T[T>1.65], C_irrep['A1g3'][T>1.65], 1)
#     diff3   = polynomial.polyval(1.5, fit2) - polynomial.polyval(1.5, fit1)

#     if diff1>0 and diff2>0 and diff3>0:
#         good_idx.append(indices)

# print(good_idx)


# UTe2 = ElasticConstantsTemperatureDependence(filenames[idx[0]], fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear')
UTe2 = ElasticConstantsTemperatureDependence(filenames, fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear')
UTe2.analyze_data()

jumps = {}

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# plot c11, c22, c33
# ----------------------------------------------------------------------------------------------------
colors = {'c11':'#1eb33a', 'c22':'#1c2ec9', 'c33':'#8a09b5'}
plt.figure()
for el in ['c11', 'c22', 'c33']:
    dat       = ( np.array(UTe2.CofT_dict[el]) - np.array(UTe2.CofT_dict[el])[-1] ) / 1e9
    mask1     = (UTe2.T>1.45) & (UTe2.T<1.48)
    fit1      = polynomial.polyfit(UTe2.T[UTe2.T<1.45], dat[UTe2.T<1.45], 1)
    mask2     = (UTe2.T>1.52) & (UTe2.T<1.65)
    fit2      = polynomial.polyfit(UTe2.T[mask2    ], dat[mask2    ], 1)
    jumps[el] = - polynomial.polyval(1.5, fit2) + polynomial.polyval(1.5, fit1)

    color = colors[el]   
    line = plt.plot(UTe2.T, dat, label=el)
    plt.setp(line, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    
    plt.plot(np.linspace(1.4, 1.7), polynomial.polyval(np.linspace(1.4, 1.7), fit1), '--', color=color)
    plt.plot(np.linspace(1.4, 1.7), polynomial.polyval(np.linspace(1.4, 1.7), fit2), '--', color=color)

plt.legend()


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# plot c12, c13, c23
# ----------------------------------------------------------------------------------------------------
colors = {'c12':'#1eb33a', 'c13':'#1c2ec9', 'c23':'#8a09b5'}
plt.figure()
for el in ['c12', 'c13', 'c23']:
    dat       = ( np.array(UTe2.CofT_dict[el]) - np.array(UTe2.CofT_dict[el])[-1] ) / 1e9
    mask1     = (UTe2.T>1.45) & (UTe2.T<1.48)
    fit1      = polynomial.polyfit(UTe2.T[UTe2.T<1.45], dat[UTe2.T<1.45], 1)
    mask2     = (UTe2.T>1.52) & (UTe2.T<1.65)
    fit2      = polynomial.polyfit(UTe2.T[mask2    ], dat[mask2    ], 1)
    jumps[el] = - polynomial.polyval(1.5, fit2) + polynomial.polyval(1.5, fit1)

    color = colors[el]   
    line = plt.plot(UTe2.T, dat, label=el)
    plt.setp(line, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
    
    plt.plot(np.linspace(1.4, 1.7), polynomial.polyval(np.linspace(1.4, 1.7), fit1), '--', color=color)
    plt.plot(np.linspace(1.4, 1.7), polynomial.polyval(np.linspace(1.4, 1.7), fit2), '--', color=color)

plt.legend()

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# plot c44, c55, c66
# ----------------------------------------------------------------------------------------------------
colors = {'c44':'#1eb33a', 'c55':'#1c2ec9', 'c66':'#8a09b5'}
plt.figure()
for el in ['c55', 'c66', 'c44']:
    dat       = ( np.array(UTe2.CofT_dict[el]) - np.array(UTe2.CofT_dict[el])[-1] ) / 1e9
    color = colors[el]   
    line = plt.plot(UTe2.T, dat, label=el)
    plt.setp(line, ls ="-", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)

plt.legend()



jumps_matrix = np.array([[jumps['c11'], jumps['c12'], jumps['c13']],
                         [jumps['c12'], jumps['c22'], jumps['c23']],
                         [jumps['c13'], jumps['c23'], jumps['c33']]])

print ('the jumps in elastic constants are:')
print (jumps)

eigvals = np.linalg.eigh(jumps_matrix)[0]
print ('The eigenvalues for the "jumps" matrix are:')
print (eigvals)
plt.show()