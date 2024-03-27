from rus.old.elasticconstantswithtemperature_old import ElasticConstantsTemperatureDependence
import numpy as np


folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\good_data"
fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\Mn3.1Sn0.89_2010A_out.txt"

crystal_structure = 'hexagonal'

reference_temperature = 455
# manual_indices = np.array( [1, 2, 3, 4, 5, 6, 8, 12, 16, 17, 18, 20, 19, 22, 25, 27, 29, 38, 39, 40, 41, 42, 43, 44] )
manual_indices = np.array( [1, 2, 3, 4, 5, 6, 9, 15, 17, 18, 22, 25, 27, 29, 38, 39, 40, 41, 42, 43, 44, 45] )
manual_indices = np.array( [ 1,  2,  3,  4,  5,  6,  7,  9, 12, 15, 17, 25, 27, 29, 38, 43, 44, 45] )

# mean
high_T_el_const = {
    'c11': 119.824e9,
    'c12': 28.284e9,
    'c13': 13.769e9,
    'c33': 142.562e9,
    'c44': 44.922e9
    # 'c66': 45.3
}


Mn31Sn089 = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear')
# Mn31Sn089 = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear', manual_indices=manual_indices)
Mn31Sn089.do_everything()