from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
import numpy as np


folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\good_data"
fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out.txt"
# fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\5um_error\\Mn3.1Sn0.89_out_err1.txt"
# fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\5um_error\\Mn3.1Sn0.89_out_err3.txt"
# fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\5um_error\\Mn3.1Sn0.89_out_err5.txt"

crystal_structure = 'hexagonal'

reference_temperature = 460
# manual_indices = np.array( [1, 2, 3, 4, 5, 6, 8, 12, 16, 17, 18, 20, 19, 22, 25, 27, 29, 38, 39, 40, 41, 42, 43, 44] )
manual_indices = np.array( [1, 2, 3, 4, 5, 6, 8, 12, 16, 17, 18, 20, 19, 22, 27, 29, 38, 39, 40, 41, 42, 43, 44] )
manual_indices = np.array( [1, 2, 3, 4, 5, 6, 8, 12, 16, 17, 18] )

# mean
high_T_el_const = {
    'c11': 119.542e9,
    'c12': 28.098e9,
    'c13': 13.82e9,
    'c33': 142.709e9,
    'c44': 43.922e9
    # 'c66': 45.3
}

# # error 1
# high_T_el_const = {
#     'c11': 120.049e9,
#     'c12': 28.734e9,
#     'c13': 13.557e9,
#     'c33': 141.634e9,
#     'c44': 43.74e9
#     # 'c66': 45.3
# }

# # error 3
# high_T_el_const = {
#     'c11': 119.558e9,
#     'c12': 28.166e9,
#     'c13': 13.791e9,
#     'c33': 141.851e9,
#     'c44': 43.847e9
#     # 'c66': 45.3
# }

# # error 5
# high_T_el_const = {
#     'c11': 119.037e9,
#     'c12': 27.978,
#     'c13': 13.912e9,
#     'c33': 143.327e9,
#     'c44': 43.963e9
#     # 'c66': 45.3
# }

Mn31Sn089 = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature)#, manual_indices=manual_indices)
Mn31Sn089.do_everything()