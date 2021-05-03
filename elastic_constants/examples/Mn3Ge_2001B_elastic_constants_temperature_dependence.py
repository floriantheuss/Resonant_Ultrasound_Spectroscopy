from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
import numpy as np


folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\temp_dependent_data_with_Labview\\improved_setup\\good_data"
fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err6.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err5.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err4.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err3.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err2.txt'
# fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\5um_error\\Mn3Ge_out_err1.txt'

crystal_structure = 'hexagonal'

reference_temperature = 387
manual_indices = np.array( [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 36, 40, 41, 42] )

# mean
high_T_el_const = {
    'c11': 138.357e9,
    'c12': 42.159e9,
    'c13': 14.464e9,
    'c33': 194.327e9,
    'c44': 45.059e9
}

# # error 6
# high_T_el_const = {
#     'c11': 138.917e9,
#     'c12': 42.326e9,
#     'c13': 14.405e9,
#     'c33': 193.476e9,
#     'c44': 45.023e9
# }

# # error 5
# high_T_el_const = {
#     'c11': 137.819e9,
#     'c12': 42.016e9,
#     'c13': 14.527e9,
#     'c33': 195.14e9,
#     'c44': 45.101e9
# }

# # error 4
# high_T_el_const = {
#     'c11': 138.649e9,
#     'c12': 42.539e9,
#     'c13': 14.892e9,
#     'c33': 195.376e9,
#     'c44': 45.18e9
# }

# # error 3
# high_T_el_const = {
#     'c11': 138.011e9,
#     'c12': 41.763e9,
#     'c13': 13.932e9,
#     'c33': 193.155e9,
#     'c44': 44.96e9
# }

# # error 2
# high_T_el_const = {
#     'c11': 137.938e9,
#     'c12': 41.478e9,
#     'c13': 16.202e9,
#     'c33': 195.439e9,
#     'c44': 45.281e9
# }

# # error 1
# high_T_el_const = {
#     'c11': 138.579e9,
#     'c12': 42.371e9,
#     'c13': 15.151e9,
#     'c33': 194.03e9,
#     'c44': 44.843e9
# }

Mn3Ge = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, manual_indices=manual_indices)
Mn3Ge.do_everything()