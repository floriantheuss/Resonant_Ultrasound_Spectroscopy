from rus.old.elasticconstantswithtemperature_old import ElasticConstantsTemperatureDependence
import numpy as np


folder_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.019Sn0.981\\RUS\\2007A\\good_data'
fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.019Sn0.981\\RUS\\2007A\\fit_report_Mn3.019Sn0.981_2207A_high_T_rpr_fit_68_resonances_least_squares_lmfit.txt'

crystal_structure = 'hexagonal'

reference_temperature = 435


# mean
high_T_el_const = {
    'c11': 130.848e9,
    'c12': 28.478e9,
    'c13': 16.953e9,
    'c33': 151.303e9,
    'c44': 48.107e9
}

manual_indices = [1,2,4,5,7,8,9,11,13,14,15,20,21,22,29,33,34]

Mn3Ge = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear', manual_indices=manual_indices)
# Mn3Ge = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, interpolation_method='linear')
Mn3Ge.do_everything()