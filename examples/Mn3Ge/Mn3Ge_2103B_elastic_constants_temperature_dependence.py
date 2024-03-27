from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence
import numpy as np


folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2103B\\good_data"
fit_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2103B\\high_temperature_scan\\210329\\Mn3Ge_2103B_out.txt'

crystal_structure = 'hexagonal'

reference_temperature = 440


# mean
high_T_el_const = {
    'c11': 140.077e9,
    'c12': 44.778e9,
    'c13': 17.387e9,
    'c33': 194.724e9,
    'c44': 44.404e9
}

Mn3Ge = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature)#, manual_indices=manual_indices)
Mn3Ge.do_everything()