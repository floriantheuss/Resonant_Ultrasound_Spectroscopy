from elastic_solid.elasticconstantswithtemperature import ElasticConstantsTemperatureDependence



folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\good_data"
# fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\Mn3.1Sn0.89_2010A_out.txt"
fit_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\Mn3.1Sn0.89_2010A_out.txt"

crystal_structure = 'hexagonal'

reference_temperature = 456.6
manual_indices = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14] )

high_T_el_const = {
    'c11': 121,
    'c12': 30.4,
    'c13': 16.1,
    'c33': 145,
    'c44': 42.7,
    # 'c66': 45.3
}

Mn31Sn089 = ElasticConstantsTemperatureDependence(folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature)#, manual_indices=manual_indices)
Mn31Sn089.do_everything()