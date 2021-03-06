import matplotlib.pyplot as plt
from fitcriticalexponent import CriticalExponentFit


# folder = "C:\\Users\\Florian\\Box Sync\\Projects"
# project = "\\Mn3Ge\\RUS\\Mn3Ge_2001B\\irreducible_elastic_constants_with_error.txt"
# filepath = folder+project

# Mn3Ge bulk modulus >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# save_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3Ge\\RUS\\Mn3Ge_2001B\\Mn3Ge_critical_exponent_bulk_modulus_fit_result.json'

# elastic_constant_to_fit = 'Bulk'
# include_errors = True
# fit_algorithm = 'odr' #options are 'lmfit' or 'odr'

# initial_conditions = {
#     'Tc': {'initial_value':369.9, 'bounds':[368, 371], 'vary':True},
#     'alpha': {'initial_value':-0.0121, 'bounds':[-0.8, 0.8], 'vary':True},
#     'delta': {'initial_value':0.529, 'bounds':[0, 1], 'vary':False},
#     'Am': {'initial_value':118.53, 'bounds':[0, 200], 'vary':True},
#     'Ap': {'initial_value':132.67, 'bounds':[0, 200], 'vary':True},
#     'Bm': {'initial_value':0.34, 'bounds':[-2, 2], 'vary':True},
#     'Bp': {'initial_value':0, 'bounds':[-1, 1], 'vary':False}
#     }

# fit_ranges = {
#     'background':{'Tmin':390, 'Tmax':500},
#     'critical_exponent':{'T1':352, 'T2':367.9, 'T3':370.36, 'T4':396}
#     }


# Mn3Ge shear modulus >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# save_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3Ge\\RUS\\Mn3Ge_2001B\\Mn3Ge_critical_exponent_E2g_fit_result.json'

# elastic_constant_to_fit = 'E2g'
# include_errors = False

# initial_conditions = {
#     'Tc': {'initial_value':369.9, 'bounds':[368, 371], 'vary':True},
#     'alpha': {'initial_value':0.35, 'bounds':[0, 0.8], 'vary':True},
#     'delta': {'initial_value':0, 'bounds':[0, 100], 'vary':False},
#     'A': {'initial_value':-2, 'bounds':[-10, 0], 'vary':True},
#     'B': {'initial_value':0, 'bounds':[-50, 0], 'vary':False}
#     }

# fit_ranges = {
#     'background':{'Tmin':390, 'Tmax':500},
#     'critical_exponent':{'Tmin':370.5, 'Tmax':500}
#     }



# Mn3.019Sn0.981 bulk modulus >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
folder = "C:\\Users\\Florian\\Box Sync\\Projects"
project = "\\Mn3.019Sn0.981\\RUS\\2007A\\irreducible_elastic_constants.txt"
filepath = folder+project

save_path = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3.019Sn0.981\\RUS\\2007A\\Mn3.019Sn0.981_critical_exponent_bulk_modulus_fit_result.json'

elastic_constant_to_fit = 'Bulk'
include_errors = False
fit_algorithm = 'odr' #options are 'lmfit' or 'odr'

initial_conditions = {
    'Tc': {'initial_value':416, 'bounds':[414, 418], 'vary':True},
    'alpha': {'initial_value':-0.0121, 'bounds':[-0.8, 0.8], 'vary':True},
    'delta': {'initial_value':0.529, 'bounds':[0, 1], 'vary':False},
    'Am': {'initial_value':118.53, 'bounds':[0, 200], 'vary':True},
    'Ap': {'initial_value':132.67, 'bounds':[0, 200], 'vary':True},
    'Bm': {'initial_value':0.34, 'bounds':[-2, 2], 'vary':True},
    'Bp': {'initial_value':0, 'bounds':[-1, 1], 'vary':False}
    }

fit_ranges = {
    'background':{'Tmin':430, 'Tmax':500},
    'critical_exponent':{'T1':382, 'T2':410, 'T3':415.8, 'T4':450}
    }

# run the fit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

fit = CriticalExponentFit(filepath, initial_conditions, fit_ranges, elastic_constant_to_fit, include_errors, fit_algorithm, save_path)
fit.fit()

