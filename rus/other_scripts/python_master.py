import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt
from scipy.optimize import curve_fit
from scipy import odr


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# class data_extract
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
class data_extract:
    
    # instance attributes
    def __init__(self, filepath):
        self.path = filepath # the attribute filepath gives the location of the file to import

    # define the function which actually imports the data
    def import_data (self, number_of_lines_to_skip=0, number_of_columns='all'):

        unsorted_data = []
        f = open(self.path, 'r')

        for i in np.arange(number_of_lines_to_skip):
            f.readline()

        if number_of_columns == 'all':
            for line in f:
                line = line.strip()
                line = line.split()
                for i in np.arange(len(line)):
                    line[i] = float(line[i])
                unsorted_data.append(line)
        else:
            for line in f:
                line=line.strip()
                line=line.split()
                for i in np.arange(number_of_columns):
                    line[i] = float(line[i])
                unsorted_data.append(line[:number_of_columns])
        
        #unsorted_data.sort(key = lambda x:x[0])     # this sorts all data files according to temperature    
        data = np.array(unsorted_data).transpose()

        return (data)






# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# class frequency
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
class frequency:

    # instance attributes
    def __init__(self, T):
        self.temperature = T # the attribute temperature gives the temperature array for the frequency