import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d




class ElasticConstantsTemperatureDependence:

    # instance attributes
    def __init__(self, folder_path):
        '''
        folder_path: folder where all the individual files containing temperature dependeces of resonance frequencies are stored
        '''

        self.folder_path = folder_path
        self.filenames = self.get_filenames()
        
        self.temperature_raw = []
        self.frequency_raw = []
        self.gamma_raw = []
        for file in self.filenames:
            T, f, g = self.import_data(file)
            self.temperature_raw.append(T)
            self.frequency_raw.append(f)
            self.gamma_raw.append(g)





    def get_filenames (self):
        data_files = os.listdir(self.folder_path)
        data_files = [self.folder_path+'\\'+i for i in data_files if i[-4:]=='.dat']
        return data_files

        
        
    def import_data (self, filepath, number_of_headers=1, number_of_columns=3):
        data = []
        f = open(filepath, 'r')

        for i in np.arange(number_of_headers):
            f.readline()

        for line in f:
            line=line.strip()
            line=line.split()
            for i in np.arange(number_of_columns):
                line[i] = float(line[i])
            data.append(np.array(line[:number_of_columns]))
               
        data = np.array(data).transpose()
        return (data)


    def interpolate (self):
        '''
        this creates interpolated frequency data at the same temperature points in the biggest temperature
        range possible (all frequency curves have to be measured in at least this temperature range);
        the resulting arrays will have points at evenly spaced temperatures with the number of points being the average over all the different resonances
        '''
        Tmin = max([min(t) for t in self.temperature_raw])
        Tmax = min([max(t) for t in self.temperature_raw])

        n = np.mean(np.array([len(t) for t in self.temperature_raw]))
        T_interpolation = np.linspace(Tmin, Tmax, int(n))

        fint = []
        gint = []
        Tint = []

        for i in np.arange(len(self.temperature_raw)):
            fint.append( interp1d(self.temperature_raw[i], self.frequency_raw[i], kind='linear')(T_interpolation) )
            gint.append( interp1d(self.temperature_raw[i], self.gamma_raw[i], kind='linear')(T_interpolation) )
            Tint.append(T_interpolation)

        fint = np.array(fint)
        gint = np.array(gint)
        Tint = np.array(Tint)

        return Tint, fint, gint







# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if __name__ == '__main__':

    folder_path = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\good_data"

    Mn31Sn089 = ElasticConstantsTemperatureDependence(folder_path)

    Tint, fint, gint = Mn31Sn089.interpolate()

    plt.figure()
    for idx, _ in enumerate(Mn31Sn089.frequency_raw):
        plt.scatter(Mn31Sn089.temperature_raw[idx], Mn31Sn089.frequency_raw[idx])
        plt.plot(Tint[idx], fint[idx])


    plt.figure()
    for idx, _ in enumerate(Mn31Sn089.frequency_raw):
        plt.scatter(Mn31Sn089.temperature_raw[idx], Mn31Sn089.gamma_raw[idx])
        plt.plot(Tint[idx], gint[idx])
    plt.show()


        