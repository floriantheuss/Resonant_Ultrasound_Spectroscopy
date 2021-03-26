import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
import json




class ElasticConstantsTemperatureDependence:

    # instance attributes
    def __init__(self, resonances_folder_path, fit_path, crystal_structure, high_T_el_const, reference_temperature, manual_indices=[]):
        '''
        folder_path: folder where all the individual files containing temperature dependeces of resonance frequencies are stored
        '''

        # initialize attributes
        self.crystal_structure = crystal_structure
        self.high_T_el_const = high_T_el_const

        self.manual_indices = manual_indices

        # temperature at which the fit of elastic constants is done
        self.reference_temperature = reference_temperature
        
        # import the resonances and widths, as well as temperature
        self.folder_path = resonances_folder_path
        self.filenames = self.get_filenames()
        
        self.temperature_raw = []
        self.frequency_raw = []
        self.gamma_raw = []
        for file in self.filenames:
            T, f, g = self.import_data(file)
            self.temperature_raw.append(T)
            self.frequency_raw.append(f/1e3) # in MHz
            self.gamma_raw.append(g)

        # import fit result
        self.fit_path = fit_path





    def get_filenames (self):
        '''
        get the filenames of all files in the specified folder
        '''
        data_files = os.listdir(self.folder_path)
        data_files = [self.folder_path+'\\'+i for i in data_files if i[-4:]=='.dat']
        for i in data_files:
            print (i)
        return data_files

        
        
    def import_data (self, filepath, number_of_headers=1, number_of_columns=3):
        '''
        import temperature dependence of frequency and resonance width of one resonance with filename filepath
        '''

        data = []
        f = open(filepath, 'r')

        for i in np.arange(number_of_headers):
            f.readline()

        for line in f:
            line=line.strip()
            line=line.split()
            number = min([number_of_columns, len(line)])
            for i in np.arange(number):
                line[i] = float(line[i])
            if len(line) >= number_of_columns:
                data.append(np.array(line[:number_of_columns]))
            else:
                n_columns = len(line)
                difference = number_of_columns - n_columns
                data.append(np.array(line + difference*[0]))
               
        T, f, g = np.array(data).transpose()
        # for idx, _ in enumerate(T):
        #     f[idx] = np.array(f)[idx][np.argsort(T[idx])]
        #     g[idx] = g[idx][np.argsort(T[idx])]
        #     T[idx] = np.sort(T[idx])
        return (T, f, g)


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
        Tint = []
        gint = []

        for i in np.arange(len(self.temperature_raw)):
            fint.append( interp1d(self.temperature_raw[i], self.frequency_raw[i], kind='linear')(T_interpolation) )
            gint.append( interp1d(self.temperature_raw[i], self.gamma_raw[i], kind='linear')(T_interpolation) )
            Tint.append(T_interpolation)

        fint = np.array(fint)
        gint = np.array(gint)
        Tint = np.array(Tint)

        return Tint, fint, gint

    
    def import_fit_result (self):
        data = np.loadtxt(self.fit_path, dtype="float", comments="#")

        # high temperature resonance frequencies
        fht_exp = data[:, 1]
        fht_calc = data[:, 2]
        # logarithmic derivatives
        dlnf_dlnc = data[:, 4:]
        return fht_exp, fht_calc, dlnf_dlnc


    def find_correct_indices (self, fht_exp):
        if len(self.manual_indices) == 0:
            # idx = np.zeros(len(self.temperature_raw))
            idx = []
            f_ref_list = []
            for ii, T in enumerate(self.temperature_raw):
                jj = ( np.arange(len(T))[ abs(T-self.reference_temperature) < 0.5 ] ) [0]
                f_ref = self.frequency_raw[ii][jj]
                f_ref_list.append(f_ref)

                idx_mask = ( abs(fht_exp - f_ref) == min(abs(fht_exp - f_ref)) )
                idx.append(int( np.arange(len(fht_exp)) [ idx_mask ] [0] ))
            idx = np.array(idx)
        else:
            idx = self.manual_indices
            f_ref_list = []
            for ii, T in enumerate(self.temperature_raw):
                jj = ( np.arange(len(T))[ abs(T-self.reference_temperature) < 0.5 ] ) [0]
                f_ref = self.frequency_raw[ii][jj]
                f_ref_list.append(f_ref)

        if len (fht_exp) < max(idx)+3:
            nb = len (fht_exp)
        else:
            nb = max(idx)+3
        jj = 0
        for ii in np.arange(nb):
            if ii in idx:
                print (ii, fht_exp[ii], f_ref_list[jj])
                jj+=1
            else:
                print (ii, fht_exp[ii])
        
        return idx


    def elastic_constants(self, fint, Tint):

        fht_exp, fht_calc, dlnf_dlnc = self.import_fit_result()
        idx = self.find_correct_indices(fht_exp)
        
        a = dlnf_dlnc[idx]
        fht = fht_exp[idx]
        
        ChT_array = []
        for key in sorted(self.high_T_el_const):
            ChT_array.append(self.high_T_el_const[key])
        ChT_array = np.array(ChT_array)


        dfof_ht = np.array([(fint[i]-fht[i])/fht[i] for i in np.arange(len(fht))])

        step1 = np.linalg.inv(np.matmul(a.transpose(), a))
        step1[abs(step1)<1e-14] = 0
        step2 = np.matmul(step1, a.transpose())
        step2[abs(step2)<1e-14] = 0
        step3 = 2* np.array([np.matmul(step2, dfof_ht.transpose()[i]) for i in np.arange(len(dfof_ht.transpose()))])

        CofT_array = np.array([(step3.transpose()[i]*ChT_array[i])+ChT_array[i] for i in np.arange(len(ChT_array))])

        CofT_dict = {}
        ii = 0
        for key in sorted(self.high_T_el_const):
            CofT_dict[key] = CofT_array[ii]
            ii += 1

        print (dlnf_dlnc)

        return CofT_dict, Tint


    def get_irreps (self, fint, Tint):
        CofT_dict, T = self.elastic_constants(fint, Tint)
        C_irrep = {}
        if self.crystal_structure == 'hexagonal':
            if 'c11' in self.high_T_el_const.keys():
                C_irrep['A1g1']   =   ( CofT_dict['c11'] + CofT_dict['c12'] ) / 2
                C_irrep['E2g']    =   ( CofT_dict['c11'] - CofT_dict['c12'] ) / 2
            elif 'c66' in self.high_T_el_const.keys():
                C_irrep['A1g1']   =   ( CofT_dict['c66'] + CofT_dict['c12'] )
                C_irrep['E2g']    =   CofT_dict['c66']
            else:   
                print ('A value for either c11 or c66 has to be given at high temperature')
            C_irrep['A1g2']   =   CofT_dict['c33']
            C_irrep['A1g3']   =   CofT_dict['c13']
            C_irrep['E1g']    =   CofT_dict['c44']
            

        dC_irrep = {key: item-item[-1] for key, item in C_irrep.items()}

        return C_irrep, dC_irrep, T[0]


    def plot_irreps (self, C, T, ylabel):
        if self.crystal_structure == 'hexagonal':
            plt.figure()
            plt.plot(T, C['A1g1'], label='A1g1')
            plt.plot(T, C['A1g2'], label='A1g2')
            plt.plot(T, C['A1g3'], label='A1g3')
            plt.legend()
            plt.xlabel('T (K)',fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            # plt.ylabel('$\mathrm{C_{irrep}}$ (GPa) ',fontsize=15)
            plt.tick_params(axis="both",direction="in", labelsize=15, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')

            plt.figure()
            plt.plot(T, C['E1g'], label='E1g')
            plt.plot(T, C['E2g'], label='E2g')
            plt.legend()
            plt.xlabel('T (K)',fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            plt.tick_params(axis="both",direction="in", labelsize=15, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')

    def plot_data (self, fint, Tint):
        plt.figure()
        for ii, f in enumerate(fint):
            # plt.scatter(self.temperature_raw[ii], self.frequency_raw[ii])
            plt.plot(self.temperature_raw[ii], self.frequency_raw[ii], 'o-')
            # plt.plot(Tint[ii], (f-max(f))/(max(f)-min(f)) + ii)
        
        plt.xlim(min(Tint[0]), max(Tint[0]))
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency (MHz)')

        plt.figure()
        for ii, _ in enumerate(fint):
            dfdT = np.gradient(fint[ii], Tint[ii])
            plt.plot(Tint[ii], (dfdT-max(dfdT))/(max(dfdT)-min(dfdT)) + ii, 'o-')
        
        plt.xlim(min(Tint[0]), max(Tint[0]))
        plt.xlabel('Temperature (K)')
        plt.ylabel('df/dT (arb. units')

    
    def save_data (self, C_dict, T, save_path):
        c_save = {}
        for key, item in C_dict.items():
            c_save[key] = list(item)
        report = {
            'elastic constants': c_save,
            'temperature': list(T)
        }
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        return (1)

    

    def do_everything (self):
        Tint, fint, gint = self.interpolate()
        C_irrep, dC_irrep, T = self.get_irreps(fint, Tint)

        self.save_data(C_irrep, T, self.fit_path[:-4]+'_elastic_constants.json')
        
        self.plot_data(fint, Tint)
        self.plot_irreps (dC_irrep, T, '$\\Delta \\mathrm{c}$ (GPa) ')
        plt.show()
        




