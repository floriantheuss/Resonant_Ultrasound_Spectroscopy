import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
import json
from copy import deepcopy
from scipy.optimize import leastsq
import sys




class ElasticConstantsTemperatureDependence:

    # instance attributes
    def __init__(self, resonances_names_list, fit_path, crystal_structure, high_T_el_const,
                reference_temperature, interpolation_method='linear', manual_indices=[],a_error_path=None, el_const_save_path=None):
        '''
        folder_path: folder where all the individual files containing temperature dependeces of resonance frequencies are stored
        fit_path: filepath of the _out file which contains the high temperature fit (i.e. the logarithmic derivatives)
        '''

        # initialize attributes
        self.crystal_structure = crystal_structure
        self.high_T_el_const = high_T_el_const

        self.manual_indices = manual_indices

        self.kind = interpolation_method

        # temperature at which the fit of elastic constants is done
        self.reference_temperature = reference_temperature
        
        # import the resonances and widths, as well as temperature
        # self.folder_path = resonances_folder_path
        self.filenames = resonances_names_list
        # self.filenames  = self.get_filenames()
        
        self.temperature_raw = []
        self.frequency_raw = []
        self.gamma_raw = []
        for file in self.filenames:
            T, f, g = self.import_data(file)
            self.temperature_raw.append(T)
            self.frequency_raw.append(f/1e3) # Arkady's labview gives frequencies in kHz; here they are converted to MHz to match the units of the fit_path file
            self.gamma_raw.append(g)

        # import fit result
        self.fit_path           = fit_path
        self.a_error_path       = a_error_path
        self.el_const_save_path = el_const_save_path

        self.CofT_dcit = {}
        self.bulk_modulus = None

        self.CofT_dict = None
        self.dcoc_dict = None
        self.T         = None
        self.dcoc_dict_err_plus  = None
        self.dcoc_dict_err_minus = None


        
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
            fint.append( interp1d(self.temperature_raw[i], self.frequency_raw[i], kind=self.kind)(T_interpolation) )
            gint.append( interp1d(self.temperature_raw[i], self.gamma_raw[i], kind=self.kind)(T_interpolation) )
            Tint.append(T_interpolation)

        fint = np.array(fint)
        gint = np.array(gint)
        Tint = np.array(Tint)

        return Tint, fint, gint

    
    def import_fit_result (self):
        for skip in [15, 16, 17, 18]:
            temp = np.loadtxt(self.fit_path, comments=None, skiprows=skip, max_rows=1, dtype='str')
            if temp[1] == 'data':
                num_res = int(temp[-1])
                break            
        data = np.loadtxt(self.fit_path, dtype="float", comments="#", max_rows=num_res)

        # high temperature resonance frequencies
        fht_exp = data[:, 1]
        fht_calc = data[:, 2]
        # logarithmic derivatives
        dlnf_dlnc = data[:, 5:]
        print()
        print(np.shape(dlnf_dlnc))
        print()
        return fht_exp, fht_calc, dlnf_dlnc


    def find_correct_indices (self, fht_exp):
        if len(self.manual_indices) == 0:
            # idx = np.zeros(len(self.temperature_raw))
            idx_list   = []
            f_ref_list = []
            f_compare  = deepcopy(fht_exp)
            for ii, T in enumerate(self.temperature_raw):
                jj = ( np.arange(len(T))[ abs(T-self.reference_temperature) < 0.5 ] ) [0]
                f_ref = self.frequency_raw[ii][jj]
                f_ref_list.append(f_ref)

                idx_mask  = ( abs(f_compare - f_ref) == min(abs(f_compare - f_ref)) )
                idx_value = int( np.arange(len(f_compare)) [ idx_mask ] [0] ) 
                idx_list.append(idx_value)
                if idx_value < len(fht_exp)-2:
                    f_compare[:idx_value+1] = np.zeros(idx_value+1)
                else:
                    f_compare = np.zeros_like(f_compare)
            idx = np.array(idx_list)
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
        
        return np.array(idx)


    def elastic_constants_LA (self, fint, Tint, idx, fht_exp, dlnf_dlnc, print_output=True):        
        a = dlnf_dlnc[idx]

        if print_output:
            singular_values = np.linalg.svd(a, compute_uv=False)
            print()
            print('------------------------------------------------')
            print('the condition number of the alpha matrix is:')
            print(max(singular_values)/min(singular_values))
            print('------------------------------------------------')
            print()


            print('the total contribution of elastic constants to all used resonances is:')
            print('the numbers are normalized so they add up to 1; fyi 1/',len(self.high_T_el_const),'=', round(1/len(self.high_T_el_const),5))
            print('the second number are the logarithmic derivatives for the first resonance used;')
            print('use these numbers as a check that the order of everything is correct')
            a_sum = np.sum(np.absolute(a), axis=0)
            a_sum = a_sum/np.sum(a_sum)
            ii = 0
            for key in sorted(self.high_T_el_const):
                print(key, ': ', a_sum[ii], '            ', a[0,ii])
                ii+=1

        
        fht = fht_exp[idx]
        
        ChT_array = []
        for key in sorted(self.high_T_el_const):
            ChT_array.append(self.high_T_el_const[key])
        ChT_array = np.array(ChT_array)


        dfof_ht = np.array([(fint[i]-fint[i][-1])/fint[i][-1] for i in np.arange(len(fht))])

        step1 = np.linalg.inv(np.matmul(a.transpose(), a))
        step1[abs(step1)<1e-14] = 0
        step2 = np.matmul(step1, a.transpose())
        step2[abs(step2)<1e-14] = 0
        step3 = 2* np.array([np.matmul(step2, dfof_ht.transpose()[i]) for i in np.arange(len(dfof_ht.T))])
        dcoc_array = step3.transpose()
        CofT_array = np.array([(dcoc_array[i]*ChT_array[i])+ChT_array[i] for i in np.arange(len(ChT_array))])
        

        CofT_dict = {}
        dcoc_dict = {}
        ii = 0
        for key in sorted(self.high_T_el_const):
            CofT_dict[key] = CofT_array[ii]
            dcoc_dict[key] = dcoc_array[ii]
            # print (key, ': ', CofT_array[ii][-1], ' GPa')
            ii += 1

        return dcoc_dict, CofT_dict, Tint[0]


    def residual_function (self, dcoc, dfof_data, alpha):
        # alpha is frequencies along the rows and elastic moduli along the columns
        dcoc_matrix = np.array([dcoc for _ in alpha])
        dfof        = 0.5 * np.sum(alpha * dcoc, axis=1)
        residual    = dfof-dfof_data
        return residual


    def elastic_constants_leastsq (self, fint, Tint, idx, fht_exp, dlnf_dlnc):        
        a   = dlnf_dlnc[idx]       
        fht = fht_exp[idx]
        
        ChT_array = []
        for key in sorted(self.high_T_el_const):
            ChT_array.append(self.high_T_el_const[key])
        ChT_array = np.array(ChT_array)

        # dfof_ht should have frequencies along rows and temperatures along columns
        dfof_ht = np.array([(fint[i]-fint[i][-1])/fint[i][-1] for i in np.arange(len(fht))]).T
        
        dcoc_array = np.zeros((len(dfof_ht), len(a[0])))
        x0 = 0*ChT_array
        # run the for loop from the highest temperatures first, so that dc/c is really 0 at the beginning
        for idx, dfof_data in enumerate(dfof_ht[::-1]):
            result = leastsq(self.residual_function, x0=x0, args=(dfof_data, a))
            dcoc_array[idx] = result[0]
            x0 = result[0]

        dcoc_array = (dcoc_array[::-1]).T
        CofT_array = np.array([(dcoc_array[i]*ChT_array[i])+ChT_array[i] for i in np.arange(len(ChT_array))])
        print(dcoc_array)


        CofT_dict = {}
        dcoc_dict = {}
        ii = 0
        for key in sorted(self.high_T_el_const):
            CofT_dict[key] = CofT_array[ii]
            dcoc_dict[key] = dcoc_array[ii]
            # print (key, ': ', CofT_array[ii][-1], ' GPa')
            ii += 1

        return dcoc_dict, CofT_dict, Tint[0]



    def get_bulk_modulus (self, CofT_dict):
        if self.crystal_structure == 'orthorhombic':
            c11, c22, c33, c12, c13, c23 = np.array(CofT_dict['c11']), np.array(CofT_dict['c22']), np.array(CofT_dict['c33']), np.array(CofT_dict['c12']), np.array(CofT_dict['c13']), np.array(CofT_dict['c23'])
            bulk = (c13**2*c22 - 2*c12*c13*c23 + c12**2*c33 + c11*(c23**2 - c22*c33)) / (c12**2 + c13**2 - c11*c22 + 2*c13*(c22 - c23) + 2*c11*c23 + c23**2 - 2*c12*(c13 + c23 - c33) - (c11 + c22)*c33)
        else:
            print ('currently the bulk modulus function is only implemented for orhtorhombic crystals')
            bulk = None
        return bulk



    def plot_data (self, fint, Tint):
        plt.figure()
        for ii, _ in enumerate(fint):
            # plt.scatter(self.temperature_raw[ii], self.frequency_raw[ii])
            plt.plot(self.temperature_raw[ii], self.frequency_raw[ii], 'o-')
            # plt.plot(Tint[ii], (f-max(f))/(max(f)-min(f)) + ii)
        
        plt.xlim(min(Tint[0]), max(Tint[0]))
        plt.xlabel('Temperature (K)')
        plt.ylabel('Frequency (MHz)')

        plt.figure()
        for ii, _ in enumerate(fint):
            dfdT = np.gradient(fint[ii], Tint[ii])
            plt.plot(Tint[ii], (dfdT-max(dfdT))/(max(dfdT)-min(dfdT)), 'o-')# + ii, 'o-')
        
        plt.xlim(min(Tint[0]), max(Tint[0]))
        plt.xlabel('Temperature (K)')
        plt.ylabel('df/dT (arb. units')

    
    def save_data (self, C_dict, T, save_path, type):
        save_data      = np.zeros((len(T), len(C_dict)+1))
        save_data[:,0] = T
        header         = 'T (K)'

        idx = 1
        for key, item in C_dict.items():
            save_data[:,idx] = item
            if type=='dcoc':
                header += ',d'+key+'/'+key
            elif type=='coT':
                header += ','+key+' (GPa)'
            idx+=1
        np.savetxt(save_path, save_data, header=header, delimiter=',')
        return (1)

    
    def shift_one_frequency(self, fint, idx, idx_shift, shift):
        print(idx_shift, shift)
        if idx_shift in idx:
            temp_idx = np.arange(len(idx))[abs(idx-idx_shift)<1e-5][0]
            print(fint[temp_idx][0])
            fint[temp_idx] = fint[temp_idx] + shift
            print(fint[temp_idx][0])
            print()
        return fint

    

    def analyze_data (self, plot_data=False, err_idx=None, err=0, method='LinearAlgebra'):
        Tint, fint, _                = self.interpolate()
        fht_exp, fht_calc, dlnf_dlnc = self.import_fit_result()
        idx                          = self.find_correct_indices(fht_exp)
        # print ('correct indices are: ', idx)

        if err_idx is not None:
            fint = self.shift_one_frequency(fint, idx, err_idx, err)

        if method == 'LinearAlgebra':
            self.dcoc_dict, self.CofT_dict, self.T = self.elastic_constants_LA(fint, Tint, idx, fht_exp, dlnf_dlnc)
            if self.a_error_path is not None:
                a_err = np.loadtxt(self.a_error_path)
                if (np.shape(a_err)[0]<np.shape(dlnf_dlnc)[0]):
                    print(f'the file for alpha errors only includes {np.shape(a_err)[0]} resonances, but we need {np.shape(dlnf_dlnc)[0]}')
                elif (np.shape(a_err)[1]<np.shape(dlnf_dlnc)[1]):
                    print(f'the file for alpha errors only includes {np.shape(a_err)[1]} elastic constants, but we need {np.shape(dlnf_dlnc)[1]}')
                else:
                    a_err = a_err[:np.shape(dlnf_dlnc)[0],:np.shape(dlnf_dlnc)[1]]
                    self.dcoc_dict_err_plus, _, _  = self.elastic_constants_LA(fint, Tint, idx, fht_exp, dlnf_dlnc+a_err, print_output=False)
                    self.dcoc_dict_err_minus, _, _ = self.elastic_constants_LA(fint, Tint, idx, fht_exp, dlnf_dlnc-a_err, print_output=False)

        elif method == 'leastsq':
            self.dcoc_dict, self.CofT_dict, self.T = self.elastic_constants_leastsq(fint, Tint, idx, fht_exp, dlnf_dlnc)
        else:
            print()
            print('your chosen method is not implemented')
            print('please choose between "LinearAlebra" or "leastsq"')
            print()
            sys.exit()

        self.bulk_modulus = self.get_bulk_modulus(self.CofT_dict)

        if self.el_const_save_path is not None:
            self.save_data(self.dcoc_dict, self.T, self.el_const_save_path, type='dcoc')
        if plot_data:
            self.plot_data(fint, Tint)
        




