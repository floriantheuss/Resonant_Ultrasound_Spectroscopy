import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
import json
from copy import deepcopy
from scipy.optimize import leastsq
from numpy.polynomial import polynomial
import sys
from copy import deepcopy




class ElasticConstantsTemperatureDependence:

    # instance attributes
    def __init__(self, resonances_names_list, fit_path, crystal_structure, 
                high_T_el_const, reference_temperature, interpolation_method='linear',
                manual_indices=[], el_const_save_path=None, fit_error_list=[]):
        '''
        folder_path: folder where all the individual files containing temperature dependeces of resonance frequencies are stored
        fit_path: filepath of the _out file which contains the high temperature fit (i.e. the logarithmic derivatives)
        '''

        # initialize attributes
        self.crystal_structure = crystal_structure
        self.high_T_el_const   = high_T_el_const

        self.manual_indices = manual_indices

        self.kind = interpolation_method

        # temperature at which the fit of elastic constants is done
        self.reference_temperature = reference_temperature
        
        # import the resonances and widths, as well as temperature
        # self.folder_path = resonances_folder_path
        self.filenames = resonances_names_list
        if (len(self.filenames) != len(self.manual_indices)) & (len(manual_indices)>0):
            print()
            print("the number of included resonances is not the same")
            print("as the number of entries in manual_indices")
            print()
            sys.exit()
        
        self.temperature_raw = []
        self.frequency_raw = []
        self.gamma_raw = []
        for file in self.filenames:
            T, f, g = self.import_data(file)
            self.temperature_raw.append(T)
            self.frequency_raw.append(f/1e3) # Arkady's labview gives frequencies in kHz; here they are converted to MHz to match the units of the fit_path file
            self.gamma_raw.append(g)

        # import fit result
        self.fit_path            = fit_path
        self.fit_error_list      = fit_error_list
        self.el_const_save_path  = el_const_save_path
        self.meta_text_save_path = None
        if el_const_save_path is not None:
            temp = el_const_save_path.split('.')
            temp[-2] = temp[-2]+ '-meta_data'
            self.meta_text_save_path = '.'.join(temp)

        # create a string with meta information about the decomposition
        self.meta_text = 'used fit report:\n'
        self.meta_text+= fit_path + '\n\n'
        self.meta_text+= 'used resonances with according index assigning it a line in the fit report:\n'
        for ii, _ in enumerate(manual_indices):
            self.meta_text+= str(manual_indices[ii]) + '       ' + resonances_names_list[ii] + '\n'
        self.save_meta_text()


        self.CofT_dcit = {}
        self.bulk_modulus = None

        self.CofT_dict       = None
        self.dcoc_dict       = None
        self.dcoc_error_dict = None
        self.T               = None
        # dictionary which will be filled with the absolute jump sizes of the elastic moduli at Tc
        self.jumps_dict = {}
        self.bulk_jump  = None
        # dictionary which will be filled with the relative jump sizes of the elastic moduli at Tc
        self.relative_jumps_dict = {}


        
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

    def save_meta_text (self):
        if self.meta_text_save_path is not None:
            text_file = open(self.meta_text_save_path, "w")
            text_file.write(self.meta_text)
            text_file.close()



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

    
    def import_fit_result (self, fit_path):
        for skip in [15, 16, 17, 18]:
            temp = np.loadtxt(fit_path, comments=None, skiprows=skip, max_rows=1, dtype='str')
            if temp[1] == 'data':
                num_res = int(temp[-1])
                data = np.loadtxt(fit_path, dtype="float", comments="#", max_rows=num_res)            
            else:
                data = np.loadtxt(fit_path, dtype="float", comments="#")

        # high temperature resonance frequencies
        fht_exp = data[:, 1]
        fht_calc = data[:, 2]
        # logarithmic derivatives
        dlnf_dlnc = data[:, -len(self.high_T_el_const):]
        print()
        print(np.shape(dlnf_dlnc))
        print()
        return fht_exp, fht_calc, dlnf_dlnc


    def find_correct_indices (self, fht_exp, print_indices=True):
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
            if print_indices:
                print(idx)
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
        if print_indices:
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
            singular_values  = np.linalg.svd(a, compute_uv=False)
            condition_number = max(singular_values)/min(singular_values)
            text = '\n------------------------------------------------\n\n'
            text+= 'the condition number of the alpha matrix is:\n'
            text+= str(condition_number) + '\n\n'
            text+= '------------------------------------------------\n'
            print(text)

            self.meta_text = text + self.meta_text
            self.save_meta_text()


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


        t1 = abs(min(self.temperature_raw[0]) - self.reference_temperature)
        t2 = abs(max(self.temperature_raw[0]) - self.reference_temperature)
        if t1>t2:
            dfof_ht = np.array([(fint[i]-fint[i][-1])/fint[i][-1] for i in np.arange(len(fht))])
        else:
            dfof_ht = np.array([(fint[i]-fint[i][0])/fint[i][0] for i in np.arange(len(fht))])

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
            print()
            print()
            print ('currently, "get_bulk_modulus" is only implemented for orhtorhombic crystals')
            print()
            print()
            bulk = None
        return bulk



    def plot_data (self, T, f, derivative_plot_bounds=[], Tc=None):
        # plot df/f for all frequencies used in the decomposition
        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.18, right = 0.8, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size
        for ii,freq in enumerate(f):
            mask  = np.argsort(T[ii])
            f_ind = freq[mask]
            T_ind = T[ii][mask]
            axes.plot(T_ind, (f_ind-f_ind[-1])/f_ind[-1]*1e5+ii*0.2, label=str(np.round(f_ind[-1],2)))
        plt.legend(loc=(1.02,-0.2), fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
        axes.set_xlabel(r"$T$ ( K )", labelpad = 8)
        axes.set_ylabel(r"$\Delta f /f$ ( $10^{-5}$ )", labelpad = 8)
        
        # plot the derivative of all resonances used in the decomposition
        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.18, right = 0.8, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size
        miny = 1e10
        maxy = -1e10
        factor = 1e6 # this is just a scaling factor to use to make the plot look nicer
        for ii,freq in enumerate(f):
            mask   = np.argsort(T[ii])
            f_ind  = freq[mask]
            T_ind  = T[ii][mask]
            df_ind = np.gradient(f_ind)
            x, y   = self.moving_average(T_ind, df_ind, 10)
            if len(derivative_plot_bounds) == 2: 
                mask   = (x>derivative_plot_bounds[0])&(x<derivative_plot_bounds[1])
                axes.plot(x[mask], y[mask]*factor, label=str(np.round(f_ind[-1],2)))
                if min(y)<miny:
                    miny=min(y)
                if max(y)>maxy:
                    maxy = max(y)
            else:
                axes.plot(x, y*factor, label=str(np.round(f_ind[-1],2)))
                if min(y)<miny:
                    miny=min(y)
                if max(y)>maxy:
                    maxy = max(y)
        
        if Tc is not None:
            color = 'black'
            line = axes.plot([Tc, Tc], [miny*factor, maxy*factor])
            plt.setp(line, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
        
        plt.legend(loc=(1.02,-0.2), fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
        axes.set_xlabel(r"$T$ ( K )", labelpad = 8)
        axes.set_ylabel(f"$\\partial f / \\partial T$ ( {1/factor} $\\times$ Hz/K )", labelpad = 8)

        return 1

    
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


    def get_dcoc (self, Tint, fint, fit_path, method, fit_path_uncertainty=None):
        fht_exp, fht_calc, dlnf_dlnc = self.import_fit_result(fit_path)
        idx                          = self.find_correct_indices(fht_exp)
        if fit_path_uncertainty is not None:
            _, _, dlnf_dlnc = self.import_fit_result(fit_path_uncertainty)

        if method == 'LinearAlgebra':
            dcoc_dict, CofT_dict, T = self.elastic_constants_LA(fint, Tint, idx, fht_exp, dlnf_dlnc)
        elif method == 'leastsq':
            dcoc_dict, CofT_dict, T = self.elastic_constants_leastsq(fint, Tint, idx, fht_exp, dlnf_dlnc)
        else:
            print()
            print('your chosen method is not implemented')
            print('please choose between "LinearAlebra" or "leastsq"')
            print()
            sys.exit()

        return dcoc_dict, CofT_dict, T

    def moving_average (self, x, y, n):
        ret = np.cumsum(y, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return x[n-1:], ret[n - 1:] / n
    

    def fit_jumps (self, T, CofT_dict, Tc, Tfit_high, Tfit_low, N=10):
        fits1_dict = {} 
        fits2_dict = {} 
        for el in np.array(['c11', 'c22', 'c33','c12', 'c13', 'c23','c55', 'c66', 'c44']):
            if el in CofT_dict:
                dat_fit        = np.array(CofT_dict[el])
                mask1          = (T>Tfit_high[0])&(T<Tfit_high[1])
                x, y           = self.moving_average(T[mask1], dat_fit[mask1], N)  
                fit1           = polynomial.polyfit(x, y, 1)
                mask2          = (T>Tfit_low[0])&(T<Tfit_low[1])
                x, y           = self.moving_average(T[mask2], dat_fit[mask2], N)  
                fit2           = polynomial.polyfit(x, y, 1)

                fits1_dict[el] = [np.linspace(Tc, max(T), 100), polynomial.polyval(np.linspace(Tc, max(T), 100), fit1)]
                fits2_dict[el] = [np.linspace(min(T), Tc, 100), polynomial.polyval(np.linspace(min(T), Tc, 100), fit2)]

                self.jumps_dict[el] = (polynomial.polyval(Tc, fit1) - polynomial.polyval(Tc, fit2))
                self.relative_jumps_dict[el] = self.jumps_dict[el] / dat_fit[-1]
        
        return fits1_dict, fits2_dict
    
    def fit_bulk_jump (self, T, bulk, Tc, Tfit_high, Tfit_low, N=10):
        dat_fit        = bulk
        mask1          = (T>Tfit_high[0])&(T<Tfit_high[1])
        x, y           = self.moving_average(T[mask1], dat_fit[mask1], N)  
        fit1           = polynomial.polyfit(x, y, 1)
        fit1_list      = [np.linspace(Tc, max(T), 100), polynomial.polyval(np.linspace(Tc, max(T), 100), fit1)]
        mask2          = (T>Tfit_low[0])&(T<Tfit_low[1])
        x, y           = self.moving_average(T[mask2], dat_fit[mask2], N)  
        fit2           = polynomial.polyfit(x, y, 1)
        fit2_list      = [np.linspace(min(T), Tc, 100), polynomial.polyval(np.linspace(min(T), Tc, 100), fit2)]

        bulk_jump = (polynomial.polyval(Tc, fit1) - polynomial.polyval(Tc, fit2))
        self.bulk_jump = bulk_jump
        return bulk_jump, fit1_list, fit2_list


    def plot_dbulk (self, T=None, bulk=None, moving_average=1, markevery=1, pdfpages=None, Tc=None, Tfit_high=None, Tfit_low=None, y_bounds=None):
        if T is None:
            T = self.T
        if bulk is None:
            bulk = self.bulk_modulus
        
        dbulk = (bulk-bulk[-1])/bulk[-1]
        x,y = self.moving_average(T, dbulk, moving_average)

        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

        if y_bounds is not None:
            axes.set_ylim(y_bounds[0], y_bounds[1])

        axes.set_xlabel(r"$T$ ( K )", labelpad = 8)
        axes.set_ylabel(r"$\Delta B /B$ ( $10^{-5}$ )", labelpad = 8)

        line = axes.plot(x, y*1e5, markevery=markevery)
        color = 'black'
        plt.setp(line, ls ="", c = color, lw = 2, marker = "o", mfc = color, ms = 7, mec = color, mew= 2)
        
        if (Tc is not None) and (Tfit_high is not None) and (Tfit_low is not None):
            bulk_jump, fit1_list, fit2_list = self.fit_bulk_jump (T, bulk, Tc=Tc, Tfit_high=Tfit_high, Tfit_low=Tfit_low, N=moving_average)

            line = axes.plot(fit1_list[0], (fit1_list[1]-bulk[-1])/bulk[-1]*1e5, markevery=markevery)
            color = 'red'
            plt.setp(line, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
            line = axes.plot(fit2_list[0], (fit2_list[1]-bulk[-1])/bulk[-1]*1e5, markevery=markevery)
            color = 'red'
            plt.setp(line, ls ="--", c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)

            axes.axvline(Tfit_high[0], ls='--', lw=2, color='gray')
            axes.axvline(Tfit_high[1], ls='--', lw=2, color='gray')
            axes.axvline(Tfit_low[0], ls='--', lw=2, color='gray')
            axes.axvline(Tfit_low[1], ls='--', lw=2, color='gray')


        if pdfpages is not None:
            pdfpages.savefig(bbox_inches='tight')


    def plot_dcoc (self, T, dcoc_dict, CofT_dict=None, dcoc_error_dict=None, N=10, colors=['#0a9396', '#ee9b00', '#ca6702'], Tc=None, Tfit_high=None, Tfit_low=None, pdfpages=None, plotstyle='points', markevery=1, plot_jump_fits=True):

        if (Tc is not None) and (Tfit_high is not None) and (Tfit_low is not None) and (plot_jump_fits==True):
            fits1_dict, fits2_dict = self.fit_jumps (T, CofT_dict, Tc, Tfit_high, Tfit_low, N=N)

        for el_const_list in np.array([['c11', 'c22', 'c33'], ['c12', 'c13', 'c23'], ['c55', 'c66', 'c44']]):
        
            fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
            fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

            extreme_y_values = [] 

            idx = 0
            for el in np.array(el_const_list):
                if el in dcoc_dict:
                    color = colors[idx]
                    dat_plot = np.array(dcoc_dict[el])
                    x, y = self.moving_average(T, dat_plot*1e5, N)
                    line = axes.plot(x, y, label=el, markevery=markevery)
                    if plotstyle=='points':
                        plt.setp(line, ls ="", c = color, lw = 2, marker = "o", mfc = color, ms = 7, mec = color, mew= 2)
                    elif plotstyle=='line':
                        plt.setp(line, ls ='-', c = color, lw = 2, marker = "", mfc = color, ms = 7, mec = color, mew= 2)
                    
                    extreme_y_values.append(np.array([max(y), min(y)]))

                    # plot error bars
                    if dcoc_error_dict is not None:
                        x1, y1 = self.moving_average(T, (dat_plot+dcoc_error_dict[el])*1e5, N)
                        x2, y2 = self.moving_average(T, (dat_plot-dcoc_error_dict[el])*1e5, N)
                        line = axes.fill_between(x1, y1, y2)
                        plt.setp(line, ls ="-", color = color, lw = 0, fc = color, ec = color, alpha=0.3)


                    if (Tc is not None) and (Tfit_high is not None) and (Tfit_low is not None) and (plot_jump_fits==True):
                        ref_value = CofT_dict[el][-1]

                        x, y = fits1_dict[el][0], fits1_dict[el][1]
                        axes.plot(x, (y-ref_value)/ref_value*1e5, '--', color='black')
                        extreme_y_values.append(np.array([max((y-ref_value)/ref_value*1e5), min((y-ref_value)/ref_value*1e5)]))

                        x, y = fits2_dict[el][0], fits2_dict[el][1]
                        axes.plot(x, (y-ref_value)/ref_value*1e5, '--', color='black')
                        extreme_y_values.append(np.array([max((y-ref_value)/ref_value*1e5), min((y-ref_value)/ref_value*1e5)]))

                    idx+=1

            if (Tc is not None) and (Tfit_high is not None) and (Tfit_low is not None) and (plot_jump_fits==True):
                axes.axvline(Tfit_high[0], ls='--', lw=2, color='gray')
                axes.axvline(Tfit_high[1], ls='--', lw=2, color='gray')
                axes.axvline(Tfit_low[0], ls='--', lw=2, color='gray')
                axes.axvline(Tfit_low[1], ls='--', lw=2, color='gray')

            plt.legend(fontsize = 16, frameon = False, numpoints=1, markerscale = 1, handletextpad=0.5)
            axes.set_xlabel(r"$T$ ( K )", labelpad = 8)
            axes.set_ylabel(r"$\Delta c /c$ ( $10^{-5}$ )", labelpad = 8)
            # axes.set_ylabel(r"$\Delta c$ ( GPa )", labelpad = 8)

            if pdfpages is not None:
                pdfpages.savefig(bbox_inches='tight')

        return self.jumps_dict

    def plot_jumps (self, color = '#ca6702', pdfpages=None, ylimits=None):
        jumps_dict = self.jumps_dict
        if len(jumps_dict)>0:
            jumps_dict = self.jumps_dict
            keys = np.array(['c11', 'c22', 'c33', 'c12', 'c13', 'c23', 'c44', 'c55', 'c66'])
            jumps_array  = np.zeros(len(jumps_dict))
            xticks_array = np.zeros(len(jumps_dict), dtype=str)
            x_ticks_all  = np.array(['$c_{11}$', '$c_{22}$', '$c_{33}$', '$c_{12}$', '$c_{13}$', '$c_{23}$', '$c_{44}$', '$c_{55}$', '$c_{66}$'])
            idx = 0
            for ii, key in enumerate(keys):
                if key in jumps_dict:
                    jumps_array[idx]  = jumps_dict[key]
                    xticks_array[idx] = x_ticks_all[ii]
                    idx+=1

            fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
            fig.subplots_adjust(left = 0.18, right = 0.82, bottom = 0.18, top = 0.95) # adjust the box of axes regarding the figure size

            axes.set_xlabel(r"Elastic Modulus", labelpad = 8)
            axes.set_ylabel(r"$\Delta c$ ( $10^{-3}$ GPa )", labelpad = 8)
            axes.set_xticks(np.linspace(1,len(jumps_dict),len(jumps_dict)))
            axes.set_xticklabels(['$c_{11}$', '$c_{22}$', '$c_{33}$', '$c_{12}$', '$c_{13}$', '$c_{23}$', '$c_{44}$', '$c_{55}$', '$c_{66}$'])
            if ylimits is not None:
                axes.set_ylim(ylimits)

            bars = axes.bar(np.linspace(1,len(jumps_dict),len(jumps_dict)), jumps_array*1e3, width=0.7, bottom=0, align='center', label='sample A', color=color, alpha=0.8)

            line = axes.axhline(y=0)
            plt.setp(line, color = 'black', linestyle = '-', lw=0.6)

            axes.text(0.5, 0.9, 'positive = downward jump\nfrom normal to SC', transform=axes.transAxes, verticalalignment='top', fontsize=15)

            if pdfpages is not None:
                pdfpages.savefig(bbox_inches='tight')
        
        else:
            print()
            print()
            print('your dictionary containing the jumps of the elastic constants is empty!')
            print()
            print()



    def analyze_data (self, method='LinearAlgebra'):
        Tint, fint, _                = self.interpolate()
        self.dcoc_dict, self.CofT_dict, self.T = self.get_dcoc(Tint, fint, self.fit_path, method)

        self.bulk_modulus = self.get_bulk_modulus(self.CofT_dict)

        if self.el_const_save_path is not None:
            self.save_data(self.dcoc_dict, self.T, self.el_const_save_path, type='dcoc')

        if len(self.fit_error_list) > 0:
            error_array = np.zeros((len(self.fit_error_list), len(self.T), len(self.dcoc_dict)))
            for ii, error_fit in enumerate(self.fit_error_list):
                dcoc, _, _ = self.get_dcoc(Tint, fint, self.fit_path, method, fit_path_uncertainty=error_fit)

                # dcoc_err   = np.zeros((len(self.T), len(self.dcoc_dict)))
                for kk, key in enumerate(self.dcoc_dict):
                    error_array[ii,:,kk] = dcoc[key] - self.dcoc_dict[key]

                # error_array[ii] = dcoc_err - dcoc_true

            error_array = error_array**2
            error_array = np.sum(error_array, axis=0)
            error_array = np.sqrt(error_array)
            error_dict = {}
            for ii, key in enumerate(self.dcoc_dict):
                error_dict[key] = error_array[:,ii]
            self.dcoc_error_dict = deepcopy(error_dict)

            if self.el_const_save_path is not None:
                temp = self.el_const_save_path.split('.')
                temp[-2] = temp[-2]+'_errors'
                error_save_path = '.'.join(temp)
                self.save_data(error_dict, self.T, error_save_path, type='dcoc')

    
    def analyze_jumps (self):
        if len(self.jumps_dict) > 0:
            jumps_dict = {key:item*1e3 for key,item in self.jumps_dict.items()} # converting the jumps in elastic constants to MPa (from GPa)
            relative_jumps_dict = {key:item*1e5 for key,item in self.relative_jumps_dict.items()} # converting the relative jumps in elastic constants to 1e-5
            jumps_text = '\n'

            if self.crystal_structure=='orthorhombic':
                jumps_matrix = np.array([[jumps_dict['c11'], jumps_dict['c12'], jumps_dict['c13']],
                                         [jumps_dict['c12'], jumps_dict['c22'], jumps_dict['c23']],
                                         [jumps_dict['c13'], jumps_dict['c23'], jumps_dict['c33']]])
            else:
                print()
                print()
                print('currently "analyze_jumps" is only implemented for orthorhombic crystal structures')
                print()
                print()
                sys.exit()

            jumps_text+= '------------------------------------------------------------\n\n'
            jumps_text+= 'the absolute jumps in elastic constants are (in MPa):\n'
            jumps_text+= str(jumps_dict) + '\n\n'
            jumps_text+= 'the relative jumps in elastic constants are (in 1e-5):\n'
            jumps_text+= str(relative_jumps_dict) + '\n\n'
            jumps_text+= '------------------------------------------------------------\n\n'
            jumps_text+= 'the square root of the product gives (in MPa):\n'
            if self.crystal_structure=='orthorhombic':
                jumps_text+= 'sqrt(c11*c22): ' + str(np.round(np.sqrt(jumps_dict['c11']*jumps_dict['c22']), 3)) + ' --- c12: ' + str(np.round(np.sqrt(jumps_dict['c12']**2), 3)) + '\n'
                jumps_text+= 'sqrt(c11*c33): ' + str(np.round(np.sqrt(jumps_dict['c11']*jumps_dict['c33']), 3)) + ' --- c13: ' + str(np.round(np.sqrt(jumps_dict['c13']**2), 3)) + '\n'
                jumps_text+= 'sqrt(c22*c33): ' + str(np.round(np.sqrt(jumps_dict['c22']*jumps_dict['c33']), 3)) + ' --- c23: ' + str(np.round(np.sqrt(jumps_dict['c23']**2), 3)) + '\n'

            jumps_text+= '\n------------------------------------------------------------\n\n'
            print()

            eigvals     = np.linalg.eigh(jumps_matrix)[0]
            eigvals_abs = np.abs(eigvals)
            eigvals_abs = np.sort(eigvals_abs)
            eigvals_ratio = eigvals_abs[-1]/eigvals_abs[-2]
            jumps_text+= 'The eigenvalues for the "jumps" matrix are (in MPa):\n'
            jumps_text+= str(eigvals) + '\n'
            jumps_text+= 'The ratio of the largest eigenvalue to the second largest (in absolute values) is:\n'
            jumps_text+= str(eigvals_ratio)
            print(jumps_text)
            self.meta_text = jumps_text + self.meta_text
            self.save_meta_text()
        
        else:
            print()
            print()
            print('your dictionary containing the jumps of the elastic constants is empty!')
            print()
            print()

        return eigvals_ratio

    
    def Ehrenfest_strain (self, delta_c, deltaC_over_T, molar_mass, density):
        """
        relates the jump in specific heat (delta C) at Tc to the jump in elastic modulus (delta c);
        proportionalitiy constant is (dTc/dstrain)^2;
        [delta_c]       = Pa;
        [deltaC_over_T] = J/mol K^2;
        [molar_mass]    = kg/mol
        [density]       = kg/m^3;
        result will be in [dTc/dstrain] = K / percent_strain
        """
        deltaC_over_T = deltaC_over_T / molar_mass * density
        # dTc_dstrain in K/strain
        dTc_dstrain = np.sqrt( np.abs(delta_c / deltaC_over_T) )
        # convert to K/percent_strain
        dTc_dstrain = dTc_dstrain/100

        return dTc_dstrain

    
    def Ehrenfest_bulk (self, bulk, delta_B, deltaC_over_T, molar_mass, density):
        """
        relates the jump in specific heat (delta C) at Tc to the jump in elastic modulus (delta c);
        proportionalitiy constant is (dTc/dstrain)^2;
        [bulk]          = Pa
        [delta_B]       = Pa;
        [deltaC_over_T] = J/mol K^2;
        [molar_mass]    = kg/mol
        [density]       = kg/m^3;
        result will be in [dTc/dP_hydro] = K / GPa
        """
        deltaC_over_T = deltaC_over_T / molar_mass * density
        # dTc_dP_hydro in K/GPa
        dTc_dV = np.sqrt( np.abs(delta_B / deltaC_over_T) ) 
        dTc_dP_hydro = dTc_dV / (bulk/1e9)


        return dTc_dP_hydro


    def Ehrenfest_relations (self, deltaC_over_T, molar_mass, density):
        """
        calculates the Ehrenfest relations for different strains
        [deltaC_over_T] = J/mol K^2;
        [molar_mass] = kg/mol;
        [density]    = kg/m^3;
        """
        if len(self.jumps_dict) > 0:
            jumps_dict = {key:item*1e9 for key,item in self.jumps_dict.items()} # converting the jumps in elastic constants to Pa (from GPa)
            dTc_dstrain_dict = {}
            conversion_dict = {'c11':'dTc/depsilon_xx', 'c22':'dTc/depsilon_yy', 'c33':'dTc/depsilon_zz',
                               'c44':'dTc/depsilon_yz', 'c55':'dTc/depsilon_xz', 'c66':'dTc/depsilon_xy',
                               'c12':'sqrt(dTc/depsilon_xx x dTc/depsilon_yy)',
                               'c13':'sqrt(dTc/depsilon_xx x dTc/depsilon_zz)',
                               'c23':'sqrt(dTc/depsilon_yy x dTc/depsilon_zz)'}
            
            Ehrenfest_text = '\n------------------------------------------------------------\n\n'
            Ehrenfest_text+= 'EHRENFEST RELATIONS\n\n'
            Ehrenfest_text+= 'we are using the following parameters:\n'
            Ehrenfest_text+= '   - jumps in elastic moduli as below\n'
            Ehrenfest_text+=f'   - density: {density} kg/m^3\n'
            Ehrenfest_text+=f'   - molar mass: {molar_mass} kg/mol\n'
            Ehrenfest_text+=f'   - jump in DeltaC/T: {deltaC_over_T} J / (mol K^2)\n\n'
            
            Ehrenfest_text+= 'the derivatives of the critical temperature with respect to strain are (in K / %_strain)\n'
            for key, item in jumps_dict.items():
                val = self.Ehrenfest_strain(delta_c=item, deltaC_over_T=deltaC_over_T, molar_mass=molar_mass, density=density)
                dTc_dstrain_dict[conversion_dict[key]] = val
                Ehrenfest_text+= f'{conversion_dict[key]} = {np.round(val,3)}\n'    

            dTc_dstress_dict = self.Ehrenfest_convert_strain_to_stress (dTc_dstrain_dict)
            Ehrenfest_text+= '\nthe derivatives of the critical temperature with respect to stress are (in K / GPa)\n'
            for name, value in dTc_dstress_dict.items():
                Ehrenfest_text+= f'{name} = {np.round(value, 3)}\n'

            if (self.bulk_modulus is not None) and (self.bulk_jump is not None):
                dTc_dP_hydro = self.Ehrenfest_bulk(self.bulk_modulus[-1]*1e9, self.bulk_jump*1e9, deltaC_over_T, molar_mass, density)
                Ehrenfest_text+= f'\nThe Ehrenfest relation for the bulk modulus yields:\n'
                Ehrenfest_text+= f'dTc/dP_hydro = {dTc_dP_hydro} K/GPa\n'
                Ehrenfest_text+= f'this value comes from a jump in Delta_B = {self.bulk_jump*1e6} MPa and a bulk modulus of {round(self.bulk_modulus[-1],2)} GPa\n'


            
            print(Ehrenfest_text)
            self.meta_text = Ehrenfest_text + self.meta_text
            self.save_meta_text()
            return dTc_dstrain_dict
        
        else:
            print()
            print()
            print('your dictionary containing the jumps of the elastic constants is empty!')
            print()
            print()
            return None


    def Ehrenfest_convert_strain_to_stress (self, dTc_dstrain_dict):
        dTc_dstress_dict = {}
        if self.crystal_structure == 'orthorhombic':
            # first let's do it for compressional strains:
            # convert dTc_dstrain from K/%_strain to K/strain
            dTc_dstrain = np.array([dTc_dstrain_dict['dTc/depsilon_xx'], dTc_dstrain_dict['dTc/depsilon_yy'], dTc_dstrain_dict['dTc/depsilon_zz'],
                                    dTc_dstrain_dict['dTc/depsilon_yz'], dTc_dstrain_dict['dTc/depsilon_xz'], dTc_dstrain_dict['dTc/depsilon_xy']])*100


            # elastic moduli are in GPa so we will get dTc/dstress in K/GPa
            c_mat       = np.array([[self.high_T_el_const['c11'], self.high_T_el_const['c12'], self.high_T_el_const['c13'], 0, 0, 0],
                                    [self.high_T_el_const['c12'], self.high_T_el_const['c22'], self.high_T_el_const['c23'], 0, 0, 0],
                                    [self.high_T_el_const['c13'], self.high_T_el_const['c23'], self.high_T_el_const['c33'], 0, 0, 0],
                                    [0,0,0, self.high_T_el_const['c44'], 0,                           0                          ],
                                    [0,0,0, 0,                           self.high_T_el_const['c55'], 0                          ],
                                    [0,0,0, 0,                           0,                           self.high_T_el_const['c66']]])
            S_mat = np.linalg.inv(c_mat)
            dTc_dstress = np.matmul(S_mat, dTc_dstrain) # in K/GPa
            dTc_dstress_dict['dTc/dsigma_xx'] = dTc_dstress[0]
            dTc_dstress_dict['dTc/dsigma_yy'] = dTc_dstress[1]
            dTc_dstress_dict['dTc/dsigma_zz'] = dTc_dstress[2]
            dTc_dstress_dict['dTc/dsigma_yz'] = dTc_dstress[3]
            dTc_dstress_dict['dTc/dsigma_xz'] = dTc_dstress[4]
            dTc_dstress_dict['dTc/dsigma_xy'] = dTc_dstress[5]

        else:
            print()
            print()
            print('currently, "Ehrenfest_convert_strain_to_stress" is only implemented for orhtorhombic crystals')
            print()
            print()

        return dTc_dstress_dict




