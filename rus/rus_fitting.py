import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution, linear_sum_assignment
import time
from copy import deepcopy
import os
import sys
from IPython.display import clear_output
from psutil import cpu_count
from rus.rus_comsol import RUSComsol
from rus.rus_rpr import RUSRPR
from lmfit import minimize, Parameters, report_fit
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class RUSFitting:
    def __init__(self, rus_object, bounds_dict,
                 freqs_file,
                 nb_freqs,
                 nb_missing=0,
                 report_name="",
                 method='differential_evolution',
                 population=15, N_generation=10000, mutation=0.7, crossing=0.9,
                 polish=True, updating='immediate', tolerance=0.01):
        """
        freqs_files should contain experimental resonance frequencies in MHz and a weight
        """
        # get initial guess from parameters given to rus_object
        self.rus_object  = rus_object
        self.init_pars   = deepcopy(self.rus_object.cij_dict)
        self.init_pars["angle_x"] = self.rus_object.angle_x
        self.init_pars["angle_y"] = self.rus_object.angle_y
        self.init_pars["angle_z"] = self.rus_object.angle_z
        self.best_pars   = deepcopy(self.init_pars)
        self.best_cij_dict   = deepcopy(self.rus_object.cij_dict)
        # bounds_dict are the bounds given to a genetic algorithm
        self.bounds_dict = bounds_dict
        # the parameters given in bounds_dict are "free" parameters, i.e. they are varied in the fit
        self.free_pars_name  = sorted(self.bounds_dict.keys())
        # fixed parameters are given as parameters which are in init_pars, but NOT in bounds_dict
        # they are fixed parameters, not varied in the fit

        self.fixed_pars_name = np.setdiff1d(sorted(self.init_pars.keys()),
                                             self.free_pars_name)
        

        ## Load data
        self.nb_freqs           = nb_freqs
        self.nb_missing         = nb_missing
        self.rus_object.nb_freq = nb_freqs + nb_missing
        self.freqs_file      = freqs_file
        self.col_freqs       = 0
        self.col_weight      = 1
        self.freqs_data      = None
        self.weight          = None
        self.load_data()
        

        ## fit algorithm
        self.method = method # "shgo", "differential_evolution", "leastsq"

        ## set up fit parameters for lmfit
        self.params = Parameters()
        for param in self.free_pars_name:
            self.params.add(param, value=self.init_pars[param], vary=True, min=self.bounds_dict[param][0], max=self.bounds_dict[param][1])
        for param in self.fixed_pars_name:
            self.params.add(param, value=self.init_pars[param], vary=False)
               

        ## Differential evolution
        self.population    = population # (popsize = population * len(x))
        self.N_generation  = N_generation
        self.mutation      = mutation
        self.crossing      = crossing
        self.polish        = polish
        self.updating      = updating
        self.tolerance     = tolerance

        self.report_name = report_name

        ## Empty spaces
        self.rms = None
        self.nb_gens   = 0
        self.best_freqs_found   = []
        self.best_index_found   = []
        self.best_freqs_missing = []
        self.best_index_missing = []

        ## empty spaces for fit properties
        self.fit_output = None
        self.fit_duration = 0

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def load_data(self):
        """
        Frequencies should be in MHz
        """
        ## Load the resonance data in MHz
        data = np.loadtxt(self.freqs_file, dtype="float", comments="#")
        if len(data.shape) > 1:
            freqs_data = data[:,self.col_freqs]
            # weight     = np.ones_like(freqs_data)
            weight     = data[:,self.col_weight]
        else:
            freqs_data = data
            weight     = np.ones_like(freqs_data)
        ## Only select the first number of "freq to compare"
        if self.nb_freqs == 'all':
            self.nb_freqs = len(freqs_data)
        try:
            assert self.nb_freqs <= freqs_data.size
        except AssertionError:
            print("You need --- nb calculated freqs <= nb data freqs")
            sys.exit(1)
        self.freqs_data = freqs_data[:self.nb_freqs]
        self.weight = weight[:self.nb_freqs]


    def assignement(self, freqs_data, freqs_sim):
        """
        Linear assigment of the simulated frequencies to the data frequencies
        in case there is one or more missing frequencies in the data
        """
        cost_matrix = distance_matrix(freqs_data[:, None], freqs_sim[:, None])**2
        index_found = linear_sum_assignment(cost_matrix)[1]
        ## index_found is the indices for freqs_sim to match freqs_data
        return index_found, freqs_sim[index_found]


    def sort_freqs(self, freqs_sim):
        if self.nb_missing != 0:
            ## Linear assignement of the simulated frequencies to the data
            index_found, freqs_found = self.assignement(self.freqs_data, freqs_sim)
            ## Give the missing frequencies in the data -------------------------
            # Let's remove the extra simulated frequencies that are beyond
            # the list of data frequencies
            bool_missing = np.ones(freqs_sim.size, dtype=bool)
            bool_missing[index_found] = False
            index_missing = np.arange(0, freqs_sim.size, 1)[bool_missing]
            # index_missing = index_missing[index_missing < self.freqs_data.size]
            freqs_missing = freqs_sim[index_missing]
        else:
            ## Only select the first number of "freq to compare"
            freqs_found = freqs_sim[:self.nb_freqs]
            index_found = np.arange(len(freqs_found))
            freqs_missing = []
            index_missing = []
        return freqs_found, index_found, freqs_missing, index_missing


    def residual_function (self, pars):
        """
        define the residual function used in lmfit;
        i.e. (simulated resonance frequencies - data)
        """
        ### update elastic constants and angles with current values
        for free_name in self.free_pars_name:
            if free_name not in ["angle_x", "angle_y", "angle_z"]:
                self.best_pars[free_name]     = pars[free_name].value
                self.best_cij_dict[free_name] = pars[free_name].value
            elif free_name=='angle_x':
                self.best_pars[free_name] = pars[free_name].value
                self.rus_object.angle_x = pars[free_name].value
            elif free_name=='angle_y':
                self.best_pars[free_name] = pars[free_name].value
                self.rus_object.angle_y = pars[free_name].value
            elif free_name=='angle_z':
                self.best_pars[free_name] = pars[free_name].value
                self.rus_object.angle_z = pars[free_name].value
        self.rus_object.cij_dict = self.best_cij_dict

        # calculate resonances with new parameters
        freqs_sim = self.rus_object.compute_resonances()
        # find missing resonances
        freqs_found, index_found, freqs_missing, index_missing = self.sort_freqs(freqs_sim)

        # update attributes
        self.nb_gens += 1
        self.best_freqs_found   = freqs_found
        self.best_index_found   = index_found
        self.best_freqs_missing = freqs_missing
        self.best_index_missing = index_missing

        # this is what we want to be minimized
        diff = (self.best_freqs_found - self.freqs_data) * self.weight
        self.rms = np.sqrt(np.sum(((diff[diff!=0]/self.freqs_data[diff!=0]))**2) / len(diff[diff!=0])) * 100

        print ('NUMBER OF GENERATIONS: ', self.nb_gens)
        print ('BEST PARAMETERS:')
        for key, item in self.best_pars.items():
            print ('\t',key, ': ', round(item, 5))
        print ('MISSING FREQUENCIES: ', freqs_missing[index_missing<len(self.freqs_data)])
        print ('RMS: ', round(self.rms, 5), ' %')
        print ('')
        print ('#', 50*'-')
        print ('')       
        
        return diff

    
    def run_fit (self, print_derivatives=False):
        if isinstance(self.rus_object, RUSComsol) and (self.rus_object.client is None):
            print ("the rus_comsol object was not started!")
            print ("it is being initialized right now ...")
            self.rus_object.start_comsol()
        if isinstance(self.rus_object, RUSRPR) and (self.rus_object.Emat is None):
            print ("the rus_rpr object was not initialized!")
            print ("it is being initialized right now ...")
            self.rus_object.initialize()

        # start timer
        t0 = time.time()
        
        # run fit
        if self.method == 'differential_evolution':
            fit_output = minimize(self.residual_function, self.params, method=self.method,
                                        updating=self.updating,
                                        polish=self.polish,
                                        maxiter=self.N_generation,
                                        popsize=self.population,
                                        mutation=self.mutation,
                                        recombination=self.crossing,
                                        tol=self.tolerance)
        elif self.method == 'leastsq':
            fit_output = minimize(self.residual_function, self.params, method=self.method)
        else:
            print ('your fit method is not a valid method')

        self.fit_output = fit_output
        # stop timer
        self.fit_duration = time.time() - t0


        if print_derivatives == False:
            v_spacing = '\n' + 79*'#' + '\n' + 79*'#' + '\n' + '\n'
            report  = v_spacing
            report += self.report_sample_text()
            report += v_spacing
            report += self.report_fit()
            report += v_spacing
            report += self.report_best_pars()
            report += v_spacing
            report += self.report_best_freqs()            
            print(report)
        else:
            report = self.report_total()
            print(report)
        self.save_report(report)
        return self.rus_object

    # the following methods are just to display the fit report and data in a nice way
    
    def report_best_pars(self):
        report = "#Variables" + '-'*(70) + '\n'
        for free_name in self.free_pars_name:
            if free_name[0] == "c": unit = "GPa"
            else: unit = "deg"
            report+= "\t# " + free_name + " : " + r"{0:.3f}".format(self.best_pars[free_name]) + " " + \
                     unit + \
                     " (init = [" + str(self.bounds_dict[free_name]) + \
                     ", " +         unit + "])" + "\n"
        report+= "#Fixed values" + '-'*(67) + '\n'
        if len(self.fixed_pars_name) == 0:
            report += "\t# " + "None" + "\n"
        else:
            for fixed_name in self.fixed_pars_name:
                if fixed_name[0] == "c": unit = "GPa"
                else: unit = "deg"
                report+= "\t# " + fixed_name + " : " + \
                        r"{0:.8f}".format(self.best_pars[fixed_name]) + " " + \
                        unit + "\n"
        # report += "#Missing frequencies" + '-'*(60) + '\n'
        # for freqs_missing in self.best_freqs_missing:
        #     report += "\t# " + r"{0:.4f}".format(freqs_missing) + " MHz\n"
        return report


    def report_fit(self):
        fit_output  = self.fit_output
        duration    = np.round(self.fit_duration, 2)
        N_points    = self.nb_freqs
        N_variables = len(self.bounds_dict)
        chi2 = fit_output.chisqr
        reduced_chi2 = chi2 / (N_points - N_variables)
        report = "#Fit Statistics" + '-'*(65) + '\n'
        report+= "\t# fitting method     \t= " + self.method + "\n"
        report+= "\t# data points        \t= " + str(N_points) + "\n"
        report+= "\t# variables          \t= " + str(N_variables) + "\n"
        report+= "\t# fit success        \t= " + str(fit_output.success) + "\n"
        # report+= "\t# generations        \t= " + str(fit_output.nit) + " + 1" + "\n"
        report+= "\t# function evals     \t= " + str(fit_output.nfev) + "\n"
        report+= "\t# fit duration       \t= " + str(duration) + " seconds" + "\n"
        report+= "\t# chi-square         \t= " + r"{0:.8f}".format(chi2) + "\n"
        report+= "\t# reduced chi-square \t= " + r"{0:.8f}".format(reduced_chi2) + "\n"
        return report


    def report_best_freqs(self, nb_additional_freqs=10):
        if (nb_additional_freqs != 0) or (self.best_freqs_found == []):
            if isinstance(self.rus_object, RUSComsol) and (self.rus_object.client is None):
                self.rus_object.start_comsol()
            if isinstance(self.rus_object, RUSRPR) and (self.rus_object.Emat is None):
                self.rus_object.initialize()
            self.rus_object.nb_freq = self.nb_freqs + len(self.best_index_missing) + nb_additional_freqs
            freqs_sim = self.rus_object.compute_resonances()
            freqs_found, index_found, freqs_missing, index_missing = self.sort_freqs(freqs_sim)
        else:
            freqs_found   = self.best_freqs_found
            index_found   = self.best_index_found
            freqs_missing = self.best_freqs_missing
            index_missing = self.best_index_missing
            # print(len(freqs_found) + len(freqs_missing))
            freqs_sim = np.empty(len(freqs_found) + len(freqs_missing))
            # print(freqs_sim.size)
            freqs_sim[index_found]   = freqs_found
            freqs_sim[index_missing] = freqs_missing

        freqs_data = np.empty(len(freqs_found) + len(freqs_missing))
        freqs_data[index_found] = self.freqs_data
        freqs_data[index_missing] = 0

        weight = np.empty(len(freqs_found) + len(freqs_missing))
        weight[index_found] = self.weight
        weight[index_missing] = 0

        diff = np.zeros_like(freqs_data)
        for i in range(len(freqs_data)):
            if freqs_data[i] != 0:
                diff[i] = np.abs(freqs_data[i]-freqs_sim[i]) / freqs_data[i] * 100 * weight[i]
        rms = np.sqrt(np.sum((diff[diff!=0])**2) / len(diff[diff!=0]))

        template = "{0:<8}{1:<13}{2:<13}{3:<13}{4:<8}"
        report  = template.format(*['#index', 'f exp(MHz)', 'f calc(MHz)', 'diff (%)', 'weight']) + '\n'
        report += '#' + '-'*(79) + '\n'
        for j in range(len(freqs_sim)):
            if j < len(freqs_data):
                report+= template.format(*[j, round(freqs_data[j],6), round(freqs_sim[j],6), round(diff[j], 3), round(weight[j], 0)]) + '\n'
            else:
                report+= template.format(*[j, '', round(freqs_sim[j],6)], '') + '\n'
        report += '#' + '-'*(79) + '\n'
        report += '# RMS = ' + str(round(rms,3)) + ' %\n'
        report += '#' + '-'*(79) + '\n'

        return report

    def report_sample_text(self):
        sample_template = "{0:<40}{1:<20}"
        sample_text = '# [[Sample Characteristics]] \n'
        sample_text += '# ' + sample_template.format(*['crystal symmetry:', self.rus_object.symmetry]) + '\n'
        if isinstance(self.rus_object, RUSRPR):
            sample_text += '# ' + sample_template.format(*['sample dimensions (mm) (x,y,z):', str(self.rus_object.dimensions*1e3)]) + '\n'
            sample_text += '# ' + sample_template.format(*['mass (mg):', self.rus_object.mass*1e6]) + '\n'
            sample_text += '# ' + sample_template.format(*['highest order basis polynomial:', self.rus_object.order]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'RUS_RPR']) + '\n'
        if isinstance(self.rus_object, RUSComsol):
            sample_text += '# ' + sample_template.format(*['Comsol file:', self.rus_object.mph_file]) + '\n'
            sample_text += '# ' + sample_template.format(*['resonance frequencies calculated with:', 'Comsol']) + '\n'
        return sample_text


    def report_total(self, comsol_start=True):
        report_fit = self.report_fit()
        report_fit += self.report_best_pars()
        if isinstance(self.rus_object, RUSRPR):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False)
        if isinstance(self.rus_object, RUSComsol):
            freq_text  = self.report_best_freqs(nb_additional_freqs=10)
            der_text = self.rus_object.print_logarithmic_derivative(print_frequencies=False, comsol_start=False)
            self.rus_object.stop_comsol()


        sample_text = self.report_sample_text()

        data_text = ''
        freq_text_split = freq_text.split('\n')
        freq_text_prepend = [' '*len(freq_text_split[0])] + freq_text_split
        for j in np.arange(len(freq_text.split('\n'))):
            if j == 2 or j==len(freq_text.split('\n'))-3:
                data_text += '#' + '-'*119 +'\n'
            elif j < len(der_text.split('\n')):
                data_text += freq_text_prepend[j] + der_text.split('\n')[j] + '\n'
            else:
                if j==len(freq_text.split('\n'))-1:
                    data_text += '#' + '-'*119 +'\n'
                else:
                    data_text += freq_text_prepend[j] + '\n'

        v_spacing = '\n' + 120*'#' + '\n' + 120*'#' + '\n' + '\n'
        report_total = v_spacing + sample_text + v_spacing +\
                 report_fit + v_spacing + data_text

        return report_total


    def save_report(self, report):
        if self.report_name == "":
            self.report_name = "fit_report.txt"
        report_file = open(self.report_name, "w")
        report_file.write(report)
        report_file.close()


