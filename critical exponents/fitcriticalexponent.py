import numpy as np
import time
from lmfit import minimize, Parameters, report_fit, Parameter
from scipy import odr
import matplotlib.pyplot as plt
import json

class CriticalExponentFit:

    def __init__ (self, filepath, initial_conditions, fit_ranges, type_of_elastic_constant, include_error_bars, fit_algorithm, save_path):
        """
        fileapath: location of text file containing irreducible elastic constants with errors
        initial_conditions: dictionary giving initial conditions, boundaries, and True/False if parameter is supposed to be fitted/kept fixed
        fit_ranges: dictionary of two fit ranges; one for the background fit and another one for fit of critical exponent
        type_of_elastic_constant: which elastic constant do you want to fit? Options are 'Bulk' for bulk modulus or 'E2g'
        include_error_bars: do you want to include error bars of the elastic constants in your fit (True/False)
        fit_algorithm: in this code you can fit the data with lmfit or scipy.odr so the options are 'lmfit'/'odr'
        save_path: location where fit results are saved
        """
        self.filepath = filepath
        self.initial_conditions = initial_conditions
        self.fit_ranges = fit_ranges
        self.type = type_of_elastic_constant

        self.T = 0
        self.elastic_constant = 0
        self.error = 0
        self.bkg_mask = 0
        self.exponent_mask = 0
        self.results = {}

        self.include_errors = include_error_bars

        self.fit_algorithm = fit_algorithm

        self.save_path = save_path

        ## Initialize fit parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.params = Parameters()
        for parameter in self.initial_conditions:
            conditions = self.initial_conditions[parameter]
            self.params.add(parameter, value=conditions['initial_value'], vary=conditions['vary'], min=conditions['bounds'][0], max=conditions['bounds'][-1])




    def import_data (self):
        """
        import the data and select the right combination of elastic constants for future fit
        """
        data = []
        f = open(self.filepath, 'r')

        f.readline()
        f.readline()

        for line in f:
            line = line.strip()
            line = line.split()
            for i in np.arange(len(line)):
                line[i] = float(line[i])
            data.append(line)
        
        data = np.array(data).transpose()

        T = data[0]
        A1g1 = data[1]
        A1g2 = data[2]
        A1g3 = data[3]
        # E1g = data[4]
        E2g = data[5]
        dA1g1 = data[6]
        dA1g2 = data[7]
        dA1g3 = data[8]
        # dE1g = data[9]
        dE2g = data[10]
        Bulk = ( A1g1 * A1g2 - A1g3**2 ) / ( A1g1 + A1g2 - 2*A1g3 )
        dBulk = np.sqrt( ((A1g2-A1g3)**2*dA1g1)**2 + ((A1g1-A1g3)**2*dA1g2)**2 + (2*(A1g1-A1g3)*(A1g2-A1g3)*dA1g3)**2 ) / (A1g1+A1g2-2*A1g3)**2
        
        self.T = T
        if self.type == 'E2g':
            self.elastic_constant = E2g
            self.error = dE2g
        elif self.type == 'Bulk':
            self.elastic_constant = Bulk
            self.error = dBulk
        else:
            print ('give a different type of elastic constant to fit')
        return 0

    def line (self, p, T):
        """
        define a straight line for background estimation
        """
        A, B = p
        return A+B*T

    def fit_background (self):
        """
        fit a straight line to selected parts of the data (the part is selected in the dictionary fit_ranges);
        the results of this fit are put into the initial_conditions dictionary used for the fit of critical exponents
        """
        self.bkg_mask = ( (self.T>self.fit_ranges['background']['Tmin']) & (self.T<self.fit_ranges['background']['Tmax']) )
        # actually run the fitting routine for the linear background above Tmin
        bpdata = odr.RealData(self.T[self.bkg_mask], self.elastic_constant[self.bkg_mask])
        initial_guess = [0,0]
        fix = [1,1]
        model = odr.Model(self.line)
        fit = odr.ODR(bpdata, model, beta0=initial_guess, ifixb=fix)
        out = fit.run()
        poptbg = out.beta
        
        self.initial_conditions['C'] = {'initial_value':poptbg[0], 'bounds':[poptbg[0]-100, poptbg[0]+100], 'vary':True}
        self.initial_conditions['D'] = {'initial_value':poptbg[1], 'bounds':[poptbg[1]/100, poptbg[1]*100], 'vary':True}
        self.params['C'] = Parameter(name='C', value=poptbg[0], vary='True', min=poptbg[0]-100, max=poptbg[0]+100)
        self.params['D'] = Parameter(name='D', value=poptbg[1], vary='True', min=poptbg[1]/100, max=poptbg[1]*100)
        return 0


    def simulated_elastic_constant (self, params):
        """
        calculate the divergence of the elastic constant, i.e. t^(-alpha) and other parts for lmfit
        """
        if self.type == 'E2g':
            Tmin = self.fit_ranges['critical_exponent']['Tmin']
            Tmax = self.fit_ranges['critical_exponent']['Tmax']
            self.exponent_mask = ( (self.T>Tmin) & (self.T<Tmax) )
            Tc = params["Tc"].value
            alpha = params["alpha"].value
            delta = params["delta"].value
            A = params["A"].value
            B = params["B"].value
            C = params["C"].value
            D = params["D"].value

            ## Compute predicted shear modulus
            Tmasked = self.T[self.exponent_mask]
            t = (Tmasked-Tc)/Tc
            prediction = A * t**(-alpha) * ( 1 + B * t**delta) + C + D*Tmasked
            
        elif self.type == 'Bulk':
            T1 = self.fit_ranges['critical_exponent']['T1']
            T2 = self.fit_ranges['critical_exponent']['T2']
            T3 = self.fit_ranges['critical_exponent']['T3']
            T4 = self.fit_ranges['critical_exponent']['T4']
            self.exponent_mask = ((self.T>T1) & (self.T<T2)) | ((self.T>T3) & (self.T<T4))
            Tc = params["Tc"].value
            alpha = params["alpha"].value
            delta = params["delta"].value
            Am = params["Am"].value
            Bm = params["Bm"].value
            Ap = params["Ap"].value
            Bp = params["Bp"].value
            C = params["C"].value
            D = params["D"].value

            Tmasked = self.T[self.exponent_mask]
            tm = (Tc-Tmasked[Tmasked<=Tc])/Tc
            tp = (Tmasked[Tmasked>Tc]-Tc)/Tc
            C1 = Am * tm**(-alpha) * ( 1 + Bm * tm**delta)
            C2 = Ap * tp**(-alpha) * ( 1 + Bp * tp**delta)
            prediction = np.append(C1, C2) + C + D*Tmasked
        
        else:
            print ('You have not defined a viable elastic constant to fit to')
        return (prediction)
        
            


    def residual_function (self, params):
        """
        define the residual function used in lmfit;
        i.e. (simulated_elastic_constant - data)
        """
        prediction = self.simulated_elastic_constant (params)
        delta = prediction - self.elastic_constant[self.exponent_mask]

        if self.include_errors == True:
            errorbars = self.error
            if np.any(errorbars[self.exponent_mask]<1e-6) == True:
                print('There are errors smaller than 1e-6. They are set to 1e-6 in the fit.')
            for i in np.arange(len(errorbars)):
                if errorbars[i] < 1e-6:
                    errorbars[i] = 1e-6
            return delta / errorbars[self.exponent_mask]
        else:
            return delta

        

    def run_lmfit_fit (self):

        out = minimize(self.residual_function, self.params, method='least_squares')
        ## Display fit report
        report_fit(out)

        
        for name, param in out.params.items():
            self.results[name] = {'value':param.value, 'stderr':param.stderr}

        return self.results


    
    def odr_fit_function (self, p, T):
        """
        define fit function for scipy.odr fit
        """
        if self.type == 'E2g':
            Tc, alpha, delta, A, B, C, D = p
            t = (T-Tc)/Tc
            prediction = A * t**(-alpha) * ( 1 + B * t**delta) + C + D*T

        elif self.type == 'Bulk':
            Tc, alpha, delta, Am, Bm, Ap, Bp, C, D = p
            tm = (Tc-T[T<=Tc])/Tc
            tp = (T[T>Tc]-Tc)/Tc
            C1 = Am * tm**(-alpha) * ( 1 + Bm * tm**delta)
            C2 = Ap * tp**(-alpha) * ( 1 + Bp * tp**delta)
            prediction = np.append(C1, C2) + C + D*T
        
        else:
            print ('You have not defined a viable elastic constant to fit to')
        return (prediction)


    
    def run_odr_fit (self):
        """
        run a fit for a critical exponent with scipy.odr instead of lmfit
        (one of the advantages here is that y-errors and x-erros can be included)
        """

        if self.type == 'E2g':
            Tmin = self.fit_ranges['critical_exponent']['Tmin']
            Tmax = self.fit_ranges['critical_exponent']['Tmax']
            self.exponent_mask = ( (self.T>Tmin) & (self.T<Tmax) )  
            initial_guess = [self.initial_conditions['Tc']['initial_value'], self.initial_conditions['alpha']['initial_value'], self.initial_conditions['delta']['initial_value'], self.initial_conditions['A']['initial_value'], self.initial_conditions['B']['initial_value'], self.initial_conditions['C']['initial_value'], self.initial_conditions['D']['initial_value']]
            fix = [self.initial_conditions['Tc']['vary'], self.initial_conditions['alpha']['vary'], self.initial_conditions['delta']['vary'], self.initial_conditions['A']['vary'], self.initial_conditions['B']['vary'], self.initial_conditions['C']['vary'], self.initial_conditions['D']['vary']]
        elif self.type == 'Bulk':
            T1 = self.fit_ranges['critical_exponent']['T1']
            T2 = self.fit_ranges['critical_exponent']['T2']
            T3 = self.fit_ranges['critical_exponent']['T3']
            T4 = self.fit_ranges['critical_exponent']['T4']
            self.exponent_mask = ((self.T>T1) & (self.T<T2)) | ((self.T>T3) & (self.T<T4))
            initial_guess = [self.initial_conditions['Tc']['initial_value'], self.initial_conditions['alpha']['initial_value'], self.initial_conditions['delta']['initial_value'], self.initial_conditions['Am']['initial_value'], self.initial_conditions['Bm']['initial_value'], self.initial_conditions['Ap']['initial_value'], self.initial_conditions['Bp']['initial_value'], self.initial_conditions['C']['initial_value'], self.initial_conditions['D']['initial_value']]
            fix = [self.initial_conditions['Tc']['vary'], self.initial_conditions['alpha']['vary'], self.initial_conditions['delta']['vary'], self.initial_conditions['Am']['vary'], self.initial_conditions['Bm']['vary'], self.initial_conditions['Ap']['vary'], self.initial_conditions['Bp']['vary'], self.initial_conditions['C']['vary'], self.initial_conditions['D']['vary']]
        else:
            print ('You have not define a viable elastic constant to fit to')
        

        if self.include_errors == True:
            errorbars = self.error
            if np.any(errorbars[self.exponent_mask]<1e-6) == True:
                print('There are errors smaller than 1e-6. They are set to 1e-6 in the fit.')
            for i in np.arange(len(errorbars)):
                if errorbars[i] < 1e-6:
                    errorbars[i] = 1e-6
            data = odr.RealData(self.T[self.exponent_mask], self.elastic_constant[self.exponent_mask], sy=errorbars[self.exponent_mask])
        else:
            data = odr.RealData(self.T[self.exponent_mask], self.elastic_constant[self.exponent_mask])
        fix_new = []
        for i in fix:
            if i == True:
                fix_new.append(1)
            else:
                fix_new.append(0)
        model = odr.Model(self.odr_fit_function)
        fit = odr.ODR(data, model, beta0=initial_guess, ifixb=fix_new)
        out = fit.run()
        popt = out.beta
        perr = out.sd_beta
        
        if self.type == 'E2g':
            self.results = {
                'Tc': {'value':popt[0], 'stderr':perr[0]},
                'alpha': {'value':popt[1], 'stderr':perr[1]},
                'delta': {'value':popt[2], 'stderr':perr[2]},
                'A': {'value':popt[3], 'stderr':perr[3]},
                'B': {'value':popt[4], 'stderr':perr[4]},
                'C': {'value':popt[5], 'stderr':perr[5]},
                'D': {'value':popt[6], 'stderr':perr[6]}
            }
        elif self.type == 'Bulk':
            self.results = {
                'Tc': {'value':popt[0], 'stderr':perr[0]},
                'alpha': {'value':popt[1], 'stderr':perr[1]},
                'delta': {'value':popt[2], 'stderr':perr[2]},
                'Am': {'value':popt[3], 'stderr':perr[3]},
                'Bm': {'value':popt[4], 'stderr':perr[4]},
                'Ap': {'value':popt[5], 'stderr':perr[5]},
                'Bp': {'value':popt[6], 'stderr':perr[6]},
                'C': {'value':popt[7], 'stderr':perr[7]},
                'D': {'value':popt[8], 'stderr':perr[8]}
            }
        
        for key, item in self.results.items():
            if item['value'] !=0:
                print (key, ': ', item, '    ', item['stderr']/item['value']*100, '% relative error')
            else:
                print (key, ': ', item)
        return self.results

    
    def fit (self):
        self.import_data()
        self.fit_background()

        if self.fit_algorithm == 'lmfit':
            results = self.run_lmfit_fit()
        elif self.fit_algorithm == 'odr':
            results = self.run_odr_fit()
        
        self.save_results(self.save_path)

        self.plot_results()

        return 1



    def save_results (self, name):
        report = {
            'fitted elastic constant': self.type,
            'include errors for elastic constant': self.include_errors,
            'fit algorithm used': self.fit_algorithm,
            'bounds for fits': self.fit_ranges,
            'input parameters': self.initial_conditions,
            'fit results': self.results
            }

        with open(name, 'w') as f:
            json.dump(report, f, indent=4)



    def plot_results (self):
        # calculate the fitted elastic constant
        Tc = self.results['Tc']['value']
        f, ax = plt.subplots(dpi=90)#figsize=(10,7))
        
        if self.type == 'E2g':
            Tsim = np.linspace(Tc+0.01, max(self.T), int(1e3))
            
            A = self.results['A']['value']
            B = self.results['B']['value']
            C = self.results['C']['value']
            D = self.results['D']['value']
            alpha = self.results['alpha']['value']
            delta = self.results['delta']['value']
            Tc = self.results['Tc']['value']

            tsim = (Tsim-Tc)/Tc
            sim = A * tsim**(-alpha) * ( 1 + B * tsim**delta) + C + D*Tsim

            # plot the fit result
            ax.plot(Tsim, sim, zorder=2, c='tab:orange', label='$\\mathrm{C_{E_{2g}}}$', linewidth=3)
            # plot the measured data
            ax.fill_between(self.T, self.elastic_constant-self.error, self.elastic_constant+self.error, alpha=0.3, facecolor='lightgrey', zorder=-1)
            ax.plot(self.T[self.exponent_mask], self.elastic_constant[self.exponent_mask], c='black', zorder=-2, linewidth=3)
            ax.plot(self.T[np.invert(self.exponent_mask)], self.elastic_constant[np.invert(self.exponent_mask)], c='lightgrey', linewidth=3, zorder=1)

            plt.ylabel('$\\mathrm{C_{E2g}}$ (GPa)', fontsize=18)

        if self.type == 'Bulk':
            Tsim = np.linspace(min(self.T), max(self.T), int(1e3))
            
            Am = self.results['Am']['value']
            Bm = self.results['Bm']['value']
            Ap = self.results['Ap']['value']
            Bp = self.results['Bp']['value']
            C = self.results['C']['value']
            D = self.results['D']['value']
            alpha = self.results['alpha']['value']
            delta = self.results['delta']['value']
            Tc = self.results['Tc']['value']

            tm = (Tc-Tsim[Tsim<=Tc])/Tc
            tp = (Tsim[Tsim>Tc]-Tc)/Tc
            C1 = Am * tm**(-alpha) * ( 1 + Bm * tm**delta)
            C2 = Ap * tp**(-alpha) * ( 1 + Bp * tp**delta)
            sim = np.append(C1, C2) + C + D*Tsim

            # plot the fit result
            ax.plot(Tsim, sim, zorder=2, c='tab:orange', label='$\\mathrm{C_{E_{2g}}}$', linewidth=3)
            # plot the measured data
            ax.fill_between(self.T, self.elastic_constant-self.error, self.elastic_constant+self.error, alpha=0.3, facecolor='lightgrey', zorder=-1)
            ax.plot(self.T, self.elastic_constant, c='black', zorder=-2, linewidth=3)
            ax.plot(self.T[np.invert(self.exponent_mask)], self.elastic_constant[np.invert(self.exponent_mask)], c='lightgrey', linewidth=3, zorder=1)

            plt.ylabel('Bulk Modulus (GPa)', fontsize=18)

        # ------------------------- other plot settings
        ax.set_xlabel('T (K)', fontsize=18)
        ax.set_xlim(min(self.T)-1, max(self.T)+1)

        ax.tick_params(axis="y", direction="in", labelsize=15, left='True', right='True', length=4, width=1, which = 'major')
        ax.tick_params(axis="x", direction="in", labelsize=15, bottom='True', top='False', length=4, width=1, which = 'major')
        ax.xaxis.tick_bottom()

        #--------------- this creates a second axis at the top with (T-Tc)/T instead of T
        def Ttot (T):
            return (T-Tc)/Tc

        def ttoT (t):
            return (Tc + Tc*t)
        
        secax = ax.secondary_xaxis('top', functions=(Ttot, ttoT))
        secax.set_xlabel('$\\mathrm{(T - T_c)/T_c}$', fontsize=18)

        secax.tick_params(axis="x", direction="in", labelsize=15, bottom='False', top='True', length=4, width=1, which = 'major')
        secax.xaxis.tick_top()

        plt.show()