import numpy as np
import time
from lmfit import minimize, Parameters, report_fit, Parameter
from scipy import odr
import matplotlib.pyplot as plt
import json

class CriticalExponentFit:

    def __init__ (self, filepath, initial_conditions, fit_ranges, type_of_elastic_constant, include_error_bars):
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

        ## Initialize fit parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.params = Parameters()
        for parameter in self.initial_conditions:
            conditions = self.initial_conditions[parameter]
            self.params.add(parameter, value=conditions['initial_value'], vary=conditions['vary'], min=conditions['bounds'][0], max=conditions['bounds'][-1])




    def import_data (self):
        ## import data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        E1g = data[4]
        E2g = data[5]
        dA1g1 = data[6]
        dA1g2 = data[7]
        dA1g3 = data[8]
        dE1g = data[9]
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

    def line (self, p, T): # define straight line for background estimation
        A, B = p
        return A+B*T

    def fit_background (self):
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

    def residual_function (self, params):
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
            delta = prediction - self.elastic_constant[self.exponent_mask]

        if self.type == 'Bulk':
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
            delta = prediction - self.elastic_constant[self.exponent_mask]
    
        if self.include_errors == True:
            errorbars = self.error
            for i in np.arange(len(errorbars)):
                # errorbars[i] = 1e-6
                if errorbars[i] < 1e-6:
                    errorbars[i] = 1e-6
            # errorbars = errorbars / min(errorbars)
            return delta / errorbars[self.exponent_mask]
        else:
            return delta


    def run_fit (self):

        out = minimize(self.residual_function, self.params, method='least_squares')
        ## Display fit report
        report_fit(out)

        
        for name, param in out.params.items():
            self.results[name] = {'value':param.value, 'stderr':param.stderr}

        return self.results

    def save_results (self, name):
        report = {
            'fitted elastic constant': self.type,
            'include errors for elastic constant': self.include_errors,
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



# test >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if __name__ == '__main__':

    ## Initial parameters 
    initial_conditions = {
        'Tc': {'initial_value':369.9, 'bounds':[368, 371], 'vary':False},
        'alpha': {'initial_value':0.35, 'bounds':[0, 0.8], 'vary':True},
        'delta': {'initial_value':0.1, 'bounds':[0, 100], 'vary':True},
        'A': {'initial_value':-2, 'bounds':[-10, 0], 'vary':True},
        'B': {'initial_value':-1, 'bounds':[-50, 0], 'vary':True}
        }
    
    fit_ranges = {
        'background':{'Tmin':390, 'Tmax':500},
        'critical_exponent':{'Tmin':370.5, 'Tmax':500}
    }

    elastic_constant_to_fit = 'E2g'
    include_errors = True

    folder = "C:\\Users\\Florian\\Box Sync\\Projects"
    project = "\\Mn3Ge\\RUS\\Mn3Ge_2001B\\irreducible_elastic_constants_with_error.txt"
    filepath = folder+project

    test = CriticalExponentFit(filepath, initial_conditions, fit_ranges, elastic_constant_to_fit, include_errors)
    test.import_data()
    test.fit_background()
    results = test.run_fit()
    test.save_results('test.json')
    

    plt.figure()
    plt.plot(test.T, test.elastic_constant)
    plt.plot(test.T, test.line([test.initial_conditions['C']['initial_value'], test.initial_conditions['D']['initial_value']], test.T))
    
    test.plot_results()