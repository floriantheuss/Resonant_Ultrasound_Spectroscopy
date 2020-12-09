import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt
from scipy.optimize import curve_fit
from scipy import odr
from lmfit import minimize, Parameters, Minimizer, report_fit


# import irrdeducible elastic constants ----------------------------------------------------------------------
folder = "C:\\Users\\j111\\Box Sync\\Projects"
project = "\\Mn3Ge\\RUS\\sample_with_green_face_from_010920\\irreducible_elastic_constants.txt"


data = []
f = open(folder+project, 'r')
    
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
Bulk = ( A1g1 * A1g2 - A1g3**2 ) / ( A1g1 + A1g2 - 2*A1g3 )

# define background functions -------------------------------------------------------------------------------------
def strline (p, T):
    return p[0]*T + p[1]

def sqrtbckg (p, T):
    return p[0]*T + p[1] + p[2]*np.sqrt((p[3]-T)/p[3])

# define the critical divergence with sqrt background below and linear above Tc ---------------------------------
def critsqrt (T, Tc, alpha, Am, Ap, Bm, Bp, Cm, Cp, Dm):
    tm = (Tc-T[T<=Tc])/Tc
    C1 = Am * tm**(-alpha) + Bm*T[T<=Tc] + Cm + Dm*np.sqrt(tm)
    tp = (T[T>Tc]-Tc)/Tc
    C2 = Ap * tp**(-alpha) + Bp*T[T>Tc] + Cp
    C = np.append(C1, C2)
    return C


def residual(params, T, data=None):
    Tc = params['Tc']
    alpha = params['alpha']
    Am = params['Am']
    Ap = params['Ap']
    Bm = params['Bm']
    Bp = params['Bp']
    Cm = params['Cm']
    Cp = params['Cp']
    Dm = params['Dm']
    model = critsqrt (T, Tc, alpha, Am, Ap, Bm, Bp, Cm, Cp, Dm)
    if data is None:
        return model
    return (data-model)



# fit to elastic constant ------------------------------------------------------------------------------------------------------------------------------------------
data = (-A1g1 - min(-A1g1))
x = T

# fit background
Tmax = 356
bdata = odr.RealData(x[x<Tmax], data[x<Tmax])
initial_guess = [0,0,0,370]
model = odr.Model(sqrtbckg)
fit = odr.ODR(bdata, model, beta0=initial_guess)
out = fit.run()
poptsqrt = out.beta
print(poptsqrt)

Tmin = 390
bdata = odr.RealData(x[x>Tmin], data[x>Tmin])
initial_guess = [0,0]
model = odr.Model(strline)
fit = odr.ODR(bdata, model, beta0=initial_guess)
out = fit.run()
poptline = out.beta


# fit critical behavior --------------------------------------------------------------    
# define data to fit
# set fit range 
T1 = 390
T2 = 368.5
T3 = 369.8
T4 = 395
fitmask = ((T>T1) & (T<T2)) | ((T>T3) & (T<T4))
fitmaskinverse =  np.invert(fitmask)


# define fit parameters -------------------------------------------------------------------------
# i.e. initial guess, fit range, fixed values, ...
params = Parameters()
params.add('Tc', value=369.5, vary=False)#, max=369.7)
params.add('alpha', value=-0.014, vary=True)
params.add('Am', value=0, vary=False)#, max=0)
params.add('Ap', value=-10, vary=True)#, max=0)#, expr='1.06*Am')
params.add('Bm', value=0, vary=False)#poptsqrt[0], vary=True)#, max=2*poptsqrt[0])
params.add('Bp', value=poptline[0], vary=True)#, max=2*poptline[0])
params.add('Cm', value=0, vary=False)#poptsqrt[1], vary=True)#, max=2*poptsqrt[1])
params.add('Cp', value=poptline[1], vary=True)#, max=2*poptline[1])#, expr='Cm + (Bm-Bp)*Tc')
params.add('Dm', value=0, vary=False)#poptsqrt[2], vary=True)#, max=2*poptsqrt[2])


# do the fit
out = minimize(residual, params, args=(x[fitmask], data[fitmask]), method='leastsq')

# write error report
report_fit(out)
out.params.pretty_print()


# calculate final result
#final = data[fitmask] - out.residual

result = {}
for name, param in out.params.items():
    result[name] = param.value

for i in result:
    print (i, result[i])

err = {}
for name, param in out.params.items():
    err[name] = param.stderr
    

# try to plot results
plt.scatter(x, data, s=2, c='black')
plt.scatter(x[fitmaskinverse], data[fitmaskinverse], s=2, c='red')
# plt.plot(x[fitmask], final)
plt.plot(x, residual( result, x ), zorder=1 )
plt.plot(x[x<result['Tc']], sqrtbckg(poptsqrt, x[x<result['Tc']]), '--')
plt.plot(x[x>result['Tc']], strline(poptline, x[x>result['Tc']]), '--')

# also plot the fitted background
resultbckg = result
resultbckg['Am'] = 0
resultbckg['Ap'] = 0
plt.plot(x, residual( resultbckg, x ) )


plt.show()




#############################################################################################################################################
# fit for several temperature ranges
# N = 20
# M = 10

# T1 = np.linspace(353, 365, N)
# T2 = np.linspace(366, 369, M)
# T3 = np.linspace(369, 371, M)
# T4 = np.linspace(375, 394, N)

# expdata = -A1g1 - min(-A1g1)

# Tfit = np.array( [[[[[i,j, k, l] for l in T4] for k in T3] for j in T2] for i in T1] )
# Tfit = np.reshape(Tfit, (M**2*N**2,4))

# Touter = np.array([[[i,j] for j in T4] for i in T1])
# Touter = np.reshape(Touter, (N*M,2))



# fit_result = []
# for i in Tfit:
#     # define fit parameters -------------------------------------------------------------------------
#     # i.e. initial guess, fit range, fixed values, ...
#     params = Parameters()
#     params.add('Tc', value=369, vary=True, max=390)
#     params.add('alpha', value=-0.014, vary=True)
#     params.add('Am', value=-10, vary=True)
#     params.add('Ap', value=-10, vary=True)
#     params.add('Bm', value=poptsqrt[0], vary=True)
#     params.add('Bp', value=poptline[0], vary=True)
#     params.add('Cm', value=poptsqrt[1], vary=True)
#     params.add('Cp', value=poptline[1], vary=True)#, expr='Cm + (Bm-Bp)*Tc')
#     params.add('Dm', value=poptsqrt[2], vary=True)

#     # do the fit
#     out = minimize(residual, params, args=(x[fitmask], data[fitmask]), method='leastsq')

#     out.params.pretty_print()

#     result = []
#     for name, param in out.params.items():
#         result.append(param.value)
#     fit_result.append(result)


# folder = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\"
# project = "A1g1_fits//sqrt_background.txt"
# filename = folder+project

# if os.path.isfile(filename) == True:
#     x='w'
# else:
#     x='x'

# with open(filename, x) as g:
#     g.write('T1' + '\t' + 'T2' + '\t' + 'T3' + '\t' + 'T4' + '\t' + 'Tc' + '\t' + 'alpha' + '\t' + 'Aminus' + '\t' + 'Aplus' + '\t' +  'Bminus' + '\t' + 'Bplus' + '\t' + 'Cminus' + '\t' + 'Cplus' + '\t' + 'Dminus' + '\n')
#     for i in np.arange(len(Tfit)):
#         a = ''
#         for j in Tfit[i]:
#             a = a + str(j) + '\t'
#         for j in fit_result[i]:
#             a = a + str(j) + '\t'
#         a = a[:-1] + '\n'
#         g.write(a)


#################################################################################################################################################