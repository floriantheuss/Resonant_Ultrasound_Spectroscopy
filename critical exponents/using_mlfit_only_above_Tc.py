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


# define the critical divergence only above Tc ---------------------------------
def critsqrt (p, T):
    Tc, alpha, Ap, Bp, Cp = p
    tp = abs((T-Tc))
    C2 = Ap * tp**(-alpha) + Bp*T + Cp
    return C2


# fit to elastic constant ------------------------------------------------------------------------------------------------------------------------------------------
data = (-A1g1 - min(-A1g1))
x = T

# fit background
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
T1 = 369.9
T2 = 395
fitmask = ((x>T1) & (x<T2))
fitmaskinverse =  np.invert(fitmask)

fdata = odr.RealData(x[fitmask], data[fitmask])
initial_guess = [369.68, -0.014, -35721, poptline[0], poptline[1]]
fix = [0,1,1,1,1]
model = odr.Model(critsqrt)
fit = odr.ODR(fdata, model, beta0=initial_guess, ifixb=fix)
out = fit.run()
poptcr = out.beta

names = ['Tc', 'alpha', 'Ap', 'Bp', 'Cp']
for i in np.arange(len(names)):
    print( names[i] + ' = ' + str(poptcr[i]) )



plotmask = fitmask# & (x>poptcr[0])
# try to plot results
plt.scatter(x, data, s=2, c='black')
plt.scatter(x[fitmaskinverse], data[fitmaskinverse], s=2, c='red')

plt.plot(x[plotmask], critsqrt(poptcr, x[plotmask]), zorder=1)

plt.plot(x[x>poptcr[0]], strline(poptline, x[x>poptcr[0]]), '--')
plt.plot(x[x>poptcr[0]], strline([poptcr[3], poptcr[4]], x[x>poptcr[0]]), '--')


plt.show()