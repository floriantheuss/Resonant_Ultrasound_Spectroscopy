import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt
from scipy.optimize import curve_fit
from scipy import odr

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import irrdeducible elastic constants ----------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
folder = "C:\\Users\\Florian\\Box Sync\\Projects"
project = "\\Mn3Ge\\RUS\\Mn3Ge_2001B\\irreducible_elastic_constants_with_error.txt"
# project = "\\Mn3.019Sn0.981\\RUS\\2007A\\irreducible_elastic_constants.txt"


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
dA1g1 = data[6]
dA1g2 = data[7]
dA1g3 = data[8]
dE1g = data[9]
dE2g = data[10]
Bulk = ( A1g1 * A1g2 - A1g3**2 ) / ( A1g1 + A1g2 - 2*A1g3 )
dBulk = np.sqrt( ((A1g2-A1g3)**2*dA1g1)**2 + ((A1g1-A1g3)**2*dA1g2)**2 + (2*(A1g1-A1g3)*(A1g2-A1g3)*dA1g3)**2 ) / (A1g1+A1g2-2*A1g3)**2




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# define critical divergence ---------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# this is only for T>Tc
def crit (p, T):
    Tc, alpha, delta, A, B, C, D  = p
    t = (T-Tc)/Tc
    if delta >= 0:# and alpha<0 and Am<0 and Ap<0:
        C = A * t**(-alpha) * ( 1 + B * t**delta) + C + D*T
    else:
        C = 0 * T
    return C

# define second order polynomial
def poly (p, T):
    A, B, C = p
    y = A + B*T + C*T**2
    return y



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# fit to elastic constant ------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------


# set fit range and data --------------------------------------------------------------------
#--------------------------------------------------------------------------------------------

# ------------------------- Mn3Ge
T1 = 369.9
T2 = 400
# ------------------------- Mn3.019Sn0.981
# T1 = 415.8
# T2 = 438

# create a mask based on the temperature range given above
fitmask = ((T>T1) & (T<T2))
fitmaskinverse =  np.invert(fitmask)

# which elasatic constant do you want to fit to
expdata = E2g
dexpdata = dE2g

# this is because errors which are zero cause problems in the fit routine
# this problem is fixed if they are replaced by NANs
for i in np.arange(len(dexpdata)):
    if dexpdata[i] < 1e-6:
        dexpdata[i] = np.nan


# this is now the finished data which we will fit to later
data = odr.RealData(T[fitmask], expdata[fitmask], sy=dexpdata[fitmask])


# fit a linear background to the data above a certain temperature Tmin ----------------------
# this will work as initial conditions for parameters C, and D in the final fit -------------

# ------------------------- Mn3Ge
Tmin = 390

# actually run the fitting routine for the linear background above Tmin
bpdata = odr.RealData(T[T>Tmin], expdata[T>Tmin])
initial_guess = [0,0,0]
fix = [0,1,1]
model = odr.Model(poly)
fit = odr.ODR(bpdata, model, beta0=initial_guess, ifixb=fix)
out = fit.run()
poptbg = out.beta
print(poptbg)



# give an initial guess for the fit parameters ----------------------------------------------
# a "1" in the second elemnt of each array means this parameter is actually varied ----------
# a "0" means that this parameter is kept fixed at the initial value ------------------------
# the order of parameters is: Tc, alpha, delta, A, B, C, D ----------------------------------
#--------------------------------------------------------------------------------------------

# ------------------------- Mn3Ge
guess = np.array([[369.9, 0], [0.35, 1], [0.529, 0], [-3.33, 1], [0, 0], [poptbg[0], 1], [poptbg[1], 1]])
guess = np.array([[369.9, 0], [0.35, 1], [10, 1], [-2.32, 1], [-1, 1], [poptbg[0], 1], [poptbg[1], 1]])
# guess = np.array([[369.9, 0], [0.35, 1], [0, 0], [-2.32, 1], [0, 0], [poptbg[0], 1], [poptbg[1], 1]])

initial_guess = guess[:,0]
fix = guess[:,1]



# set up and actually run the fit -----------------------------------------------------------
#--------------------------------------------------------------------------------------------
# model object
model = odr.Model(crit)
fit = odr.ODR(data, model, beta0=initial_guess, ifixb=fix)
    
# Run the regression.
out = fit.run()
popt = out.beta
perr = out.sd_beta


# this puts the fit results in a string so they can -----------------------------------------
# be displayed in the plot later ------------------------------------------------------------
result1 = 'Tc = ' + str(round(popt[0],2)) + '\n$\\alpha$ = ' + str(round(popt[1],4)) + ', $\\Delta$ = ' + str(round(popt[2],3)) 
result2 = '\nA = ' + str(round(popt[3],2)) + ', B = ' + str(round(popt[4],2))
result3 = '\nC = ' + str(round(popt[5],2)) + ', D = ' + str(round(popt[6],2))
result = result1 + result2 + result3

print (result)




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot the fit result ----------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
f1, ax = plt.subplots(figsize=(10,7))

# ------------------------- plot the fit results
Tint = np.linspace(popt[0], max(T), int(1e3))
ax.plot(Tint, crit(popt, Tint), zorder=2, c='tab:orange', label='negative bulk modulus', linewidth=3)


# ------------------------- plot the data
ax.fill_between(T, expdata-dexpdata, expdata+dexpdata, alpha=0.3, facecolor='lightgrey')
ax.plot(T[fitmask], expdata[fitmask], c='black', zorder=-2, linewidth=3)
ax.plot(T[fitmaskinverse], expdata[fitmaskinverse], c='lightgrey', linewidth=3, zorder=1)

# ------------------------- plot the fit results
text = plt.text(380, 44,  result, fontsize=16)

# ------------------------- other plot settings
ax.set_xlabel('T (K)', fontsize=18)
plt.ylabel('$\\mathrm{-B}$ (GPa - y-offset)', fontsize=18)
#ax.set_ylabel('specific heat', fontsize=18)
ax.set_xlim(min(T)-1, max(T)+1)

ax.tick_params(axis="y", direction="in", labelsize=15, left='True', right='True', length=4, width=1, which = 'major')
ax.tick_params(axis="x", direction="in", labelsize=15, bottom='True', top='False', length=4, width=1, which = 'major')
ax.xaxis.tick_bottom()

# ------------------------- this creates a second axis at the top with (T-Tc)/T instead of T
def Ttot (T):
    return (T-popt[0])/popt[0]

def ttoT (t):
    return (popt[0] + popt[0]*t)

secax = ax.secondary_xaxis('top', functions=(Ttot, ttoT))
secax.set_xlabel('$\\mathrm{(T - T_c)/T_c}$', fontsize=18)

secax.tick_params(axis="x", direction="in", labelsize=15, bottom='False', top='True', length=4, width=1, which = 'major')
secax.xaxis.tick_top()


plt.show()