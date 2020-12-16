import numpy as np
import time
from lmfit import minimize, Parameters, report_fit
from scipy import odr
import matplotlib.pyplot as plt


## import data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
E2g = data[5]
dE2g = data[10]


## Initial parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Tc_ini = 369.9  # in K
Tc_int = [368, 371]  # in K
Tc_vary = False

alpha_ini = 0.35
alpha_int = [0, 0.8] 
alpha_vary = True

delta_ini = 0.1   
delta_int = [0, 100] 
delta_vary = True

A_ini = -2 
A_int = [-10, 0] 
A_vary = True

B_ini = -1
B_int = [-50, 0]
B_vary = True


# fit a linear background to the data above a certain temperature Tmin ----------------------
# this will work as initial conditions for parameters C, and D in the final fit -------------

# ------------------------- Mn3Ge
Tmin = 390

# define second order polynomial
def poly (p, T):
    A, B, C = p
    y = A + B*T + C*T**2
    return y
# actually run the fitting routine for the linear background above Tmin
bpdata = odr.RealData(T[T>Tmin], E2g[T>Tmin])
initial_guess = [0,0,0]
fix = [1,1,0]
model = odr.Model(poly)
fit = odr.ODR(bpdata, model, beta0=initial_guess, ifixb=fix)
out = fit.run()
poptbg = out.beta


C_ini = poptbg[0]
C_int = [poptbg[0]/100, poptbg[0]*100]
C_vary = True

D_ini = poptbg[1] 
D_int = [poptbg[1]/1e2, poptbg[1]*1e2]
D_vary = True




## Initialize fit parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
pars = Parameters()
pars.add("Tc", value=Tc_ini, vary=Tc_vary, min=Tc_int[0], max=Tc_int[-1])
pars.add("alpha", value=alpha_ini, vary=alpha_vary, min=alpha_int[0], max=alpha_int[-1])
pars.add("delta", value=delta_ini, vary=delta_vary, min=delta_int[0], max=delta_int[-1])
pars.add("A", value=A_ini, vary=A_vary, min=A_int[0], max=A_int[-1])
pars.add("B", value=B_ini, vary=B_vary, min=B_int[0], max=B_int[-1])
pars.add("C", value=C_ini, vary=C_vary, min=C_int[0], max=C_int[-1])
pars.add("D", value=D_ini, vary=D_vary, min=D_int[0], max=D_int[-1])



## Residual function >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
fitmask = T>370.5
fitmaskinverse =  np.invert(fitmask)

def residual_func(pars):
    ## Get fit parameters
    Tc = pars["Tc"].value
    alpha = pars["alpha"].value
    delta = pars["delta"].value
    A = pars["A"].value
    B = pars["B"].value
    C = pars["C"].value
    D = pars["D"].value

    print("Tc = ", np.round(Tc, 3), " K")
    print("alpha = ", np.round(alpha, 3))
    print("delta = ", np.round(delta, 3))
    print("A = ", np.round(A, 3))
    print("B = ", np.round(B, 3))
    print("C = ", np.round(C, 3))
    print("D = ", np.round(D, 3))

    ## Compute predicted shear modulus
    t = (T[fitmask]-Tc)/Tc
    shear_prediction = A * t**(-alpha) * ( 1 + B * t**delta) + C + D*T[fitmask]

    return shear_prediction - E2g[fitmask]



## Run fit algorithm >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
out = minimize(residual_func, pars, method='differential_evolution')
# out = minimize(residual_func, pars, method='shgo',
#                sampling_method='sobol', options={"f_tol": 1e-16}, n = 100, iters=20)

## Display fit report >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
report_fit(out)

fit_results = np.array( [param.value for name, param in out.params.items()] )
Tsim = np.linspace(fit_results[0]+0.01, max(T), int(1e3))
tsim = (Tsim-fit_results[0])/fit_results[0]
E2g_sim = fit_results[3] * tsim**(-fit_results[1]) * ( 1 + fit_results[4] * tsim**fit_results[2]) + fit_results[5] + fit_results[6]*Tsim



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot the fit result ----------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
f1, ax = plt.subplots(figsize=(10,7))

# ------------------------- plot the fit results
ax.plot(Tsim, E2g_sim, zorder=2, c='tab:orange', label='negative bulk modulus', linewidth=3)


# ------------------------- plot the data
ax.fill_between(T, E2g-dE2g, E2g+dE2g, alpha=0.3, facecolor='lightgrey')
ax.plot(T[fitmask], E2g[fitmask], c='black', zorder=-2, linewidth=3)
ax.plot(T[fitmaskinverse], E2g[fitmaskinverse], c='lightgrey', linewidth=3, zorder=1)

# ------------------------- other plot settings
ax.set_xlabel('T (K)', fontsize=18)
plt.ylabel('$\\mathrm{c_{E2g}}$ (GPa)', fontsize=18)
#ax.set_ylabel('specific heat', fontsize=18)
ax.set_xlim(min(T)-1, max(T)+1)

ax.tick_params(axis="y", direction="in", labelsize=15, left='True', right='True', length=4, width=1, which = 'major')
ax.tick_params(axis="x", direction="in", labelsize=15, bottom='True', top='False', length=4, width=1, which = 'major')
ax.xaxis.tick_bottom()

# ------------------------- this creates a second axis at the top with (T-Tc)/T instead of T
def Ttot (T):
    return (T-fit_results[0])/fit_results[0]

def ttoT (t):
    return (fit_results[0] + fit_results[0]*t)

secax = ax.secondary_xaxis('top', functions=(Ttot, ttoT))
secax.set_xlabel('$\\mathrm{(T - T_c)/T_c}$', fontsize=18)

secax.tick_params(axis="x", direction="in", labelsize=15, bottom='False', top='True', length=4, width=1, which = 'major')
secax.xaxis.tick_top()


plt.show()