import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt
from scipy.optimize import curve_fit
from scipy import odr


# import irrdeducible elastic constants ----------------------------------------------------------------------
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


# define the critical divergence
def crit (p, T):
    Tc, aminus, aplus, alpha, b, c, d = p
    tm = (Tc-T[T<=Tc])/Tc
    C1 = aminus * tm**(-alpha) 
    tp = (T[T>Tc]-Tc)/Tc
    C2 = aplus * tp**(-alpha)
    C = np.append(C1, C2) + b + c*T + d*T**2
    return C


# includes higher order terms
def crit_more (p, T):
    Tc, alpha, delta, Am, Ap, Bm, Bp, C, D, E  = p
    tm = (Tc-T[T<=Tc])/Tc
    tp = (T[T>Tc]-Tc)/Tc
    if delta >= 0:# and alpha<0 and Am<0 and Ap<0:
        C1 = Am * tm**(-alpha) * ( 1 + Bm * tm**delta)
        C2 = Ap * tp**(-alpha) * ( 1 + Bp * tp**delta)
        C = np.append(C1, C2) + C + D*T + E*T**2
    else:
        C = 0 * T
    return C



# includes higher order terms and distribution of critical temperature
def crit_most (p, T):
    dTc, Tc, alpha, Am, Ap, bm, bp, B, C, delta = p

    Tcrit = np.random.normal(Tc, dTc, 10000)
    Tcrit = Tcrit[Tcrit<=Tc]
    Ctotal = np.zeros(len(T))
    for i in Tcrit:
        tm = (i-T[T<=i])/i
        C1 = Am * tm**(-alpha) * ( 1 + bm * tm**delta) + B*T[T<=i] + C
        tp = (T[T>i]-i)/i
        C2 = Ap * tp**(-alpha) * ( 1 + bp * tp**delta) + B*T[T>i] + C
        Ctotal = Ctotal + np.append(C1, C2)
    Ctotal = Ctotal / len(Tcrit)
    return Ctotal


# define Varshni background
def bkg (p, T):
    C0, A, Theta = p
    C = C0 - A/( np.exp(Theta/T) - 1 )
    return C

# define second order polynomial
def poly (p, T):
    A, B, C = p
    y = A*T**2 + B*T + C
    return y



# fit to elastic constant ------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
# set fit range #######################################################
# ------------------------- Mn3Ge
T1 = 352#0
T2 = 367.9
T3 = 370.36#2.221
T4 = 396
# ------------------------- Mn3.019Sn0.981
# T1 = 380
# T2 = 410
# T3 = 415.8
# T4 = 438
fitmask = ((T>T1) & (T<T2)) | ((T>T3) & (T<T4))
fitmaskinverse =  np.invert(fitmask)


expdata = -A1g1 - min(-A1g1)
expdata = -Bulk - min(-Bulk)
dexpdata = dBulk
for i in np.arange(len(dexpdata)):
    if dexpdata[i] < 1e-6:
        dexpdata[i] = np.nan


# fit background -----------------------------------------------------------------------------------------------------------------------------


Tmin = 390 # Mn3Ge
bpdata = odr.RealData(T[T>Tmin], expdata[T>Tmin])
initial_guess = [0,0,0]
fix = [0,1,1]
model = odr.Model(poly)
fit = odr.ODR(bpdata, model, beta0=initial_guess, ifixb=fix)
out = fit.run()
poptp = out.beta
print(poptp)




#x = (T[fitmask]-2.216)/2.16 # here I am exponentially weighing the data with more weight closer to Tc
#sigma = 10*np.e**(-.01/abs(x))+1e-10      # Tc is assumed to be 369.8 K
data = odr.RealData(T[fitmask], expdata[fitmask], sy=dexpdata[fitmask])


    
# Set up ODR with the model and data #################################################################

# these are for crit_more #############################################
# parameters are: Tc, alpha, delta, Am, Ap, Bm, Bp, C (constant), D (linear), E (quadratic)
guess = np.array([[369.9, 1], [-0.0121, 1], [0.529, 0], [-118.53, 1], [-132.67, 1], [.34, 1], [0, 0], [poptp[2], 1], [poptp[1], 1], [0, 0]]) #Mn3Ge bulk
# guess = np.array([[415.25, 1], [-0.01236, 1], [0.529, 0], [-100, 1], [-100, 1], [0, 1], [0, 0], [poptp[2], 1], [poptp[1], 1], [0, 0]]) #Mn3Sn
# initial_guessmore = guessmore[:,0]
# fixmore = guessmore[:,1]


initial_guess = guess[:,0]
fix = guess[:,1]




# model object
model = odr.Model(crit_more)

fit = odr.ODR(data, model, beta0=initial_guess, ifixb=fix)
    
# Run the regression.
out = fit.run()
popt = out.beta
perr = out.sd_beta



# fit results in a string
result1 = 'Tc = ' + str(round(popt[0],2)) + '\n$\\alpha$ = ' + str(round(popt[1],4)) + ', $\\Delta$ = ' + str(round(popt[2],3)) 
result2 = '\nAm = ' + str(round(popt[3],2)) + ', Ap = ' + str(round(popt[4],2)) + '\nAp/Am = ' + str(round(popt[4]/popt[3],3)) 
result3 = '\nBm = ' + str(round(popt[5],2)) + ', Bp = ' + str(round(popt[6],2))
result4 = '\nC = ' + str(popt[7]) + ', D = ' + str(popt[8]) + ', E = ' + str(popt[9])
result = result1 + result2 + result3


print (result)


f1, ax = plt.subplots(figsize=(10,7))


Tint = np.linspace(min(T), max(T), int(1e3))
# fit results
ax.plot(T, crit_more(popt, T), zorder=1, c='tab:orange', label='negative bulk modulus', linewidth=2) # comlete fit


#data
ax.fill_between(T, expdata-dBulk, expdata+dBulk, alpha=0.3)
plotmask1 = ( ((T>T1) & (T<T2)) | ((T>T3) & (T<T4)) ) & (T>368)
plotmask2 = ( ((T>T1) & (T<T2)) | ((T>T3) & (T<T4)) ) & (T<368)
ax.plot(T[plotmask1], expdata[plotmask1], c='black', zorder=-2, linewidth=2)
ax.plot(T[plotmask2], expdata[plotmask2], c='black', zorder=-2, linewidth=2)
ax.plot(T[fitmaskinverse], expdata[fitmaskinverse], c='lightgrey', zorder=-1, linewidth=2)


#plt.title(result)
text = plt.text(375, 5,  result, fontsize=16)
#plt.legend(fontsize=16)
ax.set_xlabel('T (K)', fontsize=18)
plt.ylabel('$\\mathrm{-B}$ (GPa - y-offset)', fontsize=18)
#ax.set_ylabel('specific heat', fontsize=18)
ax.set_xlim(min(T)-1, max(T)+1)

ax.tick_params(axis="y", direction="in", labelsize=15, left='True', right='True', length=4, width=1, which = 'major')
ax.tick_params(axis="x", direction="in", labelsize=15, bottom='True', top='False', length=4, width=1, which = 'major')
ax.xaxis.tick_bottom()


def Ttot (T):
    return (T-popt[0])/popt[0]

def ttoT (t):
    return (popt[0] + popt[0]*t)

secax = ax.secondary_xaxis('top', functions=(Ttot, ttoT))
secax.set_xlabel('$\\mathrm{(T - T_c)/T_c}$', fontsize=18)

secax.tick_params(axis="x", direction="in", labelsize=15, bottom='False', top='True', length=4, width=1, which = 'major')
secax.xaxis.tick_top()





plt.show()




# #############################################################################################################################################
# # fit for several temperature ranges
# N = 20
# M = 10

# T1 = np.linspace(353, 366, N)
# T2 = np.linspace(366.5, 368.6, M)
# T3 = np.linspace(369.4, 371, M)
# T4 = np.linspace(375, 394, N)


# Tfit = np.array( [[[[[i,j, k, l] for l in T4] for k in T3] for j in T2] for i in T1] )
# Tfit = np.reshape(Tfit, (M**2*N**2,4))

# import the good fit ranges from A1g1 fits
# folder = "C:\\Users\\j111\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\"
# project = "A1g1_fits//A1g1_good_fit_ranges.txt"

# Tfit = []
# f = open(folder+project, 'r')
    
# f.readline()
    
# for line in f:
#     line = line.strip()
#     line = line.split()
#     for i in np.arange(len(line)):
#         line[i] = float(line[i])
#     Tfit.append(line)

# Tfit = np.array(Tfit)


# guess = np.array([[370, 0], [-0.014, 1], [0.529, 1], [-205, 1], [-219, 1], [0, 1], [0, 0], [poptp[2], 1], [poptp[1], 1], [0, 0]])
# initial_guess = guess[:,0]
# fix = guess[:,1]


# fit_result = []
# for i in Tfit:
#     fitmask = ((T>i[0]) & (T<i[1])) | ((T>i[2]) & (T<i[3]))
#     #fitmaskinverse =  np.invert(fitmask)

#     #x = (T[fitmask]-369.8)/369.8 # here I am exponentially weighing the data with more weight closer to Tc
#     #sigma = 50*np.e**(-.01/abs(x))+1e-10      # Tc is assumed to be 369.8 K
#     data = odr.RealData(T[fitmask], expdata[fitmask])#, sy=sigma)

#     fit = odr.ODR(data, model1, beta0=initial_guess, ifixb=fix)
#     out = fit.run()
#     fit_result.append(out.beta)
#     print(i)

# folder = "C:\\Users\\j111\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\"
# project = "A1g1_fits//E2g_fixed_correction_below_Tc-linear-same_fixed_Tc_370.txt"
# filename = folder+project

# if os.path.isfile(filename) == True:
#     x='w'
# else:
#     x='x'

# with open(filename, x) as g:
#     g.write('T1' + '\t' + 'T2' + '\t' + 'T3' + '\t' + 'T4' + '\t' + 'Tc' + '\t' + 'alpha' + '\t' + 'delta' + '\t' + 'Am' + '\t' + 'Ap' + '\t' + 'Bm' + '\t' +  'Bp' + '\t' + 'C' + '\t' + 'D' + '\t' + 'E' + '\n')
#     for i in np.arange(len(Tfit)):
#         a = ''
#         for j in Tfit[i]:
#             a = a + str(j) + '\t'
#         for j in fit_result[i]:
#             a = a + str(j) + '\t'
#         a = a[:-1] + '\n'
#         g.write(a)


#################################################################################################################################################
