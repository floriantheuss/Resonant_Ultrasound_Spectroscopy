import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid


# import irrdeducible elastic constants ----------------------------------------------------------------------
folder = "C:\\Users\\Florian\\Box Sync\\Projects"
project = "\\Mn3Ge\\RUS\\Mn3Ge_1.1\\irreducible_elastic_constants.txt"


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
#expdata = -A1g1 - min(-A1g1)
expdata = -Bulk - min(-Bulk)

# import fit results ----------------------------------------------------------------------
folder = "C:\\Users\\Florian\\Box Sync\\Projects\\Mn3Ge\\critical exponent_fitting\\Mn3Ge_1.1\\A1g1_fits\\"
#project = "no_corrections-linear-meet.txt"
#project = "no_corrections-linear-dont_meet.txt"
#project = "no_corrections-qudratic-same.txt"
#project = "fixed_correction_below_Tc-linear-same.txt"
#project = "correction_below_Tc-linear-same.txt"
project = "bulk-modulus_fixed_correction_below_Tc-linear-same.txt"


data = []
f = open(folder+project, 'r')
f.readline()
    
for line in f:
    line = line.strip()
    line = line.split()
    for i in np.arange(len(line)):
        line[i] = float(line[i])
    data.append(line)

fits = np.array(data)


# define the critical divergence
def crit (p, T):
    Tc, aminus, aplus, alpha, b, c, d = p
    tm = (Tc-T[T<=Tc])/Tc
    C1 = aminus * tm**(-alpha) 
    tp = (T[T>Tc]-Tc)/Tc
    C2 = aplus * tp**(-alpha)
    C = np.append(C1, C2) + b + c*T + d*T**2
    return C

# define the critical divergence with different linear background below and above Tc but they meet at Tc
def critdb (p, T):
    Tc, aminus, aplus, alpha, bminus, bplus, C = p
    tm = (Tc-T[T<=Tc])/Tc
    C1 = aminus * tm**(-alpha) + bminus*T[T<=Tc] + C
    tp = (T[T>Tc]-Tc)/Tc
    C2 = aplus * tp**(-alpha) + bplus*T[T>Tc] + C + (bminus-bplus)*Tc
    C = np.append(C1, C2)
    return C

# define the critical divergence with different linear background below and above Tc and they don't meet at Tc
def critdb1 (p, T):
    Tc, aminus, aplus, alpha, bminus, bplus, Cm, Cp = p
    tm = (Tc-T[T<=Tc])/Tc
    C1 = aminus * tm**(-alpha) + bminus*T[T<=Tc] + Cm
    tp = (T[T>Tc]-Tc)/Tc
    C2 = aplus * tp**(-alpha) + bplus*T[T>Tc] + Cp
    C = np.append(C1, C2)
    return C


# define the critical divergence with sqrt background below and linear above Tc
def critsqrt (p, T):
    Tc, aminus, aplus, alpha, bminus, bplus, Cminus, Cplus, A = p
    tm = (Tc-T[T<=Tc])/Tc
    C1 = aminus * tm**(-alpha) + bminus*T[T<=Tc] + Cminus + A*np.sqrt(tm)
    tp = (T[T>Tc]-Tc)/Tc
    C2 = aplus * tp**(-alpha) + bplus*T[T>Tc] + Cplus
    C = np.append(C1, C2)
    return C

# includes higher order terms
def crit_more (p, T):
    Tc, alpha, delta, Am, Ap, Bm, Bp, C, D, E  = p
    tm = (Tc-T[T<Tc])/Tc
    C1 = Am * tm**(-alpha) * ( 1 + Bm * tm**delta)
    tp = (T[T>Tc]-Tc)/Tc
    C2 = Ap * tp**(-alpha) * ( 1 + Bp * tp**delta)
    C = np.append(C1, C2) + C + D*T + E*T**2
    return C


# split the imported data into useful arrays -------------------------------------------------------------------------------------------------------------------------

T1 = fits[:,0]
T2 = fits[:,1]
T3 = fits[:,2]
T4 = fits[:,3]
params = fits[:,4:]

T1reduced = T1[::2000]
T2reduced = T2[:2000:200]
T3reduced = T3[:200:20]
T4reduced = T4[:20]

# find the best fit ------------------------------------------------------------------------------------------------------
chisquared = []
for i in np.arange(len(params)):
    maskchi = ((T>T1[i]) & (T<T2[i])) | ((T>T3[i]) & (T<T4[i]))
    x = T[maskchi]
    fitdata = crit_more(params[i], x)
    exp = expdata[maskchi]
    chisquared.append(sum(fitdata - exp)**2 / ( len(x)-7 ))
chisquared = np.array(chisquared)
index = np.arange(len(chisquared))[abs( chisquared - min(chisquared) ) < 1e-10]

for i in index:
    print ('fit range: ' + str(T1[i]) + ', ' + str(T2[i]) + ', ' + str(T3[i]) + ', ' + str(T4[i]) + '\n'
    + 'parameters: ' + str(params[i]) )

maskTc = (params[:,0]>369.9) & (params[:,0]<370.1) & (params[:,1]<0)
print ('T$_c$ = ' + str( round( np.mean(params[:,0][maskTc]) ,2) ) + ' +- ' + str( round( np.std(params[:,0][maskTc]) ,2) )   )
print ('$\\alpha$ = ' + str( round( np.mean(params[:,1][maskTc]) ,5) ) + ' +- ' + str( round( np.std(params[:,1][maskTc]) ,5) )   )
print ('Ap/Am = ' + str( round( np.mean((params[:,4][maskTc])/(params[:,3][maskTc])) ,3) ) + ' +- ' + str( round( np.std((params[:,4][maskTc])/(params[:,3][maskTc])) ,3) )   )
#############################################################################################################################
# plots
# here you need to pick the temperatures you want to look at
T1pick = 361.45
T2pick = 367.97
T3pick = 369.88
T4pick = 391.15
#############################################################################################################################

mask1 = (abs(T1-T1pick) - min(abs(T1-T1pick))) < 0.01
mask2 = (abs(T2-T2pick) - min(abs(T2-T2pick))) < 0.01
mask3 = (abs(T3-T3pick) - min(abs(T3-T3pick))) < 0.01
mask4 = (abs(T4-T4pick) - min(abs(T4-T4pick))) < 0.01
mask23 = mask2 & mask3
mask14 = mask1 & mask4
maskone = mask1 & mask2 & mask3 & mask4

params14 = params[mask14] # Tc	Am	Ap	alpha	Bm	Bp	Cm	Cp	A
params23 = params[mask23] # Tc	Am	Ap	alpha	Bm	Bp	Cm	Cp	A


Tc23 = params23[:,0]
alpha23 = params23[:,1]
delta = params23[:,2]
ApoAm23 = params23[:,4] / params23[:,3]

Tc14 = params14[:,0]
alpha14 = params14[:,1]
delta14 = params14[:,2]
ApoAm14 = params14[:,4] / params14[:,3]



# plot of the critical temperature ----------------------------------------------------------------------------------------------------------------------------------
maskTc23 = (Tc23>369) & (Tc23<371)
maskTc14 = (Tc14>369) & (Tc14<371)

figTc, axTc = plt.subplots(1, 2, figsize=(15,5))
figTc.suptitle("Tc", fontsize=20)


Tcnew23 = Tc23.reshape((20, 20))
figTc1 = axTc[0].contourf(T4reduced, T1reduced, Tcnew23, cmap='inferno', levels=np.linspace(367, 373, 100), extend='both')
axTc[0].set_xlabel('T$_4$ (K)', fontsize=13)
axTc[0].set_ylabel('T$_1$ (K)', fontsize=13)
axTc[0].set_title('T$_2$ = ' + str(T2pick) + ' K, T$_3$ = ' + str(T3pick) + ' K\n'
 + 'Tc = ' + str(round(np.mean(Tc23[maskTc23]), 2)) + ' +- ' + str(round(np.std(Tc23[maskTc23]),2)), fontsize=15)

Tcnew14 = Tc14.reshape((10, 10))
figTc2 = axTc[1].contourf(T3reduced, T2reduced, Tcnew14, cmap='inferno', levels=np.linspace(367, 373, 100), extend='both')
axTc[1].set_xlabel('T$_3$ (K)', fontsize=13)
axTc[1].set_ylabel('T$_2$ (K)', fontsize=13)
axTc[1].set_title('T$_1$ = ' + str(T1pick) + ' K, T$_4$ = ' + str(T4pick) +' K\n'
 + 'Tc = ' + str(round(np.mean(Tc14[maskTc14]),2)) + ' +- ' + str(round(np.std(Tc14[maskTc14]),2)), fontsize=15)

figTc.subplots_adjust(right=0.8)
cbar_ax = figTc.add_axes([0.85, 0.11, 0.02, .75])
figTc.colorbar(figTc2, cax=cbar_ax, label='Tc (K)')



# plot of the critical exponent alpha  ------------------------------------------------------------------------------------------------------------------------------
figa, axa = plt.subplots(1, 2, figsize=(15,5))
figa.suptitle("$\\alpha$", fontsize=20)

anew23 = alpha23.reshape((20, 20))
figa1 = axa[0].contourf(T4reduced, T1reduced, anew23, cmap='seismic', levels=np.linspace(-0.05, 0.05, 100), extend='both')
axa[0].set_xlabel('T$_4$ (K)', fontsize=13)
axa[0].set_ylabel('T$_1$ (K)', fontsize=13)
axa[0].set_title('T$_2$ = ' + str(T2pick) + ' K, T$_3$ = ' + str(T3pick) + ' K\n'
 + '$\\alpha$ = ' + str(round(np.mean(alpha23[maskTc23]), 5)) + ' +- ' + str(round(np.std(alpha23[maskTc23]),5)), fontsize=15)

anew14 = alpha14.reshape((10, 10))
figa2 = axa[1].contourf(T3reduced, T2reduced, anew14, cmap='seismic', levels=np.linspace(-0.05, 0.05, 100), extend='both')
axa[1].set_xlabel('T$_3$ (K)', fontsize=13)
axa[1].set_ylabel('T$_2$ (K)', fontsize=13)
axa[1].set_title('T$_1$ = ' + str(T1pick) + ' K, T$_4$ = ' + str(T4pick) + ' K\n'
 + '$\\alpha$ = ' + str(round(np.mean(alpha14[maskTc14]), 5)) + ' +- ' + str(round(np.std(alpha14[maskTc14]),5)), fontsize=15)

figa.subplots_adjust(right=0.8)
cbar_ax = figa.add_axes([0.85, 0.11, 0.02, .75])
figTc.colorbar(figa2, cax=cbar_ax, label='$\\alpha$')



# # # plot of the ratio A+/A-  ----------------------------------------------------------------------------------------------------------------------------------
figA, axA = plt.subplots(1, 2, figsize=(15,5))
figA.suptitle("A$^+/$A$^-$", fontsize=20)

Anew23 = ApoAm23.reshape((20, 20))
figA1 = axA[0].contourf(T4reduced, T1reduced, Anew23, 100, cmap='seismic', levels=np.linspace(0.5, 1.5, 100), extend='both')
axA[0].set_xlabel('T$_4$ (K)', fontsize=13)
axA[0].set_ylabel('T$_1$ (K)', fontsize=13)
axA[0].set_title('T$_2$ = ' + str(T2pick) + ' K, T$_3$ = ' + str(T3pick) + ' K\n'
 + 'A$^+/$A$^-$ = ' + str(round(np.mean(ApoAm23[maskTc23]), 2)) + ' +- ' + str(round(np.std(ApoAm23[maskTc23]),2)), fontsize=15)

Anew14 = ApoAm14.reshape((10, 10))
figA2 = axA[1].contourf(T3reduced, T2reduced, Anew14, 100, cmap='seismic', levels=np.linspace(0.5, 1.5, 100), extend='both')
axA[1].set_xlabel('T$_3$ (K)', fontsize=13)
axA[1].set_ylabel('T$_2$ (K)', fontsize=13)
axA[1].set_title('T$_1$ = ' + str(T1pick) + ' K, T$_4$ = ' + str(T4pick) + ' K\n'
 + 'A$^+/$A$^-$ = ' + str(round(np.mean(ApoAm14[maskTc14]), 2)) + ' +- ' + str(round(np.std(ApoAm14[maskTc14]),2)), fontsize=15)

figA.subplots_adjust(right=0.8)
cbar_ax = figA.add_axes([0.85, 0.11, 0.02, .75])
figA.colorbar(figA2, cax=cbar_ax, label='A$^+/$A$^-$')


# # #######################################################################################################################################################################
# plot of the data with a specific fit
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.scatter(T, expdata, s=2, c='black', zorder=-2)


#mask1 = (abs(T1-360) - min(abs(T1-360))) < 0.01
#mask2 = (abs(T2-T2reduced[0]) - min(abs(T2-T2reduced[0]))) < 0.01
#mask3 = (abs(T3-T3reduced[0]) - min(abs(T3-T3reduced[0]))) < 0.01
#mask4 = (abs(T4-T4reduced[0]) - min(abs(T4-T4reduced[0]))) < 0.01
#mask23 = mask2 & mask3
#mask14 = mask1 & mask4
#maskone = mask1 & mask2 & mask3 & mask4
n, = plt.plot(T, crit_more(params[maskone][0], T))
m, = plt.plot(T, crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T)
    -crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T[-1]), '--', c='grey')
ymax = max([max(crit_more(p, T)) for p in params ])
ymin = min([min(crit_more(p, T)) for p in params ])
plt.ylim(-5, 46)
plt.xlabel('T ((K)', fontsize=18)
plt.ylabel('-C - min(C) (GPa)', fontsize=18)


text = ax.text(380, 10,  'Tc = ' + str(round(params[maskone][0,0], 2)) + ' K\n'
    + '$\\alpha$ = ' + str(round(params[maskone][0,1], 5)) + '\n'
    + '$\\delta$ = ' + str(round(params[maskone][0,2], 5)) + '\n'
    + 'A$^+$/A$^-$ = ' +  str(round((params[maskone][0,4])/(params[maskone][0,3]), 3)) + '\n'
    + 'A$^+$ = ' + str(round((params[maskone][0,4]),1)) + '\n'
    +  'A$^-$ = ' + str(round((params[maskone][0,3]),1)) + '\n'
    + 'B$^+$ = ' + str(round(params[maskone][0,6], 5)) + '\n'
    + 'B$^-$ = ' + str(round(params[maskone][0,5], 5))
    )


fitmask = ((T>T1reduced[0]) & (T<T2reduced[0])) | ((T>T3reduced[0]) & (T<T4reduced[0]))
fitmaskinverse =  np.invert(fitmask)
fitydata = expdata[fitmaskinverse]
fitxdata = T[fitmaskinverse]

l = plt.scatter(fitxdata, fitydata, s=2, c='red', zorder=-1)
ax.margins(x=0)

axcolor = 'lightgrey'
axT1 = plt.axes([0.25, 0.17, 0.65, 0.02], facecolor=axcolor)
axT2 = plt.axes([0.25, 0.13, 0.65, 0.02], facecolor=axcolor)
axT3 = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)
axT4 = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor)

sT1 = Slider(axT1, 'T1 (K)', min(T1reduced), max(T1reduced), valinit=T1pick, valstep=(max(T1reduced)-min(T1reduced))/len(T1reduced), color='red')
sT2 = Slider(axT2, 'T2 (K)', min(T2reduced), max(T2reduced), valinit=T2pick, valstep=(max(T2reduced)-min(T2reduced))/len(T2reduced), color='red')
sT3 = Slider(axT3, 'T3 (K)', min(T3reduced), max(T3reduced), valinit=T3pick, valstep=(max(T3reduced)-min(T3reduced))/len(T3reduced), color='red')
sT4 = Slider(axT4, 'T4 (K)', min(T4reduced), max(T4reduced), valinit=T4pick, valstep=(max(T4reduced)-min(T4reduced))/len(T4reduced), color='red')


def update(val):
    T1pick = sT1.val
    T2pick = sT2.val
    T3pick = sT3.val
    T4pick = sT4.val
    
    fitmask = ((T>T1pick) & (T<T2pick)) | ((T>T3pick) & (T<T4pick))
    fitmaskinverse =  np.invert(fitmask)
    fitydata = expdata[fitmaskinverse]
    fitxdata = T[fitmaskinverse]
    l.set_offsets(np.array(list(zip(fitxdata, fitydata))))

    mask1 = (abs(T1-T1pick) - min(abs(T1-T1pick))) < 0.01
    mask2 = (abs(T2-T2pick) - min(abs(T2-T2pick))) < 0.01
    mask3 = (abs(T3-T3pick) - min(abs(T3-T3pick))) < 0.01
    mask4 = (abs(T4-T4pick) - min(abs(T4-T4pick))) < 0.01
    maskone = mask1 & mask2 & mask3 & mask4
    
    text.set_text('Tc = ' + str(round(params[maskone][0,0], 2)) + ' K\n'
    + '$\\alpha$ = ' + str(round(params[maskone][0,1], 5)) + '\n'
    + '$\\delta$ = ' + str(round(params[maskone][0,2], 5)) + '\n'
    + 'A$^+$/A$^-$ = ' +  str(round((params[maskone][0,4])/(params[maskone][0,3]), 3)) + '\n'
    + 'A$^+$ = ' + str(round((params[maskone][0,4]),1)) + '\n'
    + 'A$^-$ = ' + str(round((params[maskone][0,3]),1)) + '\n'
    + 'B$^+$ = ' + str(round(params[maskone][0,6], 5)) + '\n'
    + 'B$^-$ = ' + str(round(params[maskone][0,5], 5))
    )


    n.set_ydata(crit_more(params[maskone][0], T))
    m.set_ydata(crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T)
        -crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T[-1]))
    fig.canvas.draw_idle()
 

sT1.on_changed(update)
sT2.on_changed(update)
sT3.on_changed(update)
sT4.on_changed(update)



plt.show()


# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)


# Tcnew = Tc23.reshape((20, 20))
# Tcplot = plt.contourf(T1reduced, T4reduced, Tcnew, cmap='inferno', levels=np.linspace(367, 373, 100), extend='both')
# plt.xlabel('T1 (K)')
# plt.ylabel('T4 (K)')
# #plt.clim(367, 373)
# plt.colorbar(label='T$_c$')
# plt.title('T$_c$')


# axcolor = 'lightgrey'
# ax1T2 = plt.axes([0.25, 0.13, 0.65, 0.02], facecolor=axcolor)
# ax1T3 = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)

# s1T2 = Slider(ax1T2, 'T2 (K)', min(T2reduced), max(T2reduced), valinit=T2reduced[0], valstep=(max(T2reduced)-min(T2reduced))/len(T2reduced), color='red')
# s1T3 = Slider(ax1T3, 'T3 (K)', min(T3reduced), max(T3reduced), valinit=T3reduced[0], valstep=(max(T3reduced)-min(T3reduced))/len(T3reduced), color='red')
# #plt.draw()

# def updated(val):
#     T2pick = s1T2.val
#     T3pick = s1T3.val
#     mask2 = (abs(T2-T2pick) - min(abs(T2-T2pick))) < 0.01
#     mask3 = (abs(T3-T3pick) - min(abs(T3-T3pick))) < 0.01
#     mask23 = mask2 & mask3
#     params23 = params[mask23]
#     Tc23 = params23[:,0]
#     Tcnew = Tc23.reshape((20, 20))

#     Tcplot = plt.contourf(T1reduced, T4reduced, Tcnew, cmap='inferno', levels=np.linspace(367, 373, 100), extend='both')
#     for coll in Tcplot.collections:
#         #plt.gca().collections.remove(coll)
#         coll.remove()
   

#     Tcplot = plt.contourf(T1reduced, T4reduced, Tcnew, cmap='inferno', levels=np.linspace(367, 373, 100), extend='both')
#     plt.xlabel('T1 (K)')
#     plt.ylabel('T4 (K)')
#     #plt.clim(367, 373)
#     plt.colorbar(label='T$_c$')
#     plt.title('T$_c$')
#     return Tcplot.collections

#   #  plt.draw()
        
# s1T2.on_changed(updated)
# s1T3.on_changed(updated)





