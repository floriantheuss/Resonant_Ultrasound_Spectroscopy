import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid


# import irrdeducible elastic constants ----------------------------------------------------------------------
folder = "C:/Users/j111/Box Sync/Projects"
project = "/Mn3Ge/RUS/sample_with_green_face_from_010920/irreducible_elastic_constants.txt"


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
expdata = -E2g - min(-E2g)

# import fit results ----------------------------------------------------------------------
folder = "C:\\Users\\j111\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\A1g1_fits\\"
#project = "no_corrections-linear-meet.txt"
#project = "no_corrections-linear-dont_meet.txt"
#project = "no_corrections-qudratic-same.txt"
#project = "fixed_correction_below_Tc-linear-same.txt"
#project = "correction_below_Tc-linear-same.txt"
project = "bulk-modulus_fixed_correction_below_Tc-linear-same.txt"
project = "E2g_fixed_correction_below_Tc-linear-same_fixed_Tc_370.txt"



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


#############################################################################################################################
# plots
# here you need to pick the temperatures you want to look at
T1pick = T1[0]
T2pick = T2[0]
T3pick = T3[0]
T4pick = T4[0]
#############################################################################################################################

mask1 = (abs(T1-T1pick) - min(abs(T1-T1pick))) < 0.01
mask2 = (abs(T2-T2pick) - min(abs(T2-T2pick))) < 0.01
mask3 = (abs(T3-T3pick) - min(abs(T3-T3pick))) < 0.01
mask4 = (abs(T4-T4pick) - min(abs(T4-T4pick))) < 0.01

maskone = mask1 & mask2 & mask3 & mask4


# # #######################################################################################################################################################################
# plot of the data with a specific fit
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.scatter(T, expdata, s=2, c='black', zorder=-2)


n, = plt.plot(T, crit_more(params[maskone][0], T))
m, = plt.plot(T, crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T)
    -crit_more(np.array([params[maskone][0][0], 0, 0, 0, 0, 0, 0] + list(params[maskone][0][7:])), T[-1]), '--', c='grey')
ymax = max([max(crit_more(p, T)) for p in params ])
ymin = min([min(crit_more(p, T)) for p in params ])
plt.ylim(-3, 7)
plt.xlabel('T ((K)', fontsize=18)
plt.ylabel('-C - min(C) (GPa)', fontsize=18)


text = ax.text(380, 2,  'Tc = ' + str(round(params[maskone][0,0], 2)) + ' K\n'
    + '$\\alpha$ = ' + str(round(params[maskone][0,1], 5)) + '\n'
    + '$\\delta$ = ' + str(round(params[maskone][0,2], 5)) + '\n'
    + 'A$^+$/A$^-$ = ' +  str(round((params[maskone][0,4])/(params[maskone][0,3]), 3)) + '\n'
    + 'A$^+$ = ' + str(round((params[maskone][0,4]),1)) + '\n'
    +  'A$^-$ = ' + str(round((params[maskone][0,3]),1)) + '\n'
    + 'B$^+$ = ' + str(round(params[maskone][0,6], 5)) + '\n'
    + 'B$^-$ = ' + str(round(params[maskone][0,5], 5))
    )


fitmask = ((T>T1[0]) & (T<T2[0])) | ((T>T3[0]) & (T<T4[0]))
fitmaskinverse =  np.invert(fitmask)
fitydata = expdata[fitmaskinverse]
fitxdata = T[fitmaskinverse]

l = plt.scatter(fitxdata, fitydata, s=2, c='red', zorder=-1)
ax.margins(x=0)

axcolor = 'lightgrey'
axT1 = plt.axes([0.25, 0.1, 0.65, 0.02], facecolor=axcolor)


sT1 = Slider(axT1, 'T1 (K)', 0, len(T1)-1, valinit=0, valstep=1, color='red')


def update(val):
    T1pick = T1[int(sT1.val)]
    T2pick = T2[int(sT1.val)]
    T3pick = T3[int(sT1.val)]
    T4pick = T4[int(sT1.val)]
    
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


plt.show()