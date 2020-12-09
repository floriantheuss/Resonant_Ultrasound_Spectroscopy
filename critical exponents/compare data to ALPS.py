import numpy as np
import matplotlib.pyplot as plt
import os.path
import os
from scipy.interpolate import interp1d
from scipy.signal import butter,filtfilt
from scipy.optimize import curve_fit
from scipy import odr
from matplotlib.widgets import Slider


# import irrdeducible elastic constants ----------------------------------------------------------------------
folder = "C:\\Users\\Florian\\Box Sync\\Projects"
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
expdata = -Bulk - min(-Bulk)


# import ALPS simulation of specific heat ---------------------------------------------------------------------------------------------------------------------------------------
folder = "C:\\Users\\Florian\\Box Sync\\Projects"
project = "\\Mn3Ge\\critical exponent\\ALPS simulations\\XY-cubic-20-slow.txt"


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

Ts = data[0]
sort_mask = np.argsort(Ts)
C = data[1][sort_mask]
dC = data[2][sort_mask]
Ts = Ts[sort_mask]



# plot the simulation together with the data -------------------------------------------------------------------------------------------------------------------------
f1 = plt.figure(figsize=(12,7))

# plot the simulation results
Tcs = 2.21
ts = (Ts-Tcs)/Tcs
B = 6.5
D = -5
E = 0
plt.scatter(ts, B*C + D + E*ts, marker='x', label='specific heat (ALPS)')

# plot the data
Tc = 370.09
t = (T-Tc)/Tc
plt.scatter(t, expdata, marker='o', label='bulk modulus (experimental)', zorder=-1)

plt.xlabel('$t = \\mathrm{(T-T_c)/T_c}$', fontsize = 18)
#plt.xlim(-0.08, 0.08)
plt.legend(fontsize=15)



# inter = interp1d(t, expdata)
# tmin = max([min(t), min(ts)])
# tmax = min([max(t), max(ts)])

# tnew = ts[(ts/10>tmin) & (ts/10<tmax)]/10
# plt.scatter(tnew, inter(tnew))


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
# plot the data
Tc = 370.09
t = (T-Tc)/Tc
plt.scatter(t, expdata, marker='o', label='bulk modulus (experimental)', zorder=-1)

plt.xlabel('$t = \\mathrm{(T-T_c)/T_c}$', fontsize = 18)
#plt.xlim(-0.08, 0.08)
#plt.legend(fontsize=15)

# plot the simulation
Tcs = 2.21
ts = (Ts-Tcs)/Tcs
B = 6.5
D = -5
E = 0
l = plt.scatter(ts, B*C + D + E*ts, marker='x', label='specific heat (ALPS)')

ax.margins(x=0)

axcolor = 'lightgrey'
axts = plt.axes([0.25, 0.17, 0.65, 0.02], facecolor=axcolor)
axB = plt.axes([0.25, 0.13, 0.65, 0.02], facecolor=axcolor)
axD = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)
axE = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor)

sts = Slider(axts, 'Tc (K)', 2.1, 2.3, valinit=2.2, valstep=0.01, color='red')
sB = Slider(axB, 'factor', 1, 15, valinit=0, valstep=.5, color='red')
sD = Slider(axD, 'offset', -100, 0, valinit=0, valstep=1, color='red')
sE = Slider(axE, 'slope', -5, 5, valinit=0, valstep=1, color='red')


def update(val):
    Tcs = sts.val
    ts = (Ts-Tcs)/Tcs
    B = sB.val
    D = sD.val
    E = sE.val
    simulation = B*C + D + E*ts
    l.set_offsets(np.array(list(zip(ts, simulation))))
    plt.ylim(min(simulation), max(simulation))
    #fig.canvas.draw_idle()
 

sts.on_changed(update)
sB.on_changed(update)
sD.on_changed(update)
sE.on_changed(update)


plt.show()
