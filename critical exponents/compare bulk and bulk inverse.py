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
project = "\\Mn3Ge\\RUS\\Mn3Ge_2007 A\\irreducible_elastic_constants.txt"


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
expdatainv = 1/Bulk - min(1/Bulk)


# plot the simulation together with the data -------------------------------------------------------------------------------------------------------------------------
f1 = plt.figure(figsize=(12,7))

# plot the inverse bulk modulus
B = 1
D = 0
E = 0
plt.scatter(T, B*expdatainv + D + E*T, marker='x', label='inverse bulk modulus')

# plot the data

plt.scatter(T, expdata, marker='o', label='-1 $\\times$ bulk modulus', zorder=-1)

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
plt.scatter(T, expdata, marker='o', s=1, label='-1 $\\times$ bulk modulus', zorder=-1)

plt.xlabel('$t = \\mathrm{(T-T_c)/T_c}$', fontsize = 18)
#plt.xlim(-0.08, 0.08)
plt.legend(fontsize=15)

# plot the simulation
B = 3840
D = 0
E = 0
l = plt.scatter(T, B*expdatainv + D + E*T, marker='x', s=1, label='inverse bulk modulus')

ax.margins(x=0)

axcolor = 'lightgrey'
axB = plt.axes([0.25, 0.13, 0.65, 0.02], facecolor=axcolor)
axD = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)
axE = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor=axcolor)


sB = Slider(axB, 'factor', 2500, 5000, valinit=3000, valstep=10, color='red')
sD = Slider(axD, 'offset', 5, 50, valinit=0, valstep=0.2, color='red')
sE = Slider(axE, 'slope', -0.05, 0.05, valinit=0, valstep=0.0001, color='red')


def update(val):
    B = sB.val
    D = sD.val
    E = sE.val
    simulation = B*expdatainv + D + E*T
    l.set_offsets(np.array(list(zip(T, simulation))))
    
 

sB.on_changed(update)
sD.on_changed(update)
sE.on_changed(update)


plt.show()
