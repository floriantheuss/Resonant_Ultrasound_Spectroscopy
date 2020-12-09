import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import AxesGrid
import os


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
expdata = -A1g1 - min(-A1g1)
#expdata = -Bulk - min(-Bulk)

# import fit results ----------------------------------------------------------------------
folder = "C:\\Users\\j111\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\A1g1_fits\\"
#project = "no_corrections-linear-meet.txt"
#project = "no_corrections-linear-dont_meet.txt"
#project = "no_corrections-qudratic-same.txt"
project = "fixed_correction_below_Tc-linear-same.txt"
#project = "correction_below_Tc-linear-same.txt"
#project = "bulk-modulus_fixed_correction_below_Tc-linear-same.txt"


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

# create a new array which gives you ones where the condition is fullfilled and zeroes where it isn't
delta = 0.004
deltaT = 0.1
maskdigital = (params[:,1] > -0.014-delta) & (params[:,1] < -0.014+delta) & (params[:,0] < 370+deltaT) & (params[:,0] > 370-deltaT)
digital = np.zeros(len(maskdigital))
for i in np.arange(len(maskdigital)):
    if maskdigital[i]==True:
        digital[i] = 1

fitrange = fits[:, :4]
fitrange = fitrange[maskdigital]

print(len(fitrange))


folder = "C:\\Users\\j111\\Box Sync\\Projects\\Mn3Ge\\critical exponent\\"
project = "A1g1_fits//A1g1_good_fit_ranges.txt"
filename = folder+project

if os.path.isfile(filename) == True:
    x='w'
else:
    x='x'

with open(filename, x) as g:
    g.write('T1' + '\t' + 'T2' + '\t' + 'T3' + '\t' + 'T4' + '\n')
    for i in np.arange(len(fitrange)):
        a = ''
        for j in fitrange[i]:
            a = a + str(j) + '\t'
        a = a[:-1] + '\n'
        g.write(a)




#############################################################################################################################
# plots
# here you need to pick the temperatures you want to look at
T1pick = 353
T2pick = 367.97
T3pick = 369.88
T4pick = 394
#############################################################################################################################

mask1 = (abs(T1-T1pick) - min(abs(T1-T1pick))) < 0.01
mask2 = (abs(T2-T2pick) - min(abs(T2-T2pick))) < 0.01
mask3 = (abs(T3-T3pick) - min(abs(T3-T3pick))) < 0.01
mask4 = (abs(T4-T4pick) - min(abs(T4-T4pick))) < 0.01
#mask23 = mask2 & mask3
mask14 = mask1 & mask4
maskone = mask1 & mask2 & mask3 & mask4

params14 = digital[mask14] # Tc	Am	Ap	alpha	Bm	Bp	Cm	Cp	A



# plot maskdigital ----------------------------------------------------------------------------------------------------------------------------------

figTc, axTc = plt.subplots(1, 1)

digital14 = params14.reshape((10, 10))
figTc2 = axTc.pcolor(T3reduced, T2reduced, digital14, cmap='inferno')
axTc.set_xlabel('T$_3$ (K)', fontsize=13)
axTc.set_ylabel('T$_2$ (K)', fontsize=13)
axTc.set_title('T$_1$ = ' + str(T1pick) + ' K, T$_4$ = ' + str(T4pick) +' K')


figTc.subplots_adjust(right=0.8)
cbar_ax = figTc.add_axes([0.85, 0.11, 0.02, .75])
figTc.colorbar(figTc2, cax=cbar_ax)

plt.show()
