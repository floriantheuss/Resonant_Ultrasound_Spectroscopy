import numpy as np
import matplotlib.pyplot as plt
import json



# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# specify file paths
# Mn3Ge
Mn3Ge = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out_elastic_constants.json'
Mn3Ge_error = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3Ge\\RUS\\Mn3Ge_2001B\\python fits\\Mn3Ge_out_elastic_constants_error.json'
# Mn3.1Sn0.89
Mn3Sn = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out_elastic_constants.json'
Mn3Sn_error = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\Mn3.1Sn0.89\\RUS\\2010A\\python fits\\Mn3.1Sn0.89_out_elastic_constants_error.json'


# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# import elastic constants with errors
T = {}
error = {}
elastic_constants = {}
with open(Mn3Ge) as json_file:
    data = json.load(json_file)
    T['Mn3Ge'] = data['temperature']
    elastic_constants['Mn3Ge'] = data['elastic constants']
with open(Mn3Ge_error) as json_file:
    data = json.load(json_file)
    error['Mn3Ge'] = data
with open(Mn3Sn) as json_file:
    data = json.load(json_file)
    T['Mn3Sn'] = data['temperature']
    elastic_constants['Mn3Sn'] = data['elastic constants']
with open(Mn3Sn_error) as json_file:
    data = json.load(json_file)
    error['Mn3Sn'] = data

dc_dict = {}
for crystal, el_const in elastic_constants.items():
    elastic_constants_intermediate_dict = {}
    for irrep, values in el_const.items():
        elastic_constants[crystal][irrep] = np.array( elastic_constants[crystal][irrep] )
        dc = (np.array(values) - np.array(values)[-1]) / 1e9
        elastic_constants_intermediate_dict[irrep] = dc
        error[crystal][irrep] = np.array(error[crystal][irrep]) / 1e9
    dc_dict[crystal] = elastic_constants_intermediate_dict






# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# plot irreps

# general parameters
fs = 20     # fontsize axis label
ls = 18     # fontsize axis numbers
lsize = 13  # fontsize legend
lw = 3      # linewidth

# temperature bounds
Tmin_Ge = min(T['Mn3Ge'])
Tmax_Ge = max(T['Mn3Ge'])
Tmin_Sn = min(T['Mn3Sn'])
Tmax_Sn = max(T['Mn3Sn'])

# y-axis bouns
comp_min_Ge = -20.5
comp_max_Ge = 3
comp_min_Sn = -4.5
comp_max_Sn = 6.5
shear_min_Ge = -5.5
shear_max_Ge = 1.5
shear_min_Sn = -5.5
shear_max_Sn = 1.5


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:gray']
label_irrep = ['$\mathrm{ (c_{11}+c_{12}/2) \, (A_{1g}) }$ ', '$\mathrm{c_{33} \, (A_{1g})}$', '$\mathrm{c_{13} \, (A_{1g})}$', '$\mathrm{c_{44} \, (E_{1g})}$', '$\mathrm{(c_{11}-c_{12}/2) \, (E_{2g})}$']
irreps = ['A1g1', 'A1g2', 'A1g3', 'E1g', 'E2g']

fig, ax = plt.subplots(2,2, figsize=(13,7), gridspec_kw={'hspace': 0, 'wspace':0.04})

########################################################################################
# Mn3Ge compression
########################################################################################
for i, irrep in enumerate(irreps[:-2]):
    ax[0,0].fill_between(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep]-error['Mn3Ge'][irrep], dc_dict['Mn3Ge'][irrep]+error['Mn3Ge'][irrep], alpha=0.3, facecolor=colors[i])
    ax[0,0].plot(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep], label=label_irrep[i], linewidth=lw, c=colors[i])
    
ax[0,0].legend(loc=(.52, .075), fontsize=lsize)
#ax[0].set_title('Compressional Modes', fontsize=fs)

ax[0,0].tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major', labelbottom=False)

ax[0,0].set_ylabel('$\mathrm{\Delta c}$ (GPa)', fontsize=fs)
# ax[0,0].set_yticks([-20, -15, -10, -5, 0])
# ax[0,0].yaxis.set_label_coords(-0.15, 0.5)

#ax[0,0].set_xlabel('Temperature (K)',fontsize=fs)
ax[0,0].set_xlim([Tmin_Ge, Tmax_Ge])
ax[0,0].set_ylim([comp_min_Ge, comp_max_Ge])


########################################################################################
# Mn3Ge shear
########################################################################################
for i, irrep in enumerate(irreps[-2:]):
    i = i + 3
    ax[1,0].fill_between(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep]-error['Mn3Ge'][irrep], dc_dict['Mn3Ge'][irrep]+error['Mn3Ge'][irrep], alpha=0.3, facecolor=colors[i])
    ax[1,0].plot(T['Mn3Ge'], dc_dict['Mn3Ge'][irrep], label=label_irrep[i], linewidth=lw, c=colors[i])
    
    
ax[1,0].legend(loc=(.52, .075), fontsize=lsize)
#ax[1].set_title('Shear Modes', fontsize=fs)
    
#ax[1,0].yaxis.tick_right()
ax[1,0].tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')

ax[1,0].set_ylabel('$\mathrm{\Delta c}$ (GPa)',fontsize=fs)
# ax[1,0].set_yticks([-5, -2.5, 0, 2.5, 5])
# ax[1,0].yaxis.set_label_coords(-0.15, 0.5)

ax[1,0].set_xlabel('T (K)',fontsize=fs)

ax[1,0].set_xlim([Tmin_Ge, Tmax_Ge])
ax[1,0].set_ylim([shear_min_Ge, shear_max_Ge])


########################################################################################
# Mn3Sn compression
########################################################################################
for i, irrep in enumerate(irreps[:-2]):
    ax[0,1].fill_between(T['Mn3Sn'], dc_dict['Mn3Sn'][irrep]-error['Mn3Sn'][irrep], dc_dict['Mn3Sn'][irrep]+error['Mn3Sn'][irrep], alpha=0.3, facecolor=colors[i])
    ax[0,1].plot(T['Mn3Sn'], dc_dict['Mn3Sn'][irrep], label=label_irrep[i], linewidth=lw, c=colors[i])

ax[0,1].set_xlim([Tmin_Sn, Tmax_Sn])
ax[0,1].set_ylim([comp_min_Sn, comp_max_Sn])
ax[0,1].yaxis.tick_right()
# ax[0,1].set_yticks([-2, 0, 2, 4])
ax[0,1].set_ylabel('$\mathrm{\Delta c}$ (GPa)',fontsize=fs, rotation=270)
ax[0,1].yaxis.set_label_coords(1.1, 0.5)
ax[0,1].tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major', labelbottom=False)



########################################################################################
# Mn3Sn shear
########################################################################################
for i, irrep in enumerate(irreps[-2:]):
    i = i + 3
    ax[1,1].fill_between(T['Mn3Sn'], dc_dict['Mn3Sn'][irrep]-error['Mn3Sn'][irrep], dc_dict['Mn3Sn'][irrep]+error['Mn3Sn'][irrep], alpha=0.3, facecolor=colors[i])
    ax[1,1].plot(T['Mn3Sn'], dc_dict['Mn3Sn'][irrep], label=label_irrep[i], linewidth=lw, c=colors[i])

ax[1,1].set_xlim([Tmin_Sn, Tmax_Sn])
ax[1,1].set_ylim([shear_min_Sn, shear_max_Sn])
ax[1,1].yaxis.tick_right()
# ax[1,1].set_yticks([-1, 0, 1, 2, 3])
ax[1,1].set_ylabel('$\mathrm{\Delta c}$ (GPa)',fontsize=fs, rotation=270)
ax[1,1].set_xlabel('T (K)',fontsize=fs)
ax[1,1].yaxis.set_label_coords(1.1, 0.5)
ax[1,1].tick_params(axis="both",direction="in", labelsize=ls, bottom='True', top='True', left='True', right='True', length=4, width=1, which = 'major')

savepath = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\elastic_constants.pdf'
plt.savefig(savepath, bbox_inches='tight')#, dpi=500)


# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# >->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->-
# plot bulk modulus

# calculate bulk modulus and stuff
Tc = {'Mn3Ge':370, 'Mn3Sn':428.4}
bulk = {}
bulk_error = {}
t = {} # reduced temperature
for sample, c in elastic_constants.items():
    bulk_modulus = ( c['A1g1'] * c['A1g2'] - c['A1g3']**2 ) / ( c['A1g1'] + c['A1g2'] - 2*c['A1g3'] ) / 1e9
    bulk[sample] = bulk_modulus - bulk_modulus[-1]
    t[sample] = (np.array(T[sample]) - Tc[sample]) / Tc[sample]
    d = error[sample]
    dBulk = np.sqrt( ((c['A1g2']-c['A1g3'])**2*d['A1g1'])**2 + ((c['A1g1']-c['A1g3'])**2*d['A1g2'])**2 + (2*(c['A1g1']-c['A1g3'])*(c['A1g2']-c['A1g3'])*d['A1g3'])**2 ) / (c['A1g1']+c['A1g2']-2*c['A1g3'])**2
    bulk_error[sample] = dBulk


# plot parameters
colors = {'Mn3Ge': 'purple', 'Mn3Sn': 'gray'}
tmin = min([min(treduced) for key, treduced in t.items()])
tmax = max([max(treduced) for key, treduced in t.items()])

Bmin_Sn = -2.7
Bmax_Sn = 3
Bmin_Ge = -14.8
Bmax_Ge = 0.15



fig, ax1 = plt.subplots(figsize=(8.5,6))

ax1.fill_between(t['Mn3Ge'], bulk['Mn3Ge']-bulk_error['Mn3Ge'], bulk['Mn3Ge']+bulk_error['Mn3Ge'], alpha=0.3, facecolor=colors['Mn3Ge'])
ax1.plot(t['Mn3Ge'], bulk['Mn3Ge'], color=colors['Mn3Ge'], linewidth=lw)

ax1.set_xlabel('$\\mathrm{(T - T_N)}/\\mathrm{T_N}$', fontsize=fs)
ax1.set_ylabel('$\\Delta$B (GPa) - Mn$_3$Ge', color=colors['Mn3Ge'], fontsize=fs)
ax1.tick_params(axis="x",direction="in", labelsize=ls, bottom='True', top='True', length=4, width=1, which = 'major')
ax1.tick_params(axis="y",direction="in", labelsize=ls, left='True', right='False', length=4, width=1, which = 'major')
ax1.set_xlim(tmin, tmax)
ax1.set_ylim(Bmin_Ge, Bmax_Ge)
    
ax2 = ax1.twinx()
ax2.fill_between(t['Mn3Sn'], bulk['Mn3Sn']-bulk_error['Mn3Sn'], bulk['Mn3Sn']+bulk_error['Mn3Sn'], alpha=0.3, facecolor=colors['Mn3Sn'])
ax2.plot(t['Mn3Sn'], bulk['Mn3Sn'], color=colors['Mn3Sn'], linewidth=lw)

ax2.set_ylabel('$\\Delta$B (GPa) - Mn$_3$Sn', color=colors['Mn3Sn'], fontsize=fs, rotation=270)
ax2.tick_params(axis="y",direction="in", labelsize=ls, left='False', right='True', length=4, width=1, which = 'major')
ax2.set_ylim(Bmin_Sn, Bmax_Sn)
ax2.yaxis.set_label_coords(1.11, 0.5)
    
fig.tight_layout()

savepath = 'C:\\Users\\Florian\\Box Sync\\Projects\\Mn3X\\bulk_modulus.pdf'
plt.savefig(savepath, bbox_inches='tight')#, dpi=500)

plt.show()