# Import and plot the spectra of lights, fluorophores, and filters used in Langendorff experiments
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

# colors_dyes = ['#5EFF00', '#FFEF00', '#000000', '#ED0000', '#AA9999']  # Excitation, Rhod2, Dicrhoic, RH237, di4anbdqpq
colors_dyes = {'Excitation Filter': '#5EFF00',
               'Dischroic': '#000000',
               'Rhod2': '#FFEF00',
               'RH237': '#ED0000',
               'di4anbdqpq': '#AA9999'}
# colors_dyes_iter = iter(colors_dyes)
lines_dyes = ['--', '-.']  # Excitation: dash, Emission: dashdot
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]

# Setup the figure
fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
ax = fig.add_subplot(111)
plt.rc('xtick', labelsize=fontsize2)
plt.rc('ytick', labelsize=fontsize2)

# # Import data from text files
# Excitation light spectra
# TODO import Solis spectra

# Lights
solis660C = np.loadtxt('Data/solis-660c-spectra.txt')
solis525C = np.loadtxt('Data/solis-525c-spectra.txt')
# Filters
et530_40 = np.loadtxt('Data/et530-40x-lot321683-transmission.txt')  # SM1 sized for Mightex lights?
ct510_60 = np.loadtxt('Data/ct510-60bp ll298805.txt')   # SM2 sized for Solis lights
dichroic_em = np.loadtxt('Data/T660lpxrxt.txt')
ET585_40m = np.loadtxt('Data/ET585-40m.txt')
ET710_LP = np.loadtxt('Data/et710lp-lot296950-transmission.txt')
HP725_LP = np.loadtxt('Data/hp725lp-edmund-transmission.txt')
# with open('Data/ct510-60bp ll298805.csv') as f:
#     lines = (line for line in f if not line.startswith('#'))
#     ct510_60_ex = np.genfromtxt(lines, delimiter=',', skip_header=True)

# Dye excitation and emission data
# Rhod-2
rhod2_flx = np.loadtxt('Data/Rhod2Ex.txt', delimiter=',')  # From ThermoFisher SpectraViewer
rhod2_flm = np.loadtxt('Data/Rhod2Em.txt', delimiter=',')  # From ThermoFisher SpectraViewer
# RH 237
rh237_flm = np.loadtxt('Data/rh237_fl.dat')  # From Choi and Salama 2000 - g3data
# Di-4-ANBDQPQ (JPW-6003)
di4anbdqpq_flx = np.loadtxt('Data/Di-4-ANBDQPQ_AbsMLV.txt')
di4anbdqpq_flm = np.loadtxt('Data/Di-4-ANBDQPQ_EmMLV.txt')

# # Plot data
# Excitation light
plot_ex, = ax.plot(solis525C[:, 0], solis525C[:, 1] * 100, linewidth=2, color=colors_dyes['Excitation Filter'],
                   label='Solis-525C')  # From Thorlabs
# plot_ex, = ax.plot(solis660C[:, 0], solis660C[:, 1] * 100, linewidth=2, color='r',
#                           label='Solis-660C')  # From Thorlabs
# Excitation light filter
# plot_filter_ex530, = ax.plot(et530_40[:600, 0], et530_40[:600, 1], linewidth=2, color=colors_dyes['Excitation Filter'],
#                           label='ET530/40')  # From Brian Manning email 2017-10-13
plot_filter_ex510, = ax.plot(ct510_60[:, 0], ct510_60[:, 1], linewidth=2, color=colors_dyes['Excitation Filter'],
                       label='CT510/60')  # From Michael Stanley email 2018-12-05

# # Dye excitation and emission spectra and emission filters
# Rhod-2
# rhod2_ex, = plt.plot(rhod2_flx[:, 0], rhod2_flx[:, 1],
#                      linewidth=2, color='gray', label='RH237_x')  # Multiply by 100
rhod2_em, = ax.plot(rhod2_flm[:, 0], rhod2_flm[:, 1],
                    linewidth=2, color=colors_dyes['Rhod2'], linestyle=lines_dyes[1], label='Rhod-2_em')
# ET585_40m, for Rhod-2 emission
ax.plot(ET585_40m[:700 - 300, 0], ET585_40m[:700 - 300, 1] * 100, linewidth=2, color=colors_dyes['Rhod2'],
        label='ET585/40')  # Multiply by 100, from Chroma site

# Dichroic
ax.plot(dichroic_em[230:, 0], dichroic_em[230:, 1] * 100, linewidth=2, color=colors_dyes['Dischroic'],
        label='Dichroic Mir.')  # Multiply by 100, from Chroma site

# RH 237
# rh237_ex, = plt.plot(rh237_flx[165:, 0], rh237_flx[165:, 1] * 100,
#                      linewidth=2, color='gray', linestyle=lines_dyes[0], label='RH237_x')  # Multiply by 100
rh237_em, = ax.plot(rh237_flm[160:, 0], rh237_flm[160:, 1] * 100,
                    linewidth=2, color=colors_dyes['RH237'], linestyle=lines_dyes[1], label='RH237_em')  # Multiply by 100
# ET710_LP, for RH237 emission
ax.plot(ET710_LP[325:, 0], ET710_LP[325:, 1], linewidth=2, color=colors_dyes['RH237'],
        label='ET710lp')  # From Brian Manning email 2017-10-13

# # Di-4-ANBDQPQ (JPW-6003)
# di4anbdqpq_ex, = ax.plot(di4anbdqpq_flx[:, 0], di4anbdqpq_flx[:, 1] * 100,
#                          linewidth=1, color=colors_dyes['di4anbdqpq'], linestyle=lines_dyes[0],
#                          label='di4anbdqpq_ex')  # Multiply by 100
# di4anbdqpq_em, = ax.plot(di4anbdqpq_flm[:, 0], di4anbdqpq_flm[:, 1] * 100,
#                          linewidth=2, color=colors_dyes['di4anbdqpq'], linestyle=lines_dyes[1],
#                          label='di4anbdqpq_em')  # Multiply by 100
# # 725 Long Pass, from Edmund Optics, for Di-4-ANBDQPQ emission
# ax.plot(HP725_LP[325:, 0], HP725_LP[325:, 1], linewidth=2, color=colors_dyes['di4anbdqpq'],
#         label='ET710lp')  # Shifted copy of ET710_LP


# Print legend using each 'label' from above
ax.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1,
          frameon=False)

ax.set_xlim([450, 800])     # For Dual Solis
# ax.set_xlim([550, 900])     # For Di-4-ANBDQPQ
ax.set_ylim([0, 100])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
ax.tick_params(axis='x', which='major', length=8, bottom=True, top=True)
# x_major_loc = plticker.MultipleLocator(base=100)     # this locator puts ticks at regular intervals

ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(10))
ax.tick_params(axis='y', which='both', right=False, left=True)

# plt.text(518,28,'Excitation',ha='left',va='bottom',fontsize=16, rotation=90)
text_y = 18
# ax.text(523, text_y, 'Excitation (Rhod-2 + RH237)', ha='left', va='bottom', fontsize=fontsize2, rotation=90)
# ax.text(583, text_y, 'Rhod-2', ha='left', va='bottom', fontsize=fontsize2, rotation=90)
# ax.text(665, text_y, 'Em. Dichroic', ha='left', va='bottom', fontsize=fontsize2, rotation=90)
# ax.text(713, text_y, 'RH237', ha='left', va='bottom', fontsize=fontsize2, rotation=90)
# ax.text(738, text_y, 'Di-4-ANBDQPQ', ha='left', va='bottom', fontsize=fontsize2, rotation=90)

ax.set_ylabel('Filter Transmission (%)', fontsize=fontsize1)
ax.set_xlabel('Wavelength (nm)', fontsize=fontsize1)

# Show and save the figure
fig.show()
# fig.savefig('LangendorffFilters_DualSolis.png')
# fig.savefig('LangendorffFilters_Di-4-ANBDQPQ.png')
