import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

# Layout of figure
fig = plt.figure(figsize=(8.5, 5.5), constrained_layout=False)
# Grid layoud of entire figure: 2 rows, 4 columns
gs = fig.add_gridspec(2, 4)
# Grid layout of each column
gsRatImages = gs[0].subgridspec(2, 1)
gsRatTraces = gs[1].subgridspec(3, 2)
gsPigImages = gs[2].subgridspec(2, 1)
gsPigTraces = gs[3].subgridspec(2, 1)

# Axes/plots for rat images and traces
axRatImageVm = fig.add_subplot(gsRatImages[0, 0])
axRatImageCa = fig.add_subplot(gsRatImages[1, 0])
axRatTracePCL1 = fig.add_subplot(gsRatTraces[0, 0])
axRatTracePCL2 = fig.add_subplot(gsRatTraces[1, 0])
axRatTracePCL3 = fig.add_subplot(gsRatTraces[2, 0])

# Axes/plots for pig images and traces
axPigImageVm = fig.add_subplot(gsPigImages[0, 0])
axPigImageCa = fig.add_subplot(gsPigImages[1, 0])
axPigTraceVm = fig.add_subplot(gsPigTraces[0, 0])
axPigTraceCa = fig.add_subplot(gsPigTraces[1, 0])


# Data
# Rat
RatImageVm = np.rot90(plt.imread('data/20180806-rata/Voltage/07-200_Vm_0001.tif'), k=3)
axRatImageVm.axis('off')
axRatImageVm.imshow(RatImageVm, cmap='bone')
axRatImageVm.set_title('Rat, Vm', fontsize=7, fontweight='bold')
RatImageCa = np.rot90(plt.imread('data/20180806-rata/Calcium/07-200_Ca_0001.tif'), k=3)
axRatImageCa.axis('off')
axRatImageCa.imshow(RatImageCa, cmap='bone')
axRatImageCa.set_title('Rat, Ca', fontsize=7, fontweight='bold')

# Pig
PigImageVm = np.rot90(plt.imread('data/20181109-pigb/Voltage/12-300_Vm_0001.tif'))
axPigImageVm.axis('off')
axPigImageVm.imshow(PigImageVm, cmap='bone')
axPigImageVm.set_title('Pig, Vm', fontsize=7, fontweight='bold')
PigTraceVm = np.loadtxt('data/20181109-pigb/Voltage/12-300_Vm_x74y228r10.csv', delimiter=',', usecols=[0], skiprows=0)
PigImageCa = np.rot90(plt.imread('data/20181109-pigb/Calcium/12-300_Ca_0001.tif'))
axPigImageCa.axis('off')
axPigImageCa.imshow(PigImageCa, cmap='bone')
axPigImageCa.set_title('Pig, Ca', fontsize=7, fontweight='bold')
PigTraceCa = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[0], skiprows=0)
PigTraceTime = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[1], skiprows=0)
PigTraces = [PigTraceVm, PigTraceCa]

xlim = [150, 450]
ylim = [0, 1]
for idk, ax in enumerate([axPigTraceVm, axPigTraceCa]):
    ax.tick_params(axis='x', which='both', direction='in')
    ax.tick_params(axis='y', which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Time (ms)', fontsize=8, fontweight='bold')

axPigTraceVm.plot(PigTraceTime, PigTraceVm, color='r', linewidth=1, label='Vm')
axPigTraceCa.plot(PigTraceTime, PigTraceCa, color='y', linewidth=1, label='Vm')


plt.show()
