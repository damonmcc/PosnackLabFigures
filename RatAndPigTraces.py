import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from matplotlib._layoutbox import plot_children
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Layout of figure
fig = plt.figure()
fig.set_size_inches(8.5, 5.5, forward=True)
# Common plot values
traceTickFontSize = 7
titleFont = fm.FontProperties(size=8, weight='bold')
scaleFont = fm.FontProperties(size=7, family='monospace')
# Grid layout of entire figure: 1 row, 4 columns
gs = fig.add_gridspec(1, 4, wspace=0.45)
# Grid layout of each column
gsRatImages = gs[0].subgridspec(2, 1)
gsRatTraces = gs[1].subgridspec(3, 1, hspace=0.3)
gsPigImages = gs[2].subgridspec(2, 1)
gsPigTraces = gs[3].subgridspec(3, 1, hspace=0.3)
# Axes/plots for rat images and traces
axRatImageVm = fig.add_subplot(gsRatImages[0, 0])
axRatImageCa = fig.add_subplot(gsRatImages[1, 0])
axRatTracePCL1 = fig.add_subplot(gsRatTraces[0, 0])
axRatTracePCL2 = fig.add_subplot(gsRatTraces[1, 0])
axRatTracePCL3 = fig.add_subplot(gsRatTraces[2, 0])
# Axes/plots for pig images and traces
axPigImageVm = fig.add_subplot(gsPigImages[0, 0])
axPigImageCa = fig.add_subplot(gsPigImages[1, 0])
axPigTracePCL1 = fig.add_subplot(gsPigTraces[0, 0])
axPigTracePCL2 = fig.add_subplot(gsPigTraces[1, 0])
axPigTracePCL3 = fig.add_subplot(gsPigTraces[2, 0])

# Rat data loading and plotting
# Rat Images
axRatImageVm.set_title('Rat, Vm', fontproperties=titleFont)
RatImageVm = np.rot90(plt.imread('data/20180806-rata/Voltage/07-200_Vm_0001.tif'), k=3)
RatNorm = colors.Normalize(vmin=RatImageVm.min(), vmax=RatImageVm.max()/2.5)
axRatImageVm.imshow(RatImageVm, cmap='bone', norm=RatNorm)
axRatImageVm.axis('off')
axRatImageCa.set_title('Rat, Ca', fontproperties=titleFont)
RatImageCa = np.rot90(plt.imread('data/20180806-rata/Calcium/07-200_Ca_0001.tif'), k=3)
RatNorm = colors.Normalize(vmin=RatImageCa.min()/0.8, vmax=RatImageCa.max()/1.7)
axRatImageCa.imshow(RatImageCa, cmap='bone', norm=RatNorm)
axRatImageCa.axis('off')
# Scale Bars
RatImageScale = [196, 196]  # pixels/cm
RatImageScaleBarVm = AnchoredSizeBar(axRatImageVm.transData, RatImageScale[0], ' ', 'upper right',
                                     pad=0.2, color='w', frameon=False, fontproperties=scaleFont)
RatImageScaleBarCa = AnchoredSizeBar(axRatImageCa.transData, RatImageScale[1], ' ', 'upper right',
                                     pad=0.2, color='w', frameon=False, fontproperties=scaleFont)
RatImageScaleBars = [RatImageScaleBarVm, RatImageScaleBarCa]
for idx, ax in enumerate([axRatImageVm, axRatImageCa]):
    ax.add_artist(RatImageScaleBars[idx])
# Region of Interest circles, adjusted for rotation(s)
RatROI_XY = (RatImageVm.shape[1] - 56, 217)
RatROI_R = 45
RatROI_CircleVm = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
RatROI_CircleCa = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
axRatImageVm.add_patch(RatROI_CircleVm)
axRatImageCa.add_patch(RatROI_CircleCa)
# Rat Traces
axRatTracePCL1.set_title('PCL: 150 ms', fontproperties=titleFont)
axRatTracePCL2.set_title('PCL: 200 ms', fontproperties=titleFont)
axRatTracePCL3.set_title('PCL: 220 ms', fontproperties=titleFont)
# Load data from .csv files
# Row format:
# SIGNAL VALUE, TIME(ms)
RatTracePCL1Vm = np.genfromtxt('data/20180806-rata/Voltage/30-150_Vm_x217y56r45.csv', delimiter=',')
RatTracePCL1Ca = np.genfromtxt('data/20180806-rata/Calcium/30-150_Ca_x217y56r45.csv', delimiter=',')
RatTracePCL2Vm = np.genfromtxt('data/20180806-rata/Voltage/24-200_Vm_x217y56r45.csv', delimiter=',')
RatTracePCL2Ca = np.genfromtxt('data/20180806-rata/Calcium/24-200_Ca_x217y56r45.csv', delimiter=',')
RatTracePCL3Vm = np.genfromtxt('data/20180806-rata/Voltage/22-220_Vm_x217y56r45.csv', delimiter=',')
RatTracePCL3Ca = np.genfromtxt('data/20180806-rata/Calcium/22-220_Ca_x217y56r45.csv', delimiter=',')
xLimitRat = [0, 0.8]
yLimitRat = [0, 1]
lineWidthRat = 0.9
# Format figures
for idk, ax in enumerate([axRatTracePCL1, axRatTracePCL2, axRatTracePCL3]):
    ax.tick_params(axis='x', labelsize=traceTickFontSize, which='both', direction='in')
    ax.tick_params(axis='y', labelsize=traceTickFontSize, which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(xLimitRat)
    ax.set_ylim(yLimitRat)
axRatTracePCL3.set_xlabel('Time (ms)', fontsize=7)
axRatTracePCL2.set_ylabel('Normalized Vm & Ca\nFluorescence @ ROI', fontproperties=titleFont)
# Plot Traces

# axRatTracePCL1.set_xlim([0, 0.6])
axRatTracePCL1.plot(RatTracePCL1Vm[:, 1], RatTracePCL1Vm[:, 0],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL1.plot(RatTracePCL1Ca[:, 1], RatTracePCL1Ca[:, 0],
                    color='y', linewidth=lineWidthRat, label='Ca')
# axRatTracePCL2.set_xlim([0.2, 0.8])
axRatTracePCL2.plot(RatTracePCL2Vm[:, 1], RatTracePCL2Vm[:, 0],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL2.plot(RatTracePCL2Ca[:, 1], RatTracePCL2Ca[:, 0],
                    color='y', linewidth=lineWidthRat, label='Ca')
# axRatTracePCL3.set_xlim([0.2, 0.6])
axRatTracePCL3.plot(RatTracePCL3Vm[:, 1], RatTracePCL3Vm[:, 0],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL3.plot(RatTracePCL3Ca[:, 1], RatTracePCL3Ca[:, 0],
                    color='y', linewidth=lineWidthRat, label='Ca')

# Pig data loading and plotting
# Pig Images
axPigImageVm.set_title('Pig, Vm', fontproperties=titleFont)
PigImageVm = np.rot90(plt.imread('data/20181109-pigb/Voltage/12-300_Vm_0001.tif'))
PigNorm = colors.Normalize(vmin=PigImageVm.min(), vmax=PigImageVm.max()/1.1)
axPigImageVm.imshow(PigImageVm, cmap='bone', norm=PigNorm)
axPigImageVm.axis('off')
axPigImageCa.set_title('Pig, Ca', fontproperties=titleFont)
PigImageCa = np.rot90(plt.imread('data/20181109-pigb/Calcium/12-300_Ca_0001.tif'))
PigNorm = colors.Normalize(vmin=PigImageCa.min(), vmax=PigImageCa.max()/1)
axPigImageCa.imshow(PigImageCa, cmap='bone', norm=PigNorm)
axPigImageCa.axis('off')
# Scale Bars
PigImageScale = [109, 109]  # pixels/cm
PigImageScaleBarVm = AnchoredSizeBar(axPigImageVm.transData, PigImageScale[0], ' ', 'upper right',
                                     pad=0.2, color='w', frameon=False, fontproperties=scaleFont)
PigImageScaleBarCa = AnchoredSizeBar(axPigImageCa.transData, PigImageScale[1], ' ', 'upper right',
                                     pad=0.2, color='w', frameon=False, fontproperties=scaleFont)
PigImageScaleBars = [PigImageScaleBarVm, PigImageScaleBarCa]
for idx, ax in enumerate([axPigImageVm, axPigImageCa]):
    ax.add_artist(PigImageScaleBars[idx])
# Region of Interest circles, adjusted for rotation(s)
PigROI_XY = (150, PigImageVm.shape[0] - 253)
PigROI_R = 80
PigROI_CircleVm = Circle(PigROI_XY, PigROI_R, edgecolor='w', fc='none', lw=1)
PigROI_CircleCa = Circle(PigROI_XY, PigROI_R, edgecolor='w', fc='none', lw=1)
axPigImageVm.add_patch(PigROI_CircleVm)
axPigImageCa.add_patch(PigROI_CircleCa)
# Pig Traces
axPigTracePCL1.set_title('PCL: 180 ms', fontproperties=titleFont)
axPigTracePCL2.set_title('PCL: 200 ms', fontproperties=titleFont)
axPigTracePCL3.set_title('PCL: 220 ms', fontproperties=titleFont)
# Load data from .csv files
# Row format:
# SIGNAL VALUE, TIME(ms)
PigTracePCL1Vm = np.genfromtxt('data/20181109-pigb/Voltage/20-180_Vm_x253y150r80.csv', delimiter=',')
PigTracePCL1Ca = np.genfromtxt('data/20181109-pigb/Calcium/20-180_Ca_x253y150r80.csv', delimiter=',')
PigTracePCL2Vm = np.genfromtxt('data/20181109-pigb/Voltage/18-200_Vm_x253y150r80.csv', delimiter=',')
PigTracePCL2Ca = np.genfromtxt('data/20181109-pigb/Calcium/18-200_Ca_x253y150r80.csv', delimiter=',')
PigTracePCL3Vm = np.genfromtxt('data/20181109-pigb/Voltage/16-220_Vm_x253y150r80.csv', delimiter=',')
PigTracePCL3Ca = np.genfromtxt('data/20181109-pigb/Calcium/16-220_Ca_x253y150r80.csv', delimiter=',')
# PigTraceTime = np.loadtxt('data/20181109-pigb/Calcium/20-180_Ca_x253y150r80.csv', delimiter=',')
xLimitPig = [0.2, 0.8]
yLimitPig = [0, 1]
lineWidthPig = 0.9
# Format figures
for idk, ax in enumerate([axPigTracePCL1, axPigTracePCL2, axPigTracePCL3]):
    ax.tick_params(axis='x', labelsize=traceTickFontSize, which='both', direction='in')
    ax.tick_params(axis='y', labelsize=traceTickFontSize, which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_xlim(xLimitPig)
    ax.set_ylim(yLimitPig)
    plt.xlabel('Time (ms)', fontsize=7)
axPigTracePCL2.set_ylabel('Normalized Vm & Ca\nFluorescence @ ROI', fontproperties=titleFont)
# Plot traces
axPigTracePCL1.plot(PigTracePCL1Vm[:, 1], PigTracePCL1Vm[:, 0],
                    color='r', linewidth=lineWidthPig, label='Vm')
axPigTracePCL1.plot(PigTracePCL1Ca[:, 1], PigTracePCL1Ca[:, 0],
                    color='y', linewidth=lineWidthPig, label='Ca')
axPigTracePCL2.plot(PigTracePCL2Vm[:, 1], PigTracePCL2Vm[:, 0],
                    color='r', linewidth=lineWidthPig, label='Vm')
axPigTracePCL2.plot(PigTracePCL2Ca[:, 1], PigTracePCL2Ca[:, 0],
                    color='y', linewidth=lineWidthPig, label='Ca')
axPigTracePCL3.plot(PigTracePCL3Vm[:, 1], PigTracePCL3Vm[:, 0],
                    color='r', linewidth=lineWidthPig, label='Vm')
axPigTracePCL3.plot(PigTracePCL3Ca[:, 1], PigTracePCL3Ca[:, 0],
                    color='y', linewidth=lineWidthPig, label='Ca')


# Legend for all traces
legend_lines = [Line2D([0], [0], color='r', lw=1),
                Line2D([0], [0], color='y', lw=1)]
axPigTracePCL1.legend(legend_lines, ['Vm', 'Ca'],
                      loc='upper right', bbox_to_anchor=(1.2, 1.1),
                      ncol=1, prop={'size': 6}, labelspacing=1, numpoints=1, frameon=False)

# plot_children(fig, fig._layoutbox, printit=False) # requires "constrained_layout=True"
plt.show()
fig.savefig('RatAndPigTraces.svg', format='svg', dpi=fig.dpi)
