import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from matplotlib._layoutbox import plot_children
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Layout of figure
fig = plt.figure(figsize=(8.5, 5.5))
# Common plot values
traceTickFontSize = 7
titleFont = fm.FontProperties(size=8, weight='bold')
scaleFont = fm.FontProperties(size=7, family='monospace')
# Grid layout of entire figure: 1 row, 4 columns
gs = fig.add_gridspec(1, 4)
# Grid layout of each column
gsRatImages = gs[0].subgridspec(2, 1)
gsRatTraces = gs[1].subgridspec(3, 1, hspace=0.3)
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
# Rat Images
RatImageVm = np.rot90(plt.imread('data/20180806-rata/Voltage/07-200_Vm_0001.tif'), k=3)
axRatImageVm.axis('off')
axRatImageVm.imshow(RatImageVm, cmap='bone')
axRatImageVm.set_title('Rat, Vm', fontproperties=titleFont)
RatImageCa = np.rot90(plt.imread('data/20180806-rata/Calcium/07-200_Ca_0001.tif'), k=3)
axRatImageCa.axis('off')
axRatImageCa.imshow(RatImageCa, cmap='bone')
axRatImageCa.set_title('Rat, Ca', fontproperties=titleFont)
# Scale Bars
RatImageScale = [196, 196]  # pixels/cm
RatImageScaleBarVm = AnchoredSizeBar(axRatImageVm.transData, RatImageScale[0], '1 cm', 'upper right',
                                     pad=0.5, color='w', frameon=False, fontproperties=scaleFont)
RatImageScaleBarCa = AnchoredSizeBar(axRatImageCa.transData, RatImageScale[1], '1 cm', 'upper right',
                                     pad=0.5, color='w', frameon=False, fontproperties=scaleFont)
RatImageScaleBars = [RatImageScaleBarVm, RatImageScaleBarCa]
for idx, ax in enumerate([axRatImageVm, axRatImageCa]):
    ax.add_artist(RatImageScaleBars[idx])
# Region of Interest circles, adjusted for rotation
# roiXY = (RatImageVm.shape[0] - 349, RatImageVm.shape[1] - 114)
RatROI_XY = (RatImageVm.shape[1] - 114, 349)
RatROI_R = 10
rRatROI_CircleVm = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
rRatROI_CircleCa = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
axRatImageVm.add_patch(rRatROI_CircleVm)
axRatImageCa.add_patch(rRatROI_CircleCa)
# Rat Traces
axRatTracePCL1.set_title('PCL: 150 ms', fontproperties=titleFont)
axRatTracePCL3.set_title('PCL: 250 ms', fontproperties=titleFont)
axRatTracePCL2.set_title('PCL: 200 ms', fontproperties=titleFont)
RatTracePCL1Vm = np.loadtxt('data/20180806-rata/Voltage/13-150_Vm_x349y114r10.csv', delimiter=',', usecols=[0])
RatTracePCL1Ca = np.loadtxt('data/20180806-rata/Calcium/13-150_Ca_x349y114r10.csv', delimiter=',', usecols=[0])
RatTracePCL2Vm = np.loadtxt('data/20180806-rata/Voltage/07-200_Vm_x349y114r10.csv', delimiter=',', usecols=[0])
RatTracePCL2Ca = np.loadtxt('data/20180806-rata/Calcium/07-200_Ca_x349y114r10.csv', delimiter=',', usecols=[0])
RatTracePCL3Vm = np.loadtxt('data/20180806-rata/Voltage/02-250_Vm_x349y114r10.csv', delimiter=',', usecols=[0])
RatTracePCL3Ca = np.loadtxt('data/20180806-rata/Calcium/02-250_Ca_x349y114r10.csv', delimiter=',', usecols=[0])
RatTraceTime = np.loadtxt('data/20180806-rata/Calcium/13-150_Ca_x349y114r10.csv', delimiter=',', usecols=[1])
xLimitRat = [0, 0.8]
yLimitRat = [0, 1]
lineWidthRat = 0.2
for idk, ax in enumerate([axRatTracePCL1, axRatTracePCL2, axRatTracePCL3]):
    ax.tick_params(axis='x', labelsize=traceTickFontSize, which='both', direction='in')
    ax.tick_params(axis='y', labelsize=traceTickFontSize, which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.set_xlim(xLimitRat)
    ax.set_ylim(yLimitRat)

axRatTracePCL3.set_xlabel('Time (ms)', fontsize=7)
axRatTracePCL2.set_ylabel('Normalized Vm & Ca\nFluorescence @ ROI', fontproperties=titleFont)

axRatTracePCL1.plot(RatTraceTime, RatTracePCL1Vm[0:len(RatTraceTime)],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL1.plot(RatTraceTime, 1-RatTracePCL1Ca[0:len(RatTraceTime)],
                    color='y', linewidth=lineWidthRat, label='Vm')
axRatTracePCL2.plot(RatTraceTime, RatTracePCL2Vm[0:len(RatTraceTime)],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL2.plot(RatTraceTime, RatTracePCL2Ca[0:len(RatTraceTime)],
                    color='y', linewidth=lineWidthRat, label='Vm')
axRatTracePCL3.plot(RatTraceTime, RatTracePCL3Vm[0:len(RatTraceTime)],
                    color='r', linewidth=lineWidthRat, label='Vm')
axRatTracePCL3.plot(RatTraceTime, RatTracePCL3Ca[0:len(RatTraceTime)],
                    color='y', linewidth=lineWidthRat, label='Vm')

# Pig Images
PigImageVm = np.rot90(plt.imread('data/20181109-pigb/Voltage/12-300_Vm_0001.tif'))
axPigImageVm.axis('off')
axPigImageVm.imshow(PigImageVm, cmap='bone')
axPigImageVm.set_title('Pig, Vm', fontproperties=titleFont)
PigImageCa = np.rot90(plt.imread('data/20181109-pigb/Calcium/12-300_Ca_0001.tif'))
axPigImageCa.axis('off')
axPigImageCa.imshow(PigImageCa, cmap='bone')
axPigImageCa.set_title('Pig, Ca', fontproperties=titleFont)
# Scale Bars
PigImageScale = [109, 109]  # pixels/cm
PigImageScaleBarVm = AnchoredSizeBar(axPigImageVm.transData, PigImageScale[0], '1 cm', 'upper right',
                                     pad=0.5, color='w', frameon=False, fontproperties=scaleFont)
PigImageScaleBarCa = AnchoredSizeBar(axPigImageCa.transData, PigImageScale[1], '1 cm', 'upper right',
                                     pad=0.5, color='w', frameon=False, fontproperties=scaleFont)
PigImageScaleBars = [PigImageScaleBarVm, PigImageScaleBarCa]
for idx, ax in enumerate([axPigImageVm, axPigImageCa]):
    ax.add_artist(PigImageScaleBars[idx])
# Region of Interest circles, adjusted for rotation
PigrROI_XY = (228, PigImageVm.shape[0] - 74)
PigrROI_R = 10
PigrROI_CircleVm = Circle(PigrROI_XY, PigrROI_R, edgecolor='w', fc='none', lw=1)
PigrROI_CircleCa = Circle(PigrROI_XY, PigrROI_R, edgecolor='w', fc='none', lw=1)
axPigImageVm.add_patch(PigrROI_CircleVm)
axPigImageCa.add_patch(PigrROI_CircleCa)

# Pig Traces
PigTraceVm = np.loadtxt('data/20181109-pigb/Voltage/12-300_Vm_x74y228r10.csv', delimiter=',', usecols=[0])
PigTraceCa = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[0])
PigTraceTime = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[1])
PigTraces = [PigTraceVm, PigTraceCa]

xLimitPig = [0, 0.8]
yLimitPig = [0, 1]
lineWidthPig = 0.4
for idx, ax in enumerate([axPigTraceVm, axPigTraceCa]):
    ax.tick_params(axis='x', labelsize=traceTickFontSize+2,  which='both', direction='in')
    ax.tick_params(axis='y', labelsize=traceTickFontSize+2,  which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    plt.xlabel('Time (ms)', fontsize=7)

axPigTraceVm.set_ylabel('Normalized Vm\nFluorescence @ ROI', fontproperties=titleFont)
axPigTraceCa.set_ylabel('Normalized Ca\nFluorescence @ ROI', fontproperties=titleFont)

axPigTraceVm.plot(PigTraceTime, PigTraceVm, color='r', linewidth=lineWidthPig, label='Vm')
axPigTraceCa.plot(PigTraceTime, 1-PigTraceCa, color='y', linewidth=lineWidthPig, label='Vm')
axPigTraceVm.set_xlim(xLimitPig)
axPigTraceCa.set_xlim(xLimitPig)

# Legend for all traces
legend_lines = [Line2D([0], [0], color='r', lw=1),
                Line2D([0], [0], color='y', lw=1)]
axPigTraceVm.legend(legend_lines, ['Vm', 'Ca'],
                    loc='upper right', ncol=1, prop={'size': 7}, numpoints=1, frameon=True)

# plot_children(fig, fig._layoutbox, printit=False)
plt.show()
