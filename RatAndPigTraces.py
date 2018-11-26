import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from matplotlib._layoutbox import plot_children
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Layout of figure
fig = plt.figure(figsize=(8.5, 5.5), constrained_layout=True)
# Common plot values
traceTickFontSize = 7
# Grid layoud of entire figure: 2 rows, 4 columns
gs = fig.add_gridspec(1, 4)
# Grid layout of each column
gsRatImages = gs[0].subgridspec(2, 1)
gsRatTraces = gs[1].subgridspec(3, 1)
gsPigImages = gs[2].subgridspec(2, 1)
gsPigTraces = gs[3].subgridspec(2, 1)

# Axes/plots for rat images and traces
axRatImageVm = fig.add_subplot(gsRatImages[0, 0])
axRatImageCa = fig.add_subplot(gsRatImages[1, 0])
axRatTracePCL1 = fig.add_subplot(gsRatTraces[0, 0])
axRatTracePCL1.set_title('PCL: 150 ms', fontsize=7, fontweight='bold')
axRatTracePCL2 = fig.add_subplot(gsRatTraces[1, 0])
axRatTracePCL2.set_title('PCL: 200 ms', fontsize=7, fontweight='bold')
axRatTracePCL3 = fig.add_subplot(gsRatTraces[2, 0])
axRatTracePCL3.set_title('PCL: 250 ms', fontsize=7, fontweight='bold')

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
axRatImageVm.set_title('Rat, Vm', fontsize=7, fontweight='bold')
RatImageCa = np.rot90(plt.imread('data/20180806-rata/Calcium/07-200_Ca_0001.tif'), k=3)
axRatImageCa.axis('off')
axRatImageCa.imshow(RatImageCa, cmap='bone')
axRatImageCa.set_title('Rat, Ca', fontsize=7, fontweight='bold')
# Region of Interest circles, adjusted for rotation
# roiXY = (RatImageVm.shape[0] - 349, RatImageVm.shape[1] - 114)
RatROI_XY = (114, 349)
RatROI_R = 10
rRatROI_CircleVm = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
rRatROI_CircleCa = Circle(RatROI_XY, RatROI_R, edgecolor='w', fc='none', lw=1)
axRatImageVm.add_patch(rRatROI_CircleVm)
axRatImageCa.add_patch(rRatROI_CircleCa)
# Rat Trace
RatTracePCL1Vm = np.loadtxt('data/20180806-rata/Voltage/13-150_Vm_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)
RatTracePCL1Ca = np.loadtxt('data/20180806-rata/Calcium/13-150_Ca_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)

RatTracePCL2Vm = np.loadtxt('data/20180806-rata/Voltage/07-200_Vm_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)
RatTracePCL2Ca = np.loadtxt('data/20180806-rata/Calcium/07-200_Ca_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)

RatTracePCL3Vm = np.loadtxt('data/20180806-rata/Voltage/02-250_Vm_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)
RatTracePCL3Ca = np.loadtxt('data/20180806-rata/Calcium/02-250_Ca_x349y114r10.csv', delimiter=',', usecols=[0], skiprows=0)

RatTraceTime = np.loadtxt('data/20180806-rata/Calcium/13-150_Ca_x349y114r10.csv', delimiter=',', usecols=[1], skiprows=0)
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
    plt.xlabel('Time (ms)', fontsize=8, fontweight='bold')

# RatTraces
# axRatTracePCL1.set_ylabel('Normalized\nFluorescence', fontsize=7, fontweight='bold')
axRatTracePCL2.set_ylabel('Normalized Fluorescence @ ROI', fontsize=7, fontweight='bold')
# axRatTracePCL3.set_ylabel('Normalized\nFluorescence', fontsize=7, fontweight='bold')

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
axPigImageVm.set_title('Pig, Vm', fontsize=7, fontweight='bold')
axPigImageCa.set_title('Pig, Ca', fontsize=7, fontweight='bold')
PigImageCa = np.rot90(plt.imread('data/20181109-pigb/Calcium/12-300_Ca_0001.tif'))

PigImageScale = [229, 229]  # pixels/cm
ScaleFont = fm.FontProperties(size=8, family='monospace')
PigImageScaleBarVm = AnchoredSizeBar(axPigImageVm.transData, PigImageScale[0], '1 cm', 'lower right',
                                     pad=0.5, color='w', frameon=False, fontproperties=ScaleFont)
PigImageScaleBarCa = AnchoredSizeBar(axPigImageCa.transData, PigImageScale[1], '1 cm', 'lower right',
                                     pad=0.5, color='w', frameon=False, fontproperties=ScaleFont)
PigImageScaleBars = [PigImageScaleBarVm, PigImageScaleBarCa]
for idx, ax in enumerate([axPigImageVm, axPigImageCa]):
    ax.axis('off')
    ax.imshow(PigImageCa, cmap='bone')
    ax.add_artist(PigImageScaleBars[idx])
# Region of Interest circles, adjusted for rotation
# roiXY = (RatImageVm.shape[0] - 349, RatImageVm.shape[1] - 114)
PigrROI_XY = (PigImageVm.shape[0] - 228, PigImageVm.shape[1] - 74)
PigrROI_R = 10
PigrROI_CircleVm = Circle(PigrROI_XY, PigrROI_R, edgecolor='w', fc='none', lw=1)
PigrROI_CircleCa = Circle(PigrROI_XY, PigrROI_R, edgecolor='w', fc='none', lw=1)
axPigImageVm.add_patch(PigrROI_CircleVm)
axPigImageCa.add_patch(PigrROI_CircleCa)

# Pig Traces
PigTraceVm = np.loadtxt('data/20181109-pigb/Voltage/12-300_Vm_x74y228r10.csv', delimiter=',', usecols=[0], skiprows=0)
PigTraceCa = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[0], skiprows=0)
PigTraceTime = np.loadtxt('data/20181109-pigb/Calcium/12-300_Ca_x74y228r10.csv', delimiter=',', usecols=[1], skiprows=0)
PigTraces = [PigTraceVm, PigTraceCa]

xLimitPig = [0, 0.5]
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
    plt.xlabel('Time (ms)', fontsize=8, fontweight='bold')

axPigTraceVm.set_ylabel('Normalized\nFluorescence, Vm @ ROI', fontsize=7, fontweight='bold')
axPigTraceCa.set_ylabel('Normalized\nFluorescence, Ca @ ROI', fontsize=7, fontweight='bold')

axPigTraceVm.plot(PigTraceTime, PigTraceVm, color='r', linewidth=lineWidthPig, label='Vm')
axPigTraceCa.plot(PigTraceTime, 1-PigTraceCa, color='y', linewidth=lineWidthPig, label='Vm')
axPigTraceVm.set_xlim(xLimitPig)
axPigTraceCa.set_xlim(xLimitPig)


# plot_children(fig, fig._layoutbox, printit=False)
plt.show()

