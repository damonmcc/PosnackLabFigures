import numpy as np
from decimal import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.patches import Ellipse, Circle, Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import colorsys
import ScientificColourMaps5 as scm
import warnings

MAX_COUNTS_16BIT = 65536
roi_colors = ['b', 'r', 'k']
signal_colors = ['#bf8637', '#bfbf37']  # Vm: orange, Ca: yellow


def plot_heart(axis, heart_image, rois=None):
    """
    Display an image of a heart on a given axis.

    Parameters
    ----------
    axis : array-like or PIL image
        The axis to plot the heart onto

    heart_image : array-like or PIL image
        The image data. See matplotlib.image.AxesImage.imshow for supported array shapes

        The first two dimensions (M, N) define the rows and columns of
        the image.

    Returns
    -------
    image : `~matplotlib.image.AxesImage`
    """
    # Setup plot
    height, width, = heart_image.shape[0], heart_image.shape[1]  # X, Y flipped due to rotation
    axis.axis('off')
    img = axis.imshow(heart_image, cmap='bone')

    if rois:
        # Create ROIs and get colors of their pixels
        for idx, roi in enumerate(rois):
            roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc=None,
                                ec=roi_colors[idx], lw=1)
            axis.add_artist(roi_circle)

    # patch = Ellipse((width/2, height/2), width=width, height=height, transform=axis.transData)
    # img.set_clip_path(patch)
    # Scale Bar
    scale_px_cm = 1 / 0.0149
    heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
    heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '1 cm',
                                      'lower right', pad=0.2, color='w', frameon=False,
                                      fontproperties=fm.FontProperties(size=7, weight='bold'))
    axis.add_artist(heart_scale_bar)

    return img


def plot_trace(axis, data, imagej=False, fps=None, x_span=0, x_end=None,
               norm=False, invert=False, color='b',):
    if imagej:
        if not x_end:
            x_end = len(data)
        x_start = x_end - x_span

        data_x = data[x_start:x_end, 0] - x_start             # All rows of the first column (skip X,Y header row)
        if fps:
            # convert to ms
            data_x = ((data_x - 1) / fps) * 1000    # ensure range is 0 - max
            data_x = data_x.astype(int)
            # axis.xaxis.set_major_locator(ticker.IndexLocator(base=1000, offset=500))
        # axis.set_xlim(xlim)
        # axis.xaxis.set_major_locator(ticker.AutoLocator())
        # axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axis.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(int(1000/4)))
        axis.tick_params(labelsize=6)

        data_y_counts = data[x_start:x_end, 1].astype(int)     # rows of the first column (skip X,Y header row)
        # data_y_counts_delta = data_y.max() - data_y_.min()
        # MAX_COUNTS_16BIT
        data_y = data_y_counts

        if norm:
            # # Normalize each trace
            data_min, data_max = np.nanmin(data_y), np.nanmax(data_y)
            data_y = np.interp(data_y, (data_min, data_max), (0, 1))
            if invert:
                data_y = 1 - data_y  # Invert a normalized signal
        else:
            if invert:
                print('!***! Can\'t invert a non-normalized trace!')

        ylim = [data_y.min(), data_y.max()]
        axis.set_ylim(ylim)
        # axis.set_ylim([0, 1.1])
        axis.yaxis.set_major_locator(ticker.LinearLocator(2))
        axis.yaxis.set_minor_locator(ticker.LinearLocator(10))
        # axis.yaxis.set_major_locator(ticker.AutoLocator())
        # axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axis.plot(data_x, data_y, color=color, linewidth=0.5)
    else:
        axis.plot(data, color=color, linewidth=0.5)


def plot_traceOverlay(axis, trace_vm, trace_ca):
    # Normalize and Plot a Vm and a Ca trace on the same plot
    plot_trace(axis, trace_vm, imagej=True, fps=408, x_span=256,
               norm=True, invert=True, color=signal_colors[0])
    plot_trace(axis, trace_ca, imagej=True, fps=408, x_span=256,
               norm=True, color=signal_colors[1])

    axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator((int(100/4))))
    # axis.set_yticks([])
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    # axis.spines['bottom'].set_visible(False)


def example_plot(axis):
    axis.plot([1, 2])
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)
    # axis.set_ylim([0, 1.1])


# Build figure
fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
gsfig = fig.add_gridspec(2, 1, height_ratios=[0.7, 0.3])  # Overall: ? row, ? columns

# Traces Section
# 3 columns for traces
gsData = gsfig[0].subgridspec(1, 2, width_ratios=[0.2, 0.8])  # Overall: ? row, ? columns
# Build heart sections
gsImages = gsData[0].subgridspec(2, 1)  # 2 rows, 1 columns for Activation Maps
axImage_Vm = fig.add_subplot(gsImages[0])
axImage_Ca = fig.add_subplot(gsImages[1])
# Build paced traces section
gsTraces = gsData[1].subgridspec(2, 1)
# Vm
gsTraces_Vm = gsTraces[0].subgridspec(2, 2)
# Pixel
axTraces_Vm_RV, axTraces_Vm_LV = fig.add_subplot(gsTraces_Vm[0]), fig.add_subplot(gsTraces_Vm[2])
# 5x5
axTraces_Vm_RV_5x5, axTraces_Vm_LV_5x5 = fig.add_subplot(gsTraces_Vm[1]), fig.add_subplot(gsTraces_Vm[3])

# Ca
gsTraces_Ca = gsTraces[1].subgridspec(2, 2)
# Pixel
axTraces_Ca_RV, axTraces_Ca_LV = fig.add_subplot(gsTraces_Ca[0]), fig.add_subplot(gsTraces_Ca[2])
# 5x5
axTraces_Ca_RV_5x5, axTraces_Ca_LV_5x5 = fig.add_subplot(gsTraces_Ca[1]), fig.add_subplot(gsTraces_Ca[3])

# Analysis Section
# 3 columns trace overlay, maps, and map statistics
gsAnalysis = gsfig[1].subgridspec(1, 3, width_ratios=[0.35, 0.35, 0.3])  # Overall: ? row, ? columns
# Build trace overlay section
axTracesOverlay = fig.add_subplot(gsAnalysis[0])
# Build maps section
gsMaps = gsAnalysis[1].subgridspec(2, 2)
axMap_ActVm = fig.add_subplot(gsMaps[0])
axMap_ActCa = fig.add_subplot(gsMaps[2])
axMap_APD = fig.add_subplot(gsMaps[1])
axMap_CAD = fig.add_subplot(gsMaps[3])
# Build map statistics section
axMapStats = fig.add_subplot(gsAnalysis[2])


# Plot Data Section
# Import heart image
heart_Vm = np.rot90(plt.imread('data/20190322-pigb/06-300_Vm_0001.tif'))
heart_Ca = np.rot90(plt.imread('data/20190322-pigb/06-300_Ca_0001.tif'))
# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
# RV, LV
H, W = heart_Vm.shape
Rois_Vm = [{'y': H - 240, 'x': 114, 'r': 5},
           {'y': H - 156, 'x': 250, 'r': 5}]
Rois_Ca = Rois_Vm

# Plot heart images
axImage_Vm.set_title('Vm', fontsize=16)
plot_heart(axis=axImage_Vm, heart_image=heart_Vm, rois=Rois_Vm)
axImage_Ca.set_title('Ca', fontsize=16)
plot_heart(axis=axImage_Ca, heart_image=heart_Ca, rois=Rois_Ca)


# Import Traces
# Load signal data, columns: index, fluorescence (counts)
Trace_Vm_RV = {1: np.genfromtxt('data/20190322-pigb/06-300_Vm_1x1-240x114.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/06-300_Vm_5x5-156x250.csv', delimiter=',')}
Trace_Vm_LV = {1: np.genfromtxt('data/20190322-pigb/06-300_Vm_1x1-240x114.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/06-300_Vm_5x5-156x250.csv', delimiter=',')}

Trace_Ca_RV = {1: np.genfromtxt('data/20190322-pigb/06-300_Ca_1x1-240x114.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/06-300_Ca_5x5-156x250.csv', delimiter=',')}
Trace_Ca_LV = {1: np.genfromtxt('data/20190322-pigb/06-300_Ca_1x1-240x114.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/06-300_Ca_5x5-156x250.csv', delimiter=',')}
# Plot paced traces
axTraces_Vm_RV.set_title('Single Pixel', fontsize=10)
axTraces_Vm_RV_5x5.set_title('5x5 Pixel', fontsize=10)

plot_trace(axTraces_Vm_RV, Trace_Vm_RV[1], imagej=True, fps=408, color='b', x_span=1024)
plot_trace(axTraces_Vm_RV_5x5, Trace_Vm_RV[5], imagej=True, fps=408, color='b', x_span=1024)
axTraces_Vm_RV.set_xticklabels([])
axTraces_Vm_RV_5x5.set_xticklabels([])
plot_trace(axTraces_Vm_LV, Trace_Vm_LV[1], imagej=True, fps=408, color='r', x_span=1024)
plot_trace(axTraces_Vm_LV_5x5, Trace_Vm_LV[5], imagej=True, fps=408, color='r', x_span=1024)


plot_trace(axTraces_Ca_RV, Trace_Ca_RV[1], imagej=True, fps=408, color='b', x_span=1024)
plot_trace(axTraces_Ca_RV_5x5, Trace_Ca_RV[5], imagej=True, fps=408, color='b', x_span=1024)
axTraces_Ca_RV.set_xticklabels([])
axTraces_Ca_RV_5x5.set_xticklabels([])
plot_trace(axTraces_Ca_LV, Trace_Ca_LV[1], imagej=True, fps=408, color='r', x_span=1024)
plot_trace(axTraces_Ca_LV_5x5, Trace_Ca_LV[5], imagej=True, fps=408, color='r', x_span=1024)

# Plot Analysis Section
# Plot trace overlay
# plot_trace(axTracesOverlay, Trace_Vm_LV[5], imagej=True, fps=408, color='r', x_span=256)
# plot_trace(axTracesOverlay, Trace_Ca_LV[5], imagej=True, fps=408, color='r', x_span=256)
plot_traceOverlay(axTracesOverlay, trace_vm=Trace_Vm_LV[5], trace_ca=Trace_Ca_LV[5])
# Plot maps
axMap_ActVm.set_title('Vm Act.', fontsize=8)
axMap_ActCa.set_title('Ca Act.', fontsize=8)
axMap_APD.set_title('APD-80', fontsize=8)
axMap_CAD.set_title('CAD-80.', fontsize=8)


# Fill rest with example plots
# Data
# example_plot(axImage_Vm)
# example_plot(axImage_Ca)

# example_plot(axTraces_Vm_RV)
# example_plot(axTraces_Vm_LV)
# example_plot(axTraces_Vm_RV_5x5)
# example_plot(axTraces_Vm_LV_5x5)
# example_plot(axTraces_Ca_RV)
# example_plot(axTraces_Ca_LV)
# example_plot(axTraces_Ca_RV_5x5)
# example_plot(axTraces_Ca_LV_5x5)

# Analysis
# example_plot(axTracesOverlay)

example_plot(axMap_ActVm)
example_plot(axMap_ActCa)
example_plot(axMap_APD)
example_plot(axMap_CAD)

example_plot(axMapStats)


# Show and save figure
fig.show()
fig.savefig('JoVE_Fig3-Paced.svg')
