
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
roi_colors = ['r', 'b', 'k']


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
    img = axis.imshow(heart_image[:, :512], cmap='bone')

    if rois:
        # Create ROIs and get colors of their pixels
        for idx, roi in enumerate(rois):
            roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='w',
                                ec=roi_colors[idx], lw=2)
            axis.add_artist(roi_circle)

    # patch = Ellipse((width/2, height/2), width=width, height=height, transform=axis.transData)
    # img.set_clip_path(patch)
    # # Scale Bar
    # heart_scale = [67, 67]  # x, y (pixels/cm)
    # heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '1 cm',
    # 'lower right', pad=0.2, color='w', frameon=False,
    # fontproperties=fm.FontProperties(size=7, weight='bold'))
    # axis.add_artist(heart_scale_bar)

    return img


def plot_trace(axis, data, imagej=False, fps=None, color='b',
               x_start=0):
    if imagej:
        data_x = data[1 + x_start:, 0] - x_start             # All rows of the first column (skip X,Y header row)
        if fps:
            # convert to ms
            data_x = ((data_x - 1) / fps) * 1000    # ensure range is 0 - max
            data_x = data_x.astype(int)
            # axis.xaxis.set_major_locator(ticker.IndexLocator(base=1000, offset=500))
        # axis.set_xlim(xlim)
        # axis.xaxis.set_major_locator(ticker.AutoLocator())
        # axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axis.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(250))
        axis.tick_params(labelsize=6)

        data_y_counts = data[1 + x_start:, 1].astype(int)     # rows of the first column (skip X,Y header row)
        # data_y_counts_delta = data_y.max() - data_y_.min()
        # MAX_COUNTS_16BIT
        # # Normalize each trace
        # data_min, data_max = np.nanmin(trace), np.nanmax(trace)
        # trace = np.interp(trace, (data_min, data_max), (0, 1))
        data_y = data_y_counts

        ylim = [data_y.min(), data_y.max()]
        axis.set_ylim(ylim)
        # axis.set_ylim([0, 1.1])
        axis.yaxis.set_major_locator(ticker.LinearLocator(2))
        axis.yaxis.set_minor_locator(ticker.LinearLocator(10))
        # axis.yaxis.set_major_locator(ticker.AutoLocator())
        # axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axis.plot(data_x, data_y, color=color, linewidth=0.1)
    else:
        axis.plot(data, color=color, linewidth=0.1)


def example_plot(axis):
    axis.plot([1, 2])
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)
    axis.set_ylim([0, 1.1])


# Build figure
fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 3, width_ratios=[0.2, 0.4, 0.4])  # Overall: ? row, ? columns

# Build heart sections
gsImages = gs0[0].subgridspec(2, 1)  # 2 rows, 1 columns for Activation Maps
axImage_Vm = fig.add_subplot(gsImages[0])
axImage_Ca = fig.add_subplot(gsImages[1])

# Build NSR Traces section
gsTracesNSR = gs0[1].subgridspec(2, 1)
# Vm
gsTracesNSR_Vm = gsTracesNSR[0].subgridspec(2, 2)
# Pixel
axTracesNSR_Vm_RV, axTracesNSR_Vm_LV = fig.add_subplot(gsTracesNSR_Vm[0]), fig.add_subplot(gsTracesNSR_Vm[2])
# 5x5
axTracesNSR_Vm_RV_5x5, axTracesNSR_Vm_LV_5x5 = fig.add_subplot(gsTracesNSR_Vm[1]), fig.add_subplot(gsTracesNSR_Vm[3])

# Ca
gsTracesNSR_Ca = gsTracesNSR[1].subgridspec(2, 2)
# Pixel
axTracesNSR_Ca_RV, axTracesNSR_Ca_LV = fig.add_subplot(gsTracesNSR_Ca[0]), fig.add_subplot(gsTracesNSR_Ca[2])
# 5x5
axTracesNSR_Ca_RV_5x5, axTracesNSR_Ca_LV_5x5 = fig.add_subplot(gsTracesNSR_Ca[1]), fig.add_subplot(gsTracesNSR_Ca[3])


# Build VF Traces section
gsTracesVF = gs0[2].subgridspec(2, 1)
# Vm
gsTracesVF_Vm = gsTracesVF[0].subgridspec(2, 2)
# Pixel
axTracesVF_Vm_RV, axTracesVF_Vm_LV = fig.add_subplot(gsTracesVF_Vm[0]), fig.add_subplot(gsTracesVF_Vm[2])
# 5x5
axTracesVF_Vm_RV_5x5, axTracesVF_Vm_LV_5x5 = fig.add_subplot(gsTracesVF_Vm[1]), fig.add_subplot(gsTracesVF_Vm[3])

# Ca
gsTracesVF_Ca = gsTracesVF[1].subgridspec(2, 2)
# Pixel
axTracesVF_Ca_RV, axTracesVF_Ca_LV = fig.add_subplot(gsTracesVF_Ca[0]), fig.add_subplot(gsTracesVF_Ca[2])
# 5x5
axTracesVF_Ca_RV_5x5, axTracesVF_Ca_LV_5x5 = fig.add_subplot(gsTracesVF_Ca[1]), fig.add_subplot(gsTracesVF_Ca[3])


# Import heart image
heart_VF_Vm = np.fliplr(np.rot90(plt.imread('data/20190322-piga/19-VFIB_Vm_0001.tif')))
# heart_VF_Vm = np.fliplr(np.rot90(plt.imread('data/20190322-piga/19-VFIB_Vm_0001.tif')[720-512:, :]))
heart_VF_Ca = np.fliplr(np.rot90(plt.imread('data/20190322-piga/19-VFIB_Ca_0001.tif')))
# ret, heart_thresh = cv2.threshold(heart, 150, np.nan, cv2.THRESH_TOZERO)


# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
W, H = heart_VF_Vm.shape
RoisVF_Vm = [{'y': W - 312, 'x': H - 550, 'r': 5},
             {'y': W - 390, 'x': H - 384, 'r': 5}]

RoisVF_Ca = [{'y': W - 336, 'x': H - 548, 'r': 5},
             {'y': W - 414, 'x': H - 382, 'r': 5}]


# Import Traces
# Load signal data, columns: time (s), fluorescence (norm)
# Pair with stim activation time used for activation maps
TraceVF_Vm_RV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_1x1-390x384.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_5x5-390x384.csv', delimiter=',')}
TraceVF_Vm_LV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_1x1-312x550.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_5x5-312x550.csv', delimiter=',')}

TraceVF_Ca_RV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_1x1-414x382.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_5x5-414x382.csv', delimiter=',')}
TraceVF_Ca_LV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_1x1-336x548.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_5x5-336x548.csv', delimiter=',')}


# Plot heart images
plot_heart(axis=axImage_Vm, heart_image=heart_VF_Vm, rois=RoisVF_Vm)
axImage_Vm.set_title('Vm', fontsize=18)
plot_heart(axis=axImage_Ca, heart_image=heart_VF_Ca, rois=RoisVF_Ca)
axImage_Ca.set_title('Ca', fontsize=18)


# Plot NSR traces
axTracesNSR_Vm_RV.set_title('Single Pixel', fontsize=10)
axTracesNSR_Vm_RV_5x5.set_title('5x5 Pixel', fontsize=10)
axTracesNSR_Vm_RV.set_ylabel('RV', fontsize=10)
axTracesNSR_Vm_LV.set_ylabel('LV', fontsize=10)
axTracesNSR_Ca_RV.set_ylabel('RV', fontsize=10)
axTracesNSR_Ca_LV.set_ylabel('LV', fontsize=10)

# Plot VF traces
axTracesVF_Vm_RV.set_title('Single Pixel', fontsize=10)
axTracesVF_Vm_RV_5x5.set_title('5x5 Pixel', fontsize=10)

plot_trace(axTracesVF_Vm_RV, TraceVF_Vm_RV[1], imagej=True, fps=408, color='b', x_start=1024)
plot_trace(axTracesVF_Vm_RV_5x5, TraceVF_Vm_RV[5], imagej=True, fps=408, color='b', x_start=1024)
plot_trace(axTracesVF_Ca_RV, TraceVF_Ca_RV[1], imagej=True, fps=408, color='b', x_start=1024)
plot_trace(axTracesVF_Ca_RV_5x5, TraceVF_Ca_RV[5], imagej=True, fps=408, color='b', x_start=1024)

plot_trace(axTracesVF_Vm_LV, TraceVF_Vm_LV[1], imagej=True, fps=408, color='r', x_start=1024)
plot_trace(axTracesVF_Vm_LV_5x5, TraceVF_Vm_LV[5], imagej=True, fps=408, color='r', x_start=1024)
plot_trace(axTracesVF_Ca_LV, TraceVF_Ca_LV[1], imagej=True, fps=408, color='r', x_start=1024)
plot_trace(axTracesVF_Ca_LV_5x5, TraceVF_Ca_LV[5], imagej=True, fps=408, color='r', x_start=1024)




# Fill rest with example plots
# NSR
example_plot(axTracesNSR_Vm_RV)
example_plot(axTracesNSR_Vm_LV)
example_plot(axTracesNSR_Vm_RV_5x5)
example_plot(axTracesNSR_Vm_LV_5x5)

example_plot(axTracesNSR_Ca_RV)
example_plot(axTracesNSR_Ca_LV)
example_plot(axTracesNSR_Ca_RV_5x5)
example_plot(axTracesNSR_Ca_LV_5x5)

# VF
# example_plot(axTracesVF_Vm_RV)
# example_plot(axTracesVF_Vm_LV)
# example_plot(axTracesVF_Vm_RV_5x5)
# example_plot(axTracesVm_VF_LV_5x5)

# example_plot(axTracesVF_Ca_RV)
# example_plot(axTracesVF_Ca_LV)
# example_plot(axTracesVF_Ca_RV_5x5)
# example_plot(axTracesVF_Ca_LV_5x5)


# Show and save figure
fig.show()
fig.savefig('JoVE_Fig4-SinusRhythms.svg')
