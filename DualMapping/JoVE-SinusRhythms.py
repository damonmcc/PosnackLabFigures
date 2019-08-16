import numpy as np
import scipy.signal as sig
from decimal import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib import rcParams
from matplotlib.patches import Ellipse, Circle, Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import colorsys
import ScientificColourMaps5 as scm
import warnings

MAX_COUNTS_16BIT = 65536
colors_rois = ['b', 'r', 'k']
rcParams['font.family'] = "Arial"
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
X_CROP = [0, 80]   # to cut from left, right
Y_CROP = [30, 80]   # to cut from bottom, top


def plot_heart(axis, heart_image, scale=True, scale_text=True, rois=None):
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

    scale : bool, optional
        If True, include scale bar.
        Defaults to True.

    scale_text : bool, optional
        If True, include text with the scale bar.
        Defaults to True.

    rois : list, optional
        A list of dictionaries with structure {'y': int, 'x': int, 'r': int}

    Returns
    -------
    image : `~matplotlib.image.AxesImage`
    """
    # Setup plot
    height, width, = heart_image.shape[0], heart_image.shape[1]  # X, Y flipped due to rotation
    x_crop, y_crop = [X_CROP[0], width - X_CROP[1]], [height - Y_CROP[0], Y_CROP[1]]
    print('Heart plot (W x H): ', width, ' x ', height)
    axis.axis('off')
    img = axis.imshow(heart_image, cmap='bone')

    if rois:
        # Create ROIs
        for idx, roi in enumerate(rois):
            roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc=None, fill=None,
                                ec=colors_rois[idx], lw=1)
            axis.add_artist(roi_circle)

    # patch = Ellipse((width/2, height/2), width=width, height=height, transform=axis.transData)
    # img.set_clip_path(patch)
    # Scale Bar
    scale_px_cm = 1 / 0.0149
    heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
    if scale:
        if scale_text:
            heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '1 cm',
                                              loc=4, pad=0.2, color='w', frameon=False,
                                              fontproperties=fm.FontProperties(size=7, weight='semibold'))
        else:
            # Scale bar, no text
            heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '', sep=0,
                                              loc=4, pad=1, color='w', frameon=False,
                                              fontproperties=fm.FontProperties(size=2))
        axis.add_artist(heart_scale_bar)

    axis.set_xlim(x_crop)
    axis.set_ylim(y_crop)

    return img


def plot_trace(axis, data, imagej=False, fps=None, x_span=0, x_end=None,
               frac=True, norm=False, invert=False, filter_lp=False,
               color='b', x_ticks=True):
    data_x, data_y = 0, 0

    if imagej:
        if not x_end:
            x_end = len(data)   # Includes X,Y header row (nan,nan)
        x_start = x_end - x_span

        # data_x = data[x_start:x_end, 0]             # All rows of the first column (skip X,Y header row)
        data_x = data[1:x_span + 1, 0]             # All rows of the first column (skip X,Y header row)
        if fps:
            # convert to ms
            data_x = ((data_x - 1) / fps) * 1000    # ensure range is 0 - max
            # data_x = data_x.astype(int)
            # axis.xaxis.set_major_locator(ticker.IndexLocator(base=1000, offset=500))
        # axis.xaxis.set_major_locator(ticker.AutoLocator())
        # axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        data_y_counts = data[x_start:x_end, 1].astype(int)     # rows of the first column (skip X,Y header row)
        counts_min = data_y_counts.min()
        # data_y_counts_delta = data_y.max() - data_y_.min()
        # MAX_COUNTS_16BIT

        if frac:
            # # convert y-axis from counts to percentage range of max counts
            # data_y = data_y_counts / MAX_COUNTS_16BIT * 100
            # axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            # Convert y-axis from counts to dF / F: (F_t - F0) / F0
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # Get max and min
            # data_min, data_max = np.nanmin(data_y_counts), np.nanmax(data_y_counts)
            f_0 = counts_min
            f_t = data_y_counts
            data_y = (f_t - f_0) / f_0
        else:
            # Shift y-axis counts to start at zero
            data_y = data_y_counts - counts_min

        if filter_lp:
            print('* Filtering data: Low Pass')
            dt = 1 / 408
            freq = 75

            fs = 1 / dt
            wn = (freq / (fs / 2))
            [b, a] = sig.butter(5, wn)
            data_y = sig.filtfilt(b, a, data_y)
            print('* Data Filtered')

        if norm:
            # Normalize each trace
            # Get max and min
            data_min, data_max = np.nanmin(data_y), np.nanmax(data_y)
            data_y = np.interp(data_y, (data_min, data_max), (0, 1))
            if invert:
                data_y = 1 - data_y  # Invert a normalized signal
        else:
            if invert:
                print('!***! Can\'t invert a non-normalized trace!')

        axis.tick_params(labelsize=fontsize4)
        if x_ticks:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(1000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(int(1000/4)))
        else:
            axis.set_xticks([])
            axis.set_xticklabels([])

        ylim = [data_y.min(), data_y.max()]
        axis.set_ylim(ylim)
        # axis.set_ylim([0, 1.1])
        axis.yaxis.set_major_locator(ticker.LinearLocator(2))
        axis.yaxis.set_minor_locator(ticker.LinearLocator(5))
        # axis.yaxis.set_major_locator(ticker.AutoLocator())
        # axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)

        axis.plot(data_x, data_y, color, linewidth=0.2)
    else:
        # axis.plot(data, color=color, linewidth=0.5)
        print('***! Not imagej traces')

    return data_x, data_y


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
gsfig = fig.add_gridspec(2, 1, height_ratios=[0.07, 0.93])  # Overall: ? row, ? columns

# Top row for Title-ish text
# gsText = gsfig[0].subgridspec(1, 1)  # Overall: ? row, ? columns
axText = fig.add_subplot(gsfig[0])  # Overall: ? row, ? columns
axText.axis('off')
# axText.set_title('Sinus Rhythm', fontsize=18)
axText.text(0.425, 1, 'Sinus Rhythm',
            ha='center', va='top', size=fontsize1, weight='semibold')
axText.text(0.825, 1, 'Ventricular Fibrillation',
            ha='center', va='top', size=fontsize1, weight='semibold')

# 3 columns for data
gs0 = gsfig[1].subgridspec(1, 3, width_ratios=[0.2, 0.4, 0.4])  # Overall: ? row, ? columns

# Build heart sections
gsImages = gs0[0].subgridspec(2, 1)  # 2 rows, 1 columns for Activation Maps
axImage_Vm = fig.add_subplot(gsImages[0])
axImage_Ca = fig.add_subplot(gsImages[1])

trace_wspace = 0.4
trace_hspace = 0.3
signal_hspace = 0.3
# Build NSR Traces section
gsTracesNSR = gs0[1].subgridspec(2, 1, hspace=signal_hspace)
# Vm
gsTracesNSR_Vm = gsTracesNSR[0].subgridspec(2, 2, hspace=trace_hspace, wspace=trace_wspace)
# Pixel
axTracesNSR_Vm_RV, axTracesNSR_Vm_LV = fig.add_subplot(gsTracesNSR_Vm[0]), fig.add_subplot(gsTracesNSR_Vm[2])
# 5x5
axTracesNSR_Vm_RV_5x5, axTracesNSR_Vm_LV_5x5 = fig.add_subplot(gsTracesNSR_Vm[1]), fig.add_subplot(gsTracesNSR_Vm[3])

# Ca
gsTracesNSR_Ca = gsTracesNSR[1].subgridspec(2, 2, hspace=trace_hspace, wspace=trace_wspace)
# Pixel
axTracesNSR_Ca_RV, axTracesNSR_Ca_LV = fig.add_subplot(gsTracesNSR_Ca[0]), fig.add_subplot(gsTracesNSR_Ca[2])
# 5x5
axTracesNSR_Ca_RV_5x5, axTracesNSR_Ca_LV_5x5 = fig.add_subplot(gsTracesNSR_Ca[1]), fig.add_subplot(gsTracesNSR_Ca[3])


# Build VF Traces section
gsTracesVF = gs0[2].subgridspec(2, 1, hspace=signal_hspace)
# Vm
gsTracesVF_Vm = gsTracesVF[0].subgridspec(2, 2, hspace=trace_hspace, wspace=trace_wspace)
# Pixel
axTracesVF_Vm_RV, axTracesVF_Vm_LV = fig.add_subplot(gsTracesVF_Vm[0]), fig.add_subplot(gsTracesVF_Vm[2])
# 5x5
axTracesVF_Vm_RV_5x5, axTracesVF_Vm_LV_5x5 = fig.add_subplot(gsTracesVF_Vm[1]), fig.add_subplot(gsTracesVF_Vm[3])

# Ca
gsTracesVF_Ca = gsTracesVF[1].subgridspec(2, 2, hspace=trace_hspace, wspace=trace_wspace)
# Pixel
axTracesVF_Ca_RV, axTracesVF_Ca_LV = fig.add_subplot(gsTracesVF_Ca[0]), fig.add_subplot(gsTracesVF_Ca[2])
# 5x5
axTracesVF_Ca_RV_5x5, axTracesVF_Ca_LV_5x5 = fig.add_subplot(gsTracesVF_Ca[1]), fig.add_subplot(gsTracesVF_Ca[3])


# Import heart image
# heart_NSR_Vm = np.fliplr(np.rot90(plt.imread('data/20190322-piga/18-NSR_Vm_0001.tif')))
# heart_NSR_Ca = np.fliplr(np.rot90(plt.imread('data/20190322-piga/18-NSR_Ca_0001.tif')))

# heart_NSR_Vm = np.rot90(plt.imread('data/20190322-piga/18-NSR_Vm_0001.tif'))
# heart_NSR_Ca = np.rot90(plt.imread('data/20190322-piga/18-NSR_Ca_0001.tif'))

heart_VF_Vm = np.rot90(plt.imread('data/20190322-piga/19-VFIB_Vm_0001.tif'))
heart_VF_Ca = np.rot90(plt.imread('data/20190322-piga/19-VFIB_Ca_0001.tif'))
# ret, heart_thresh = cv2.threshold(heart, 150, np.nan, cv2.THRESH_TOZERO)


# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
# RV, LV
H, W = heart_VF_Vm.shape
RoisVF_Vm = [{'y': H - 395, 'x': 179, 'r': 15},
             {'y': H - 300, 'x': 310, 'r': 15}]
RoisVF_Ca = RoisVF_Vm


# Import Traces
# Load signal data, columns: index, fluorescence (counts)
# NSR
TraceNSR_Vm_RV = {1: np.genfromtxt('data/20190322-piga/18-NSR_Vm_15x15-234x90.csv', delimiter=','),
                  5: np.genfromtxt('data/20190322-piga/18-NSR_Vm_30x30-234x90.csv', delimiter=',')}
TraceNSR_Vm_LV = {1: np.genfromtxt('data/20190322-piga/18-NSR_Vm_15x15-181x270.csv', delimiter=','),
                  5: np.genfromtxt('data/20190322-piga/18-NSR_Vm_30x30-181x270.csv', delimiter=',')}

TraceNSR_Ca_RV = {1: np.genfromtxt('data/20190322-piga/18-NSR_Ca_15x15-234x90.csv', delimiter=','),
                  5: np.genfromtxt('data/20190322-piga/18-NSR_Ca_30x30-234x90.csv', delimiter=',')}
TraceNSR_Ca_LV = {1: np.genfromtxt('data/20190322-piga/18-NSR_Ca_15x15-181x270.csv', delimiter=','),
                  5: np.genfromtxt('data/20190322-piga/18-NSR_Ca_30x30-181x270.csv', delimiter=',')}

# VF
TraceVF_Vm_RV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_15x15-395x179.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_30x30-395x179.csv', delimiter=',')}
TraceVF_Vm_LV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_15x15-300x310.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Vm_30x30-300x310.csv', delimiter=',')}

TraceVF_Ca_RV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_15x15-395x179.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_30x30-395x179.csv', delimiter=',')}
TraceVF_Ca_LV = {1: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_15x15-300x310.csv', delimiter=','),
                 5: np.genfromtxt('data/20190322-piga/19-VFIB_Ca_30x30-300x310.csv', delimiter=',')}


# Plot heart images
axImage_Vm.set_title('Vm', size=fontsize1, weight='semibold')
plot_heart(axis=axImage_Vm, heart_image=heart_VF_Vm, scale_text=True,
           rois=RoisVF_Vm)
# axImage_Vm.text(axImage_label_x, axImage_label_y, 'Vm', transform=axImage_Vm.transAxes,
#                 rotation=90, ha='center', va='center', fontproperties=axImages_label_font)
axImage_Ca.set_title('Ca', size=fontsize1, weight='semibold')
plot_heart(axis=axImage_Ca, heart_image=heart_VF_Ca, scale_text=False,
           rois=RoisVF_Vm)


idx_end = len(TraceNSR_Vm_RV[1]) - 300
idx_span = 512 + 300
axTracesNSR_Vm_RV.set_title('15x15 Pixel', fontsize=fontsize2, weight='semibold')
axTracesNSR_Vm_RV_5x5.set_title('30x30 Pixel', fontsize=fontsize2, weight='semibold')
axTraces_label_x = -0.25
axTraces_label_y = 0.5
axTraces_label_font = fm.FontProperties(size=fontsize3, weight='semibold')
# Plot NSR traces
# Vm
axTracesNSR_Vm_RV.text(axTraces_label_x, axTraces_label_y, 'RV', transform=axTracesNSR_Vm_RV.transAxes,
                       rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTracesNSR_Vm_RV, TraceNSR_Vm_RV[1], imagej=True, fps=408,
           color='b', x_span=idx_span, x_end=idx_end, x_ticks=False)
plot_trace(axTracesNSR_Vm_RV_5x5, TraceNSR_Vm_RV[5], imagej=True, fps=408,
           color='b', x_span=idx_span, x_end=idx_end, x_ticks=False)
axTracesNSR_Vm_LV.text(axTraces_label_x, axTraces_label_y, 'LV', transform=axTracesNSR_Vm_LV.transAxes,
                       rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTracesNSR_Vm_LV, TraceNSR_Vm_LV[1], imagej=True, fps=408,
           color='r', x_span=idx_span, x_end=idx_end)
plot_trace(axTracesNSR_Vm_LV_5x5, TraceNSR_Vm_LV[5], imagej=True, fps=408,
           color='r', x_span=idx_span, x_end=idx_end)
# Ca
axTracesNSR_Ca_RV.text(axTraces_label_x, axTraces_label_y, 'RV', transform=axTracesNSR_Ca_RV.transAxes,
                       rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTracesNSR_Ca_RV, TraceNSR_Ca_RV[1], imagej=True, fps=408,
           color='b', x_span=idx_span, x_end=idx_end, x_ticks=False)
plot_trace(axTracesNSR_Ca_RV_5x5, TraceNSR_Ca_RV[5], imagej=True, fps=408,
           color='b', x_span=idx_span, x_end=idx_end, x_ticks=False)
axTracesNSR_Ca_LV.text(axTraces_label_x, axTraces_label_y, 'LV', transform=axTracesNSR_Ca_LV.transAxes,
                       rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTracesNSR_Ca_LV, TraceNSR_Ca_LV[1], imagej=True, fps=408,
           color='r', x_span=idx_span, x_end=idx_end)
plot_trace(axTracesNSR_Ca_LV_5x5, TraceNSR_Ca_LV[5], imagej=True, fps=408,
           color='r', x_span=idx_span, x_end=idx_end)

# Plot VF traces
axTracesVF_Vm_RV.set_title('15x15 Pixel', fontsize=fontsize2, weight='semibold')
axTracesVF_Vm_RV_5x5.set_title('30x30 Pixel', fontsize=fontsize2, weight='semibold')
# Vm
plot_trace(axTracesVF_Vm_RV, TraceVF_Vm_RV[1], imagej=True, fps=408,
           color='b', x_span=idx_span, x_ticks=False)
plot_trace(axTracesVF_Vm_RV_5x5, TraceVF_Vm_RV[5], imagej=True, fps=408,
           color='b', x_span=idx_span, x_ticks=False)
plot_trace(axTracesVF_Vm_LV, TraceVF_Vm_LV[1], imagej=True, fps=408,
           color='r', x_span=idx_span)
plot_trace(axTracesVF_Vm_LV_5x5, TraceVF_Vm_LV[5], imagej=True, fps=408,
           color='r', x_span=idx_span)
# Ca
plot_trace(axTracesVF_Ca_RV, TraceVF_Ca_RV[1], imagej=True, fps=408,
           color='b', x_span=idx_span, x_ticks=False)
plot_trace(axTracesVF_Ca_RV_5x5, TraceVF_Ca_RV[5], imagej=True, fps=408,
           color='b', x_span=idx_span, x_ticks=False)
plot_trace(axTracesVF_Ca_LV, TraceVF_Ca_LV[1], imagej=True, fps=408,
           color='r', x_span=idx_span)
plot_trace(axTracesVF_Ca_LV_5x5, TraceVF_Ca_LV[5], imagej=True, fps=408,
           color='r', x_span=idx_span)


# Fill rest with example plots
# NSR
# example_plot(axTracesNSR_Vm_RV)
# example_plot(axTracesNSR_Vm_LV)
# example_plot(axTracesNSR_Vm_RV_5x5)
# example_plot(axTracesNSR_Vm_LV_5x5)

# example_plot(axTracesNSR_Ca_RV)
# example_plot(axTracesNSR_Ca_LV)
# example_plot(axTracesNSR_Ca_RV_5x5)
# example_plot(axTracesNSR_Ca_LV_5x5)

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
fig.savefig('JoVE-SinusRhythms.svg')
