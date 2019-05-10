import cv2
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
import warnings

warnings.filterwarnings('ignore')

# Set Colors
# roi_colors = ['0.45', '0.65', '0.85']
roi_colors = ['0.2', '0.2', '0.2']

colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'grey'
colorMEHP = 'black'
exposure_colors = ['black', 'red']
context_colors = ['0.8', '0.45']

colorsROI_VmRAW = []
colorsROI_VmRGB = []
colorsROI_CaRAW = []
colorsROI_CaRGB = []


def plot_heart(axis, heart_image):
    # Setup plot
    height, width, = heart_image.shape[0], heart_image.shape[1]   # X, Y flipped due to rotation
    axis.axis('off')
    img = axis.imshow(heart_image, cmap='bone')

    # patch = Ellipse((width/2, height/2), width=width, height=height, transform=axis.transData)
    # img.set_clip_path(patch)
    # Scale Bar
    heart_scale = [67, 67]  # x, y (pixels/cm)
    heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '1 cm', 'lower right', pad=0.2,
                                      color='k', frameon=False, fontproperties=fm.FontProperties(size=7))
    axis.add_artist(heart_scale_bar)


def plot_ActMapVm(axis, actMap):
    # Setup plot
    height, width, = actMap.shape[0], actMap.shape[1]   # X, Y flipped due to rotation
    x_crop, y_crop = [0, width], [height - 100, 70]

    # axis.axis('off')
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_xticklabels([])

    axis.set_ylabel('Vm', fontsize=18)
    # axis.set_xlim(x_crop)
    axis.set_ylim(y_crop)
    axis.set_title('Activation Map', fontsize=12)

    # Plot Activation Map
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # Create ROIs and get colors of their pixels
    for idx, roi in enumerate(rois):
        roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='w',
                            ec=roi_colors[idx], lw=2)
        axis.add_artist(roi_circle)
    ros_wedge = Wedge((ros['x'] - 10, ros['y']), ros['r'], 125, 185, fc='k', ec='k', lw=1)
    axis.add_artist(ros_wedge)
    for idx, roi in enumerate(rois):
        # print('# ', idx)
        colorRAW = img.cmap(img.norm(actMap[roi['y'], roi['x']]))
        colorRGB = []
        # print('ROI# ', idx, ' colorRAW: ', colorRAW)
        for idxx, norm in enumerate(colorRAW):
            colorRGB.append(int(colorRAW[idxx] * 255))

        # Desaturate colors for plotting
        print('* colorRAW was :', colorRAW)
        colorHSV = colorsys.rgb_to_hsv(colorRAW[0], colorRAW[1], colorRAW[2])
        print('colorHSV was :', colorHSV)
        colorHSV = colorHSV[0], colorHSV[1] * 0.7, colorHSV[2]
        print('colorHSV is  :', colorHSV)
        colorRAW = colorsys.hsv_to_rgb(colorHSV[0], colorHSV[1], colorHSV[2])
        print('* colorRAW is :', colorRAW)

        colorsROI_VmRAW.append(colorRAW)
        colorsROI_VmRGB.append(colorRGB)
    print('colorsRGB: ', colorsROI_VmRGB)

    return img


def plot_ActMapCa(axis, actMap):
    # Setup plot
    height, width, = actMap.shape[0], actMap.shape[1]   # X, Y flipped due to rotation
    x_crop, y_crop = [0, width], [height - 100, 70]

    # axis.axis('off')
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_xticklabels([])

    axis.set_ylabel('Ca', fontsize=18)
    # axis.set_xlim(x_crop)
    axis.set_ylim(y_crop)
    # axis.set_title('Ca', fontsize=12)

    # Plot Activation Map
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    axins1 = inset_axes(axis, width="5%",  # width = 5% of parent_bbox width
                        height="80%",  # height : 80%
                        loc=1, bbox_to_anchor=(-0, 0.5, 1, 1), bbox_transform=axis.transAxes,
                        borderpad=0)
    cb1 = plt.colorbar(img, cax=axins1, orientation="vertical")
    cb1.set_label('Activation Time (ms)', fontsize=8)
    # cb1.set_ticks([0, 50, 100, 150])
    cb1.ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    cb1.ax.tick_params(labelsize=6)
    cb1.ax.yaxis.set_label_position('left')

    # Create ROIs and get colors of their pixels
    for idx, roi in enumerate(rois):
        roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='w',
                            ec=roi_colors[idx], lw=2)
        axis.add_artist(roi_circle)
    ros_wedge = Wedge((ros['x'] - 10, ros['y']), ros['r'], 125, 185, fc='k', ec='k', lw=1)
    axis.add_artist(ros_wedge)
    for idx, roi in enumerate(rois):
        # print('# ', idx)
        colorRAW = img.cmap(img.norm(actMap[roi['y'], roi['x']]))
        colorRGB = []
        print('ROI# ', idx, ' colorRAW: ', colorRAW)
        for idxx, norm in enumerate(colorRAW):
            colorRGB.append(int(colorRAW[idxx] * 255))

        print('* colorRAW was :', colorRAW)
        colorHSV = colorsys.rgb_to_hsv(colorRAW[0], colorRAW[1], colorRAW[2])
        print('colorHSV was :', colorHSV)
        colorHSV = colorHSV[0], colorHSV[1] * 0.7, colorHSV[2]
        print('colorHSV is  :', colorHSV)
        colorRAW = colorsys.hsv_to_rgb(colorHSV[0], colorHSV[1], colorHSV[2])
        print('* colorRAW is :', colorRAW)

        colorsROI_CaRAW.append(colorRAW)
        colorsROI_CaRGB.append(colorRGB)
    print('colorsRGB: ', colorsROI_CaRGB)

    return img


def plot_TracesVm(axis, data, time_start=0.0, idx_start=None, time_window=None, time_end = None, idx_end=None):
    # Setup data and time (ms) variables
    data_Vm = []
    traces_count = data.shape[1] - 1    # Number of columns after time column
    times_Vm = (data[:, 0]) * 1000  # seconds to ms
    time_start, time_window = time_start * 1000, time_window * 1000
    # Find index of first value after start time
    for idx, time in enumerate(times_Vm):
        if time < time_start:
            pass
        else:
            idx_start = idx
            break

    if time_window:
        # Find index of first value after end time
        for idx, time in enumerate(times_Vm):
            if time < time_start + time_window:
                pass
            else:
                idx_end = idx
                break
        # Convert possibly strange floats to rounded Decimals
        time_start, time_window = Decimal(time_start).quantize(Decimal('.001'), rounding=ROUND_UP),\
                                  Decimal(time_window).quantize(Decimal('.001'), rounding=ROUND_UP)
        # Slice time array based on indices
        times_Vm = times_Vm[idx_start:idx_end]
        time_end = time_start + time_window
        axis.set_xlim([float(time_start), float(time_end)])

    # Prepare axis for plotting
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    # axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    axis.set_ylim([0, 1.1])
    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # axis.set_xlabel('Time (ms)', fontsize=12)
    # axis.set_ylabel('Norm. Fluor., Vm', fontsize=12)
    axis.set_title('Normalized Fluorescence ', fontsize=12)

    # ECG Scale: ms and Norm. Fluor. bars forming an L
    ECGScaleTime = [50, 250 / 1000]  # 50 ms, 0.25 Norm. Fluor.
    ECGScaleOrigin = [axis.get_xlim()[1] - 1.5*ECGScaleTime[0], axis.get_ylim()[0]+0.01]
    # ECGScaleOrigin = [axis.get_xlim()[1] - 20, axis.get_ylim()[0] + 0.3]
    ECGScaleOriginPad = [2, 0.05]
    # Time scale bar
    axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0] + ECGScaleTime[0]],
              [ECGScaleOrigin[1], ECGScaleOrigin[1]],
              "k-", linewidth=1)
    axis.text(ECGScaleOrigin[0], ECGScaleOrigin[1] - ECGScaleOriginPad[1],
              str(ECGScaleTime[0]) + 'ms',
              ha='left', va='top', fontsize=7, fontweight='bold')
    # # Voltage scale bar
    # axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0]],
    #           [ECGScaleOrigin[1], ECGScaleOrigin[1] + ECGScaleTime[1]],"k-", linewidth=1)
    # axis.text(ECGScaleOrigin[0] - ECGScaleOriginPad[0], ECGScaleOrigin[1],
    #           str(ECGScaleTime[1]),
    #           ha='right', va='bottom', fontsize=7, fontweight='bold')

    for idx in range(traces_count):
        trace = data[:, idx + 1]
        if time_window:
            trace = trace[idx_start:idx_end]
        # Normalize each trace
        data_min, data_max = np.nanmin(trace), np.nanmax(trace)
        trace = np.interp(trace, (data_min, data_max), (0, 1))
        data_Vm.append(trace)
        # Plot each trace
        axis.plot(times_Vm, trace,
                  color=colorsROI_VmRAW[idx], linewidth=2, label='Base')


def plot_TracesCa(axis, data, time_start=0.0, idx_start=None, time_window=None, time_end = None, idx_end=None):
    # Setup data and time (ms) variables
    data_Ca = []
    traces_count = data.shape[1] - 1    # Number of columns after time column
    times_Ca = (data[:, 0]) * 1000  # seconds to ms
    time_start, time_window = time_start * 1000, time_window * 1000
    # Find index of first value after start time
    for idx, time in enumerate(times_Ca):
        if time < time_start:
            pass
        else:
            idx_start = idx
            break

    if time_window:
        # Find index of first value after end time
        for idx, time in enumerate(times_Ca):
            if time < time_start + time_window:
                pass
            else:
                idx_end = idx
                break
        # Convert possibly strange floats to rounded Decimals
        time_start, time_window = Decimal(time_start).quantize(Decimal('.001'), rounding=ROUND_UP),\
                                  Decimal(time_window).quantize(Decimal('.001'), rounding=ROUND_UP)
        # Slice time array based on indices
        times_Ca = times_Ca[idx_start:idx_end]
        time_end = time_start + time_window
        axis.set_xlim([float(time_start), float(time_end)])

    # Prepare axis for plotting
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    # axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    axis.set_ylim([0, 1.1])
    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # axis.set_xlabel('Time (ms)', fontsize=12)
    # axis.set_ylabel('Norm. Fluor., Vm', fontsize=12)
    # axis.set_title('Normalized Fluorescence ', fontsize=12)

    # ECG Scale: ms and Norm. Fluor. bars forming an L
    ECGScaleTime = [50, 250 / 1000]  # 50 ms, 0.25 Norm. Fluor.
    ECGScaleOrigin = [axis.get_xlim()[1] - 1.5*ECGScaleTime[0], axis.get_ylim()[0] + 0.01]
    # ECGScaleOrigin = [axis.get_xlim()[1] - 20, axis.get_ylim()[0] + 0.3]
    ECGScaleOriginPad = [2, 0.05]
    # Time scale bar
    axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0] + ECGScaleTime[0]],
              [ECGScaleOrigin[1], ECGScaleOrigin[1]],
              "k-", linewidth=1)
    axis.text(ECGScaleOrigin[0], ECGScaleOrigin[1] - ECGScaleOriginPad[1],
              str(ECGScaleTime[0]) + 'ms',
              ha='left', va='top', fontsize=7, fontweight='bold')
    # # Voltage scale bar
    # axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0]],
    #           [ECGScaleOrigin[1], ECGScaleOrigin[1] + ECGScaleTime[1]],"k-", linewidth=1)
    # axis.text(ECGScaleOrigin[0] - ECGScaleOriginPad[0], ECGScaleOrigin[1],
    #           str(ECGScaleTime[1]),
    #           ha='right', va='bottom', fontsize=7, fontweight='bold')

    for idx in range(traces_count):
        trace = data[:, idx + 1]
        if time_window:
            trace = trace[idx_start:idx_end]
        # Normalize each trace
        data_min, data_max = np.nanmin(trace), np.nanmax(trace)
        trace = np.interp(trace, (data_min, data_max), (0, 1))
        trace = 1 - trace   # Need to invert, accidentally exported as voltage??
        data_Ca.append(trace)
        # Plot each trace
        axis.plot(times_Ca, trace,
                  color=colorsROI_CaRAW[idx], linewidth=2, label='Base')


def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


# Build figure
fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 3, width_ratios=[0.3, 0.3, 0.4])  # Overall: ? row, ? columns

# Build heart section
axImage = fig.add_subplot(gs0[0])

# Build Activation Map section
gsActMaps = gs0[1].subgridspec(2, 1)  # 2 rows, 1 columns for Activation Maps
axActMapsVm = fig.add_subplot(gsActMaps[0])
axActMapsCa = fig.add_subplot(gsActMaps[1])

# Build Traces section
gsTraces = gs0[2].subgridspec(2, 1)  # 2 rows, 1 column for Activation Maps
# gsTraces = gs0[2].subgridspec(2, 1, hspace=0.3)  # 2 rows, 1 column for Activation Maps
axTraces = fig.add_subplot(gsTraces[0]), fig.add_subplot(gsTraces[1])

# Build Activation Map section
ActMapTitleX = 0.1
ActMapTitleY = 1


# Import heart image
heart = np.fliplr(np.rot90(plt.imread('data/20190322-pigb/06-300_RH237_0001.tif')))
# ret, heart_thresh = cv2.threshold(heart, 150, np.nan, cv2.THRESH_TOZERO)

# Import Activation Maps
actMapsVm = {300: np.fliplr(np.rot90(np.loadtxt('data/20190322-pigb/ActMaps/ActMap-06-300_RH237.csv',
                                                delimiter=',', skiprows=0)))}
actMapsCa = {300: np.fliplr(np.rot90(np.loadtxt('data/20190322-pigb/ActMaps/ActMap-06-300_Rhod-2.csv',
                                                delimiter=',', skiprows=0)))}
# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for key, value in actMapsVm.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
for key, value in actMapsCa.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
# Create normalization range for all activation maps (round up to nearest 10)
print('Activation Maps max value: ', actMapMax)
jet_norm = colors.Normalize(vmin=0, vmax=round(actMapMax + 5.1, -1))
# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
W, H = actMapsVm[300].shape
rois = [{'y': W - 226, 'x': H - 365, 'r': 10},
        {'y': W - 252, 'x': H - 220, 'r': 10},
        {'y': W - 380, 'x': H - 140, 'r': 10}]
# Create Region of Stimulation (ROS)
ros = {'y': W - 204, 'x': H - 410, 'r': 30}

# Import Traces
# Load signal data, columns: time (s), fluorescence (norm)
# Pair with stim activation time used for activation maps
TraceVm = {300: np.genfromtxt('data/20190322-pigb/ActMaps/Signals-06-300_RH237.csv', delimiter=',')}

TraceCa = {300: np.genfromtxt('data/20190322-pigb/ActMaps/Signals-06-300_Rhod-2.csv', delimiter=',')}


# Plot heart image
plot_heart(axis=axImage, heart_image=heart)
# plot_heart(axis=axImage, heart_image=heart_thresh)
# Plot activation maps
ActMapVm = plot_ActMapVm(axis=axActMapsVm, actMap=actMapsVm[300])
# get the color at pixel 5,5 (use normalization and colormap)
plot_ActMapCa(axis=axActMapsCa, actMap=actMapsCa[300])

# Plot Traces
plot_TracesVm(axis=axTraces[0], data=TraceVm[300],
              time_start=1.15, time_window=0.5)
plot_TracesCa(axis=axTraces[1], data=TraceCa[300],
              time_start=1.15, time_window=0.5)

# Fill rest with example plots
# example_plot(axImage)
# example_plot(fig.add_subplot(gsActMaps[0]))
# example_plot(fig.add_subplot(gsActMaps[1]))
# example_plot(fig.add_subplot(gsActMaps[2]))
# example_plot(fig.add_subplot(gsActMaps[3]))

# example_plot(axTracesSlow)
# example_plot(axTracesFast)


# Show and save figure
fig.show()
fig.savefig('Pig_CV.svg')
