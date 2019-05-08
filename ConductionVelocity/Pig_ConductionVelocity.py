import numpy as np
from decimal import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.patches import Circle, Wedge
import colorsys
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set Colors
# roi_colors = ['0.45', '0.65', '0.85']
roi_colors = ['0.65', '0.65', '0.65']

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
    axis.axis('off')
    axis.imshow(heart_image, cmap='bone')

def plot_ActMapVm(axis, actMap):
    axis.axis('off')
    for idx, roi in enumerate(rois):
        roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none',
                            ec=roi_colors[idx], lw=2)
        axis.add_artist(roi_circle)
    # ros_wedge = Wedge((ros['x'], ros['y']), ros['r'], 225, 45, fc='none', ec='k', lw=2)
    # axis.add_artist(ros_wedge)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
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
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Vm', fontsize=12)
    return img


def plot_ActMapCa(axis, actMap):
    axis.axis('off')
    for idx, roi in enumerate(rois):
        roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none',
                            ec=roi_colors[idx], lw=2)
        axis.add_artist(roi_circle)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
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
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Ca', fontsize=12)
    return img


def plot_TracesVm(axis, data, time_start=0.0, idx_start=None, time_window=None, idx_end=None):
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

    # Prepare axis for plotting
    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    axis.set_ylabel('Norm. Fluor., Vm', fontsize=12)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_ylim([0, 1.1])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)

    for idx in range(traces_count):
        trace = data[:, idx + 1]
        if time_window:
            time_end = time_start + time_window
            trace = trace[idx_start:idx_end]
            axis.set_xlim([float(time_start), float(time_end)])
        # Normalize each trace
        data_min, data_max = np.nanmin(trace), np.nanmax(trace)
        trace = np.interp(trace, (data_min, data_max), (0, 1))
        data_Vm.append(trace)
        # Plot each trace
        axis.plot(times_Vm, trace,
                  color=colorsROI_VmRAW[idx], linewidth=2, label='Base')


def plot_TracesCa(axis, data, time_start=0.0, idx_start=None, time_window=None, idx_end=None):
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

    # Prepare axis for plotting
    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(20))
    axis.set_ylabel('Norm. Fluor., Ca', fontsize=12)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_ylim([0, 1.1])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)

    for idx in range(traces_count):
        trace = data[:, idx + 1]
        if time_window:
            time_end = time_start + time_window
            trace = trace[idx_start:idx_end]
            axis.set_xlim([float(time_start), float(time_end)])
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
# gs0 = fig.add_gridspec(1, 2)  # Overall: ? row, ? columns

# Build heart section
# gsHeart = gs0[0].subgridspec(2, 1, hspace=0.3)
axImage = fig.add_subplot(gs0[0])

# Build Activation Map section
gsActMaps = gs0[1].subgridspec(2, 1, hspace=0.3)  # 2 rows, 2 columns for Activation Maps
axActMapsVm = fig.add_subplot(gsActMaps[0])
axActMapsCa = fig.add_subplot(gsActMaps[1])
# axActMapsVm[0].set_ylabel('PCL 250 ms')
# axActMapsVm[1].set_ylabel('PCL 350 ms')

# Build Traces section
gsTraces = gs0[2].subgridspec(2, 1, hspace=0.3)  # 2 rows, 1 column for Activation Maps
axTraces = fig.add_subplot(gsTraces[0]), fig.add_subplot(gsTraces[1])

# Build Activation Map section
ActMapTitleX = 0.1
ActMapTitleY = 1


# Import heart image
heart = np.rot90(plt.imread('data/20190322-pigb/06-300_RH237_0001.tif'))

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
# Create normalization range for all activation maps
print('Activation Maps max value: ', actMapMax)
jet_norm = colors.Normalize(vmin=0, vmax=actMapMax)
# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
W, H = actMapsVm[300].shape
rois = [{'y': W - 226, 'x': H - 365, 'r': 15},
        {'y': W - 252, 'x': H - 220, 'r': 15},
        {'y': W - 380, 'x': H - 140, 'r': 15}]
# Create Region of Stimulation (ROS)
ros = {'y': W - 140, 'x': H - 360, 'r': 20}

# Import Traces
# Load signal data, columns: time (s), fluorescence (norm)
# Pair with stim activation time used for activation maps
TraceVm = {300: np.genfromtxt('data/20190322-pigb/ActMaps/Signals-06-300_RH237.csv', delimiter=',')}

TraceCa = {300: np.genfromtxt('data/20190322-pigb/ActMaps/Signals-06-300_Rhod-2.csv', delimiter=',')}


# Plot heart image
plot_heart(axis=axImage, heart_image=heart)
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
