import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import ScientificColourMaps5 as SCMaps

MAX_COUNTS_16BIT = 65536
colors_rois = ['b', 'r', 'k']
colors_signals = ['0', '0']  # Vm: dark, Ca: light
lines_signals = ['-', '--']  # Vm: dark, Ca: light
cmap_actMap = SCMaps.lajolla
cmap_durMap = SCMaps.bilbao
# cmap_durMap = SCMaps.oslo.reversed()
# colors_maps = ['#db7a59', '#236C46']  # Act: orange, Dur: green
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
X_CROP = [40, 20]   # to cut from left, right
Y_CROP = [80, 50]   # to cut from bottom, top


def plot_heart(axis, heart_image, scale_text=True, rois=None):
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
    
    scale_text : bool, optional
            If True, include text with the scale bar.
            
        Whether to include the scalebar text
    rois :

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
        # Create ROIs and get colors of their pixels
        for idx, roi in enumerate(rois):
            roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc=None, fill=None,
                                ec=colors_rois[idx], lw=1)
            axis.add_artist(roi_circle)

    # patch = Ellipse((width/2, height/2), width=width, height=height, transform=axis.transData)
    # img.set_clip_path(patch)
    # Scale Bar
    scale_px_cm = 1 / 0.0149
    heart_scale = [scale_px_cm, scale_px_cm]  # x, y (pixels/cm)
    if scale_text:
        heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], '1 cm',
                                          loc=4, pad=0.2, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=7, weight='bold'))
    else:
        # Scale bar, no tex
        heart_scale_bar = AnchoredSizeBar(axis.transData, heart_scale[0], ' ',
                                          loc=4, pad=0.4, color='w', frameon=False,
                                          fontproperties=fm.FontProperties(size=2, weight='bold'))
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
            # convert y-axis from counts to percentage range of max counts
            data_y = data_y_counts / MAX_COUNTS_16BIT * 100
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        else:
            # Shift y-axis counts to start at zero
            data_y = data_y_counts - counts_min

        if filter_lp:
            print('* Filtering data: Low Pass')
            dt = 1 / 408
            freq = 100

            fs = 1 / dt
            wn = (freq / (fs / 2))
            [b, a] = sig.butter(5, wn)
            data_y = sig.filtfilt(b, a, data_y)
            print('* Data Filtered')

        if norm:
            # # Normalize each trace
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


def plot_trace_overlay(axis, trace_vm, trace_ca):
    # Normalize, Filter, and Plot a Vm and a Ca trace on the same plot
    idx_end = len(trace_vm)
    idx_span = 280    # 2 transients
    # idx_span = 120
    # idx_span = idx_end-512
    # Zoom

    # idx_end = len(trace_vm) - 120
    # idx_span = 128

    trace_vm_x, trace_vm_y = plot_trace(axis, trace_vm, imagej=True, fps=408, x_span=idx_span, x_end=idx_end,
                                        norm=True, invert=True, filter_lp=True,
                                        )
    trace_ca_x, trace_ca_y = plot_trace(axis, trace_ca, imagej=True, fps=408, x_span=idx_span, x_end=idx_end,
                                        norm=True, filter_lp=True,
                                        )
    # Set the linewidth of each trace
    for idx, line in enumerate(axis.get_lines()):
        line.set_linewidth(0.5)
        line.set_color(colors_signals[idx])
        line.set_linestyle(lines_signals[idx])

    # axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator((int(100/4))))

    axis.xaxis.set_major_locator(ticker.NullLocator())
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.yaxis.set_major_locator(ticker.NullLocator())
    axis.yaxis.set_minor_locator(ticker.NullLocator())
    # axis.set_yticks([])
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    # ECG Scale: ms and Norm. Fluor. bars forming an L
    ecg_scale_time = [50, 250 / 1000]  # 100 ms, 0.25 Norm. Fluor.
    ecg_scale_origin = [axis.get_xlim()[1] - ecg_scale_time[0], axis.get_ylim()[0] + 0.01]
    # ecg_scale_origin = [axis.get_xlim()[1] - 1.5 * ecg_scale_time[0], axis.get_ylim()[0] + 0.01]
    # ecg_scale_origin = [axis.get_xlim()[1] - 20, axis.get_ylim()[0] + 0.3]
    ecg_scale_origin_pad = [2, 0.05]
    # Time scale bar
    axis.plot([ecg_scale_origin[0], ecg_scale_origin[0] + ecg_scale_time[0]],
              [ecg_scale_origin[1], ecg_scale_origin[1]],
              "k-", linewidth=1)
    axis.text(ecg_scale_origin[0], ecg_scale_origin[1] - ecg_scale_origin_pad[1],
              str(ecg_scale_time[0]) + 'ms',
              ha='left', va='top', fontsize=7, fontweight='semibold')
    # Get the max of both traces
    traces_max = max(np.nanmax(trace_vm_y), np.nanmax(trace_ca_y))
    # Lower y-axis lower limit to account for Scale bar
    axis.set_ylim([-0.1, traces_max])

    # Legend for all traces
    legend_lines = [Line2D([0], [0], color=colors_signals[0], ls=lines_signals[0], lw=1),
                    Line2D([0], [0], color=colors_signals[1], ls=lines_signals[1], lw=1)]
    axis.legend(legend_lines, ['Vm', 'Ca'],
                loc='upper right', bbox_to_anchor=(1.05, 1),
                ncol=1, prop={'size': 6}, labelspacing=1, numpoints=1, frameon=False)


def plot_map(axis, actmap, cmap, norm):
    # Setup plot
    height, width, = actmap.shape[0], actmap.shape[1]  # X, Y flipped due to rotation
    x_crop, y_crop = [X_CROP[0], width - X_CROP[1]], [height - Y_CROP[0], Y_CROP[1]]

    # axis.axis('off')
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_xticklabels([])

    axis.set_xlim(x_crop)
    axis.set_ylim(y_crop)

    # Plot Activation Map
    img = axis.imshow(actmap, norm=norm, cmap=cmap)

    return img


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
gsfig = fig.add_gridspec(2, 1, height_ratios=[0.6, 0.4])  # Overall: ? row, ? columns

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
heart_Vm = np.rot90(plt.imread('data/20190322-pigb/01-350_Vm_0001.tif'))
heart_Ca = np.rot90(plt.imread('data/20190322-pigb/01-350_Ca_0001.tif'))
# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
# RV, LV
H, W = heart_Vm.shape
Rois_Vm = [{'y': H - 398, 'x': 206, 'r': 15},
           {'y': H - 198, 'x': 324, 'r': 15}]
Rois_Ca = Rois_Vm

# Plot heart images
axImage_Vm.set_title('Dual-Images', fontsize=fontsize2, weight='semibold')
axImage_label_x = -0.15
axImage_label_y = 0.5
axImages_label_font = fm.FontProperties(size=fontsize2, weight='semibold')

plot_heart(axis=axImage_Vm, heart_image=heart_Vm, rois=Rois_Vm, scale_text=False)
axImage_Vm.text(axImage_label_x, axImage_label_y, 'Vm', transform=axImage_Vm.transAxes,
                rotation=90, ha='center', va='center', fontproperties=axImages_label_font)
# axImage_Ca.set_title('Ca', fontsize=14)
plot_heart(axis=axImage_Ca, heart_image=heart_Ca, rois=Rois_Ca, scale_text=False)
axImage_Ca.text(axImage_label_x, axImage_label_y, 'Ca', transform=axImage_Ca.transAxes,
                rotation=90, ha='center', va='center', fontproperties=axImages_label_font)


# Import Traces
# Load signal data, columns: index, fluorescence (counts)
Trace_Vm_RV = {1: np.genfromtxt('data/20190322-pigb/01-350_Vm_15x15-398x206.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/01-350_Vm_30x30-398x206.csv', delimiter=',')}
Trace_Vm_LV = {1: np.genfromtxt('data/20190322-pigb/01-350_Vm_15x15-198x324.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/01-350_Vm_30x30-198x324.csv', delimiter=',')}

Trace_Ca_RV = {1: np.genfromtxt('data/20190322-pigb/01-350_Ca_15x15-398x206.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/01-350_Ca_30x30-398x206.csv', delimiter=',')}
Trace_Ca_LV = {1: np.genfromtxt('data/20190322-pigb/01-350_Ca_15x15-198x324.csv', delimiter=','),
               5: np.genfromtxt('data/20190322-pigb/01-350_Ca_30x30-198x324.csv', delimiter=',')}
# Plot paced traces
axTraces_Vm_RV.set_title('15x15 Pixel', fontsize=fontsize2, weight='semibold')
axTraces_Vm_RV_5x5.set_title('30x30 Pixel', fontsize=fontsize2, weight='semibold')
axTraces_label_x = -0.2
axTraces_label_y = 0.5
axTraces_label_font = fm.FontProperties(size=fontsize3, weight='semibold')

axTraces_x_span = 850

axTraces_Vm_RV.text(axTraces_label_x, axTraces_label_y, 'RV', transform=axTraces_Vm_RV.transAxes,
                    rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTraces_Vm_RV, Trace_Vm_RV[1], imagej=True, fps=408, x_span=axTraces_x_span,
           color='b', x_ticks=False)
plot_trace(axTraces_Vm_RV_5x5, Trace_Vm_RV[5], imagej=True, fps=408, x_span=axTraces_x_span,
           color='b', x_ticks=False)
axTraces_Vm_LV.text(axTraces_label_x, axTraces_label_y, 'LV', transform=axTraces_Vm_LV.transAxes,
                    rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTraces_Vm_LV, Trace_Vm_LV[1], imagej=True, fps=408, color='r', x_span=axTraces_x_span)
plot_trace(axTraces_Vm_LV_5x5, Trace_Vm_LV[5], imagej=True, fps=408, color='r', x_span=axTraces_x_span)

axTraces_Ca_RV.text(axTraces_label_x, axTraces_label_y, 'RV', transform=axTraces_Ca_RV.transAxes,
                    rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTraces_Ca_RV, Trace_Ca_RV[1], imagej=True, fps=408, x_span=axTraces_x_span,
           color='b', x_ticks=False)
plot_trace(axTraces_Ca_RV_5x5, Trace_Ca_RV[5], imagej=True, fps=408, x_span=axTraces_x_span,
           color='b', x_ticks=False)
axTraces_Ca_LV.text(axTraces_label_x, axTraces_label_y, 'LV', transform=axTraces_Ca_LV.transAxes,
                    rotation=90, ha='center', va='center', fontproperties=axTraces_label_font)
plot_trace(axTraces_Ca_LV, Trace_Ca_LV[1], imagej=True, fps=408, color='r', x_span=axTraces_x_span)
plot_trace(axTraces_Ca_LV_5x5, Trace_Ca_LV[5], imagej=True, fps=408, color='r', x_span=axTraces_x_span)


# Plot Analysis Section
# Plot trace overlay
axTracesOverlay.set_title('Vm/Ca Dual Signals', fontsize=fontsize2, weight='semibold')
plot_trace_overlay(axTracesOverlay, trace_vm=Trace_Vm_LV[5], trace_ca=Trace_Ca_LV[5])
# Import activation maps
# actMapVm = np.rot90(np.loadtxt('data/20190322-pigb/OLD_06-300/ActMaps/ActMap-06-300_Vm.csv',
#                                          delimiter=',', skiprows=0))
actMapVm = np.rot90(np.loadtxt('data/20190322-pigb/ActMaps/ActMap-01-350_Vm.csv',
                               delimiter=',', skiprows=0))
actMapCa = np.rot90(np.loadtxt('data/20190322-pigb/ActMaps/ActMap-01-350_Ca.csv',
                               delimiter=',', skiprows=0))
# Import duration maps
durMapVm = np.rot90(np.loadtxt('data/20190322-pigb/APDMaps/APD-01-350_Vm.csv',
                               delimiter=',', skiprows=0))
durMapCa = np.rot90(np.loadtxt('data/20190322-pigb/APDMaps/APD-01-350_Ca.csv',
                               delimiter=',', skiprows=0))
# Crop edges of duration maps, replace with NaNs
# durMapVm[0:150, :] = np.nan
# durMapCa[0:150, :] = np.nan
# Mask  low-high durations, replace with NaNs
durMap_min = 100    # ms
durMap_max = 300    # ms
durMapVm = np.ma.masked_less(durMapVm, durMap_min)
durMapVm = np.ma.masked_greater(durMapVm, durMap_max)
durMapVm.filled(np.nan)
durMapCa = np.ma.masked_less(durMapCa, durMap_min)
durMapCa = np.ma.masked_greater(durMapCa, durMap_max)
durMapCa.filled(np.nan)

# Determine max value across all activation maps
print('Activation Map max values:')
actMapMax = max(np.nanmax(actMapVm), np.nanmax(actMapCa))
# Create normalization range for all activation maps (round up to nearest 10)
print('Activation Maps max value: ', actMapMax)
cmapNorm_actMaps = colors.Normalize(vmin=0, vmax=round(actMapMax + 5.1, -1))

# Determine max value across all duration maps
print('Duration Map max values:')
durMapMax = max(np.nanmax(durMapVm), np.nanmax(durMapCa))
# durMapMax = round(durMapMax + 5.1, -1)
# Create normalization range for all duration maps (round up to nearest 10)
print('Duration Maps max value: ', durMapMax)
cmapNorm_durMaps = colors.Normalize(vmin=durMap_min, vmax=round(durMapMax + 5.1, -1))

# Use Vm and Ca maxs for common cmaps
vmMapMax = max(np.nanmax(actMapVm), np.nanmax(durMapVm))
caMapMax = max(np.nanmax(actMapVm), np.nanmax(durMapVm))
print('Vm Maps max value: ', vmMapMax)
cmapNorm_vmMaps = colors.Normalize(vmin=0, vmax=round(vmMapMax + 5.1, -1))
cmapNorm_caMaps = colors.Normalize(vmin=0, vmax=round(caMapMax + 5.1, -1))

# Plot maps
# Activation Maps
axMap_ActVm.set_title('Activation', fontsize=fontsize3, weight='semibold')
axMaps_label_x = -0.15
axMaps_label_y = 0.5
axMaps_label_size = fontsize2
axMaps_label_font = fm.FontProperties(size=fontsize3, weight='semibold')

plot_heart(axis=axMap_ActVm, heart_image=heart_Vm, scale_text=False)
img_actMapVm = plot_map(axis=axMap_ActVm, actmap=actMapVm, cmap=cmap_actMap, norm=cmapNorm_actMaps)
axMap_ActVm.text(axMaps_label_x, axMaps_label_y, 'Vm', transform=axMap_ActVm.transAxes,
                 rotation=90, ha='center', va='center', fontproperties=axMaps_label_font)
plot_heart(axis=axMap_ActCa, heart_image=heart_Ca, scale_text=False)
img_actMapCa = plot_map(axis=axMap_ActCa, actmap=actMapCa, cmap=cmap_actMap, norm=cmapNorm_actMaps)
axMap_ActCa.set_ylabel('Ca', fontsize=fontsize3)
axMap_ActCa.text(axMaps_label_x, axMaps_label_y, 'Ca', transform=axMap_ActCa.transAxes,
                 rotation=90, ha='center', va='center', fontproperties=axMaps_label_font)
# Add colorbar (activation maps)
ax_cmap_act = inset_axes(axMap_ActCa,
                         width="8%", height="80%",  # % of parent_bbox width, height
                         loc=10, bbox_to_anchor=(0.75, 0.5, 1, 1), bbox_transform=axMap_ActCa.transAxes, borderpad=0)
cb_act = plt.colorbar(img_actMapCa, cax=ax_cmap_act, orientation="vertical")
cb_act.ax.set_xlabel('ms', fontsize=fontsize4)
# cb1.set_label('ms', fontsize=fontsize4)
cb_act.ax.yaxis.set_major_locator(ticker.LinearLocator(3))
cb_act.ax.yaxis.set_minor_locator(ticker.LinearLocator(5))
cb_act.ax.tick_params(labelsize=fontsize4)

# Duration Maps
axMap_APD.set_title('Repolarization (80%)', fontsize=fontsize3, weight='semibold')
plot_heart(axis=axMap_APD, heart_image=heart_Vm, scale_text=False)
img_durMapVm = plot_map(axis=axMap_APD, actmap=durMapVm, cmap=cmap_durMap, norm=cmapNorm_durMaps)
plot_heart(axis=axMap_CAD, heart_image=heart_Vm, scale_text=False)
img_durMapCa = plot_map(axis=axMap_CAD, actmap=durMapCa, cmap=cmap_durMap, norm=cmapNorm_durMaps)

# Add colorbar (duration maps)
ax_cmap_dur = inset_axes(axMap_CAD,
                         width="8%", height="80%",  # % of parent_bbox width, height
                         loc=10, bbox_to_anchor=(0.75, 0.5, 1, 1), bbox_transform=axMap_CAD.transAxes, borderpad=0)
cb_dur = plt.colorbar(img_durMapCa, cax=ax_cmap_dur, orientation="vertical")
cb_dur.ax.set_xlabel('ms', fontsize=fontsize4)
# cb_dur.ax.set_ylim([durMap_min, int(durMapMax)])
cb_dur.ax.yaxis.set_major_locator(ticker.LinearLocator(3))
cb_dur.ax.yaxis.set_minor_locator(ticker.LinearLocator(5))
cb_dur.ax.tick_params(labelsize=fontsize4)

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

# example_plot(axMap_ActVm)
# example_plot(axMap_ActCa)
# example_plot(axMap_APD)
# example_plot(axMap_CAD)

example_plot(axMapStats)


# Show and save figure
fig.show()
fig.savefig('JoVE_Fig3-Paced.svg')
