"""
Plots activation maps and activation curves of murine epicardial tissue, exploratory.
"""

import math
import random
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ScientificColourMaps5 as SCMaps

colors_actcurves = ['r', 'k']
labels_actcurves = ['250ms', '150ms']
# colors_signals = ['0', '0.5']  # Vm: dark, Ca: light
lines_actcurves = ['-', '-']  # Vm: dark, Ca: light
cmap_actMap = SCMaps.lajolla
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
X_CROP = [0, 0]  # to cut from left, right
Y_CROP = [0, 0]  # to cut from bottom, top


def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def generate_ActMap(conduction_v):
    # Dimensions of model data (px)
    HEIGHT = 400
    WIDTH = 200
    # Allocate space for the Activation Map
    act_map = np.zeros(shape=(HEIGHT, WIDTH))
    # Spatial resolution (cm/px)
    resolution = 0.005  # 4 cm / 200 px
    # resolution = 0.0149  # pig video resolution

    # Convert conduction velocity from cm/s to px/s
    conduction_v_px = conduction_v / resolution
    # # Convert dimensions to cm
    # HEIGHT = HEIGHT * resolution
    # WIDTH = WIDTH * resolution

    # Generate an isotropic activation map, radiating from the center
    origin_x, origin_y = WIDTH / 2, HEIGHT / 2
    # Assign an activation time to each pixel
    for ix, iy in np.ndindex(act_map.shape):
        # Compute the distance from the center (cm)
        d = math.sqrt((abs(origin_x - ix) ** 2 + abs((origin_y - iy) ** 2)))
        # Assign the time associated with that distance from the point of activation
        act_map[ix, iy] = d / conduction_v_px
        # Convert time from s to ms
        act_map[ix, iy] = act_map[ix, iy] * 1000
    print('Isotropic act. map generated. CV = ', conduction_v, ' cm/s')
    return act_map


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


def generate_actcurve(actmap, actmap_max):
    # Flatten the activation map data into a 1-D array
    actmap_flat = actmap.ravel()
    # Remove the `nan` values
    actmap_flat = actmap_flat[~np.isnan(actmap_flat)]

    # Frame rate (FPS)
    fps = 500
    # Convert to frames / ms
    fpms = fps / 1000
    # Number of frames represented in this activation map
    frames = int((np.nanmax(actmap_flat) - np.nanmin(actmap_flat)) * (1 / fpms))

    # Create the bins for histogram calculation
    xbins = np.linspace(start=0, stop=actmap_max.max(), num=frames)
    # Calculate the histogram
    hist = np.histogram(actmap_flat, xbins)
    # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    # tissueact = 100 * np.cumsum(hist(tim,xbins))/allpts;

    # Calculate cumulative sum % of tissue activation times
    tissueact = 100 * np.cumsum(hist[0]) / actmap_flat.size
    # An array for the x-axis while plotting the activation curve
    x_axes = np.linspace(start=0, stop=actmap_max.max(), num=frames - 1)
    return x_axes, tissueact


def plot_actcurve(axis, x, y, color, ls='-', label=None, x_labels=False):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.plot(x, y, color=color, linestyle=ls, label=label)
    # axis.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed')
    axis.xaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    if not x_labels:
        axis.set_xticklabels([])


# Setup the figure
# fig = plt.figure()  # _ x _ inch page
fig = plt.figure(figsize=(5, 8))  # _ x _ inch page
# General layout
# To compare 6 studies (3 youngish, 3 oldish) with 2 Pacing Cycle Lengths each (250 ms, 150 ms)
gs0 = fig.add_gridspec(2, 2, width_ratios=[0.5, 0.5])
# Top row for Title-ish text
# gsText = gsfig[0].subgridspec(1, 1)  # Overall: ? row, ? columns
# gsText = fig.add_gridspec(1, 2)  # Overall: ? row, ? columns
# axText1 = fig.add_subplot(gsText[0])
# axText1.axis('off')
# axText1.set_title('Activation Maps', fontsize=fontsize1)
# axText2 = fig.add_subplot(gsText[1])
# axText2.axis('off')
# axText2.set_title('Activation Curves', fontsize=fontsize1)

# Setup Activation Map section
# Youngish section
gsActMaps_Young = gs0[0].subgridspec(3, 2)  # 3 rows, 2 columns
axActMap_Young1_250 = fig.add_subplot(gsActMaps_Young[0])
axActMap_Young1_150 = fig.add_subplot(gsActMaps_Young[1])
axActMap_Young2_250 = fig.add_subplot(gsActMaps_Young[2])
axActMap_Young2_150 = fig.add_subplot(gsActMaps_Young[3])
axActMap_Young3_250 = fig.add_subplot(gsActMaps_Young[4])
axActMap_Young3_150 = fig.add_subplot(gsActMaps_Young[5])
# Oldish section
gsActMaps_Old = gs0[2].subgridspec(3, 2)  # 3 row, 2 columns
axActMap_Old1_250 = fig.add_subplot(gsActMaps_Old[0])
axActMap_Old1_150 = fig.add_subplot(gsActMaps_Old[1])
axActMap_Old2_250 = fig.add_subplot(gsActMaps_Old[2])
axActMap_Old2_150 = fig.add_subplot(gsActMaps_Old[3])
axActMap_Old3_250 = fig.add_subplot(gsActMaps_Old[4])
axActMap_Old3_150 = fig.add_subplot(gsActMaps_Old[5])

axActMap_Young1_250.set_title('PCL 250 ms', fontsize=fontsize2, color=colors_actcurves[0])
axActMap_Young1_150.set_title('PCL 150 ms', fontsize=fontsize2, color=colors_actcurves[1])
# axActMap_Young2_250.text(-50, 1, 'Youngish',
#                          ha='center', va='center', rotation=90, size=fontsize2)
# axActMap_Old2_250.text(-50, 1, 'Oldish',
#                        ha='center', va='center', rotation=90, size=fontsize2)

# Setup Activation Curve section
# Youngish section
gsActCurve_Young = gs0[1].subgridspec(3, 1)  # 3 rows, 1 columns
axActCurve_Young1 = fig.add_subplot(gsActCurve_Young[0])
axActCurve_Young2 = fig.add_subplot(gsActCurve_Young[1])
axActCurve_Young3 = fig.add_subplot(gsActCurve_Young[2])
# Oldish section
gsActCurve_Old = gs0[3].subgridspec(3, 1)  # 3 rows, 1 columns
axActCurve_Old1 = fig.add_subplot(gsActCurve_Old[0])
axActCurve_Old2 = fig.add_subplot(gsActCurve_Old[1])
axActCurve_Old3 = fig.add_subplot(gsActCurve_Old[2])

axActCurve_Old3.set_xlabel('Time (ms)', fontsize=fontsize3)


# Generate all activation maps (Lower PCL => Lower CV)
fast, slow = 95, 75
# Youngish section
# actMap_Young1_250 = generate_ActMap(conduction_v=55)
# actMap_Young1_150 = generate_ActMap(conduction_v=45)
# actMap_Young2_250 = generate_ActMap(conduction_v=55)
# actMap_Young2_150 = generate_ActMap(conduction_v=45)
# actMap_Young3_250 = generate_ActMap(conduction_v=55)
# actMap_Young3_150 = generate_ActMap(conduction_v=45)
# Import maps
axActMap_Young1_250.set_ylabel('P1')
actMap_Young1_250 = np.loadtxt('data/20190717-rata/ActMap-01-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P1
actMap_Young1_150 = np.loadtxt('data/20190717-rata/ActMap-03-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P1

axActMap_Young2_250.set_ylabel('P1')
actMap_Young2_250 = np.loadtxt('data/20190730-rata/ActMap-02-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P1
actMap_Young2_150 = np.loadtxt('data/20190730-rata/ActMap-04-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P1

axActMap_Young3_250.set_ylabel('P2')
actMap_Young3_250 = np.loadtxt('data/20190718-rata/ActMap-02-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P2
actMap_Young3_150 = np.loadtxt('data/20190718-rata/ActMap-04-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P2

# Oldish section
# actMap_Old1_250 = generate_ActMap(conduction_v=55)
# actMap_Old1_150 = generate_ActMap(conduction_v=35)
# actMap_Old2_250 = generate_ActMap(conduction_v=fast)
# actMap_Old2_150 = generate_ActMap(conduction_v=slow)
# actMap_Old3_250 = generate_ActMap(conduction_v=fast)
# actMap_Old3_150 = generate_ActMap(conduction_v=slow)
# Import maps
axActMap_Old1_250.set_ylabel('P9')
actMap_Old1_250 = np.loadtxt('data/20190725-rata/ActMap-02-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P9
actMap_Old1_150 = np.loadtxt('data/20190725-rata/ActMap-04-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P9


axActMap_Old2_250.set_ylabel('P10')
actMap_Old2_250 = np.loadtxt('data/20190404-ratb/ActMap-01-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P9
actMap_Old2_150 = np.loadtxt('data/20190404-ratb/ActMap-11-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P9

axActMap_Old3_250.set_ylabel('P14')
actMap_Old3_250 = np.loadtxt('data/20190404-rata/ActMap-02-250_Vm.csv',
                               delimiter=',', skiprows=0)   # P9
actMap_Old3_150 = np.loadtxt('data/20190404-rata/ActMap-12-150_Vm.csv',
                               delimiter=',', skiprows=0)   # P9


# Import heart image
# heart = np.fliplr(np.rot90(plt.imread('data/20190322-pigb/06-300_RH237_0001.tif')))
# ret, heart_thresh = cv2.threshold(heart, 150, np.nan, cv2.THRESH_TOZERO)
#
# # Import Activation Maps
# actMap_Pacing_Fast = np.loadtxt('data/20190718-rata/ActMap-02-250_Vm.csv',
#                                delimiter=',', skiprows=0)
# actMap_Pacing_Slow = np.loadtxt('data/20190718-rata/ActMap-04-150_Vm.csv',
#                                delimiter=',', skiprows=0)
#
# actMap_Ages_PFast = np.loadtxt('data/20190717-rata/ActMap-01-250_Vm.csv',
#                                delimiter=',', skiprows=0)
# actMap_Ages_PSlow = np.loadtxt('data/20190717-rata/ActMap-03-150_Vm.csv',
#                                delimiter=',', skiprows=0)


actMaps = [actMap_Young1_250, actMap_Young1_150,
           actMap_Young2_250, actMap_Young2_150,
           actMap_Young3_250, actMap_Young3_150,
           actMap_Old1_250, actMap_Old1_150,
           actMap_Old2_250, actMap_Old2_150,
           actMap_Old3_250, actMap_Old3_150]

# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for actMap in actMaps:
    print(np.nanmax(actMap))
    actMapMax = max(actMapMax, np.nanmax(actMap))
print('Activation Maps max value: ', actMapMax)
# Create normalization range for all activation maps (round up to nearest 10)
cmap_norm = colors.Normalize(vmin=0, vmax=round(actMapMax + 5.1, -1))


# Plot the activation maps
# Youngish section
plot_map(axis=axActMap_Young1_250, actmap=actMap_Young1_250, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Young1_150, actmap=actMap_Young1_150, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Young2_250, actmap=actMap_Young2_250, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Young2_150, actmap=actMap_Young2_150, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Young3_250, actmap=actMap_Young3_250, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Young3_150, actmap=actMap_Young3_150, cmap=cmap_actMap, norm=cmap_norm)
# Oldish section
plot_map(axis=axActMap_Old1_250, actmap=actMap_Old1_250, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Old1_150, actmap=actMap_Old1_150, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Old2_250, actmap=actMap_Old2_250, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Old2_150, actmap=actMap_Old2_150, cmap=cmap_actMap, norm=cmap_norm)
plot_map(axis=axActMap_Old3_250, actmap=actMap_Old3_250, cmap=cmap_actMap, norm=cmap_norm)
img_colormap = plot_map(axis=axActMap_Old3_150, actmap=actMap_Old3_150, cmap=cmap_actMap, norm=cmap_norm)

# Add colorbar (lower right of act. map)
ax_ins1 = inset_axes(axActMap_Old3_150,
                     width="80%", height="5%",  # % of parent_bbox width
                     loc=8,
                     bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=axActMap_Old3_150.transAxes,
                     borderpad=0)
cb1 = plt.colorbar(img_colormap, cax=ax_ins1, orientation="horizontal")
cb1.set_label('Activation Time (ms)', fontsize=fontsize4)
cb1.ax.xaxis.set_major_locator(ticker.LinearLocator(3))
cb1.ax.xaxis.set_minor_locator(ticker.LinearLocator(5))
cb1.ax.tick_params(labelsize=fontsize4)

# Generate activation curves
# Youngish section
actCurve_Young1_250_x, actCurve_Young1_250 = generate_actcurve(actMap_Young1_250, actMapMax)
actCurve_Young1_150_x, actCurve_Young1_150 = generate_actcurve(actMap_Young1_150, actMapMax)
actCurve_Young2_250_x, actCurve_Young2_250 = generate_actcurve(actMap_Young2_250, actMapMax)
actCurve_Young2_150_x, actCurve_Young2_150 = generate_actcurve(actMap_Young2_150, actMapMax)
actCurve_Young3_250_x, actCurve_Young3_250 = generate_actcurve(actMap_Young3_250, actMapMax)
actCurve_Young3_150_x, actCurve_Young3_150 = generate_actcurve(actMap_Young3_150, actMapMax)
# Oldish section
actCurve_Old1_250_x, actCurve_Old1_250 = generate_actcurve(actMap_Old1_250, actMapMax)
actCurve_Old1_150_x, actCurve_Old1_150 = generate_actcurve(actMap_Old1_150, actMapMax)
actCurve_Old2_250_x, actCurve_Old2_250 = generate_actcurve(actMap_Old2_250, actMapMax)
actCurve_Old2_150_x, actCurve_Old2_150 = generate_actcurve(actMap_Old2_150, actMapMax)
actCurve_Old3_250_x, actCurve_Old3_250 = generate_actcurve(actMap_Old3_250, actMapMax)
actCurve_Old3_150_x, actCurve_Old3_150 = generate_actcurve(actMap_Old3_150, actMapMax)

# Plot Pacing activation curves
# Red: Fast, Black: Slow
# Youngish section
plot_actcurve(axis=axActCurve_Young1, x=actCurve_Young1_250_x, y=actCurve_Young1_250, color=colors_actcurves[0])
plot_actcurve(axis=axActCurve_Young1, x=actCurve_Young1_150_x, y=actCurve_Young1_150, color=colors_actcurves[1])
axActCurve_Young1.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

plot_actcurve(axis=axActCurve_Young2, x=actCurve_Young2_250_x, y=actCurve_Young2_250, color=colors_actcurves[0])
plot_actcurve(axis=axActCurve_Young2, x=actCurve_Young2_150_x, y=actCurve_Young2_150, color=colors_actcurves[1])
axActCurve_Young2.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

plot_actcurve(axis=axActCurve_Young3, x=actCurve_Young3_250_x, y=actCurve_Young3_250, color=colors_actcurves[0])
plot_actcurve(axis=axActCurve_Young3, x=actCurve_Young3_150_x, y=actCurve_Young3_150, color=colors_actcurves[1])
axActCurve_Young3.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

# Oldish section
plot_actcurve(axis=axActCurve_Old1, x=actCurve_Old1_250_x, y=actCurve_Old1_250, color=colors_actcurves[0])
plot_actcurve(axis=axActCurve_Old1, x=actCurve_Old1_150_x, y=actCurve_Old1_150, color=colors_actcurves[1])
axActCurve_Old1.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

plot_actcurve(axis=axActCurve_Old2, x=actCurve_Old2_250_x, y=actCurve_Old2_250, color=colors_actcurves[0])
plot_actcurve(axis=axActCurve_Old2, x=actCurve_Old2_150_x, y=actCurve_Old2_150, color=colors_actcurves[1])
axActCurve_Old2.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

plot_actcurve(axis=axActCurve_Old3, x=actCurve_Old3_250_x, y=actCurve_Old3_250, color=colors_actcurves[0],
              x_labels=True)
plot_actcurve(axis=axActCurve_Old3, x=actCurve_Old3_150_x, y=actCurve_Old3_150, color=colors_actcurves[1],
              x_labels=True)
axActCurve_Old3.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed', lw=0.5)

# Legend for all traces
legend_lines = [Line2D([0], [0], color=colors_actcurves[0], lw=1),
                Line2D([0], [0], color=colors_actcurves[1], lw=1)]
axActCurve_Young1.legend(legend_lines, labels_actcurves,
                         loc='lower right', bbox_to_anchor=(1, 0.1),
                         ncol=1, prop={'size': fontsize4}, labelspacing=0.5, numpoints=1, frameon=False)

# example_plot(axActCurve_Pacing)
# example_plot(axActCurve_Ages)
# example_plot(axActConst_Pacing)
# example_plot(axActConst_Ages)

fig.show()
fig.savefig('Developmental_ActivationCurves_EXPLORATION.svg')
