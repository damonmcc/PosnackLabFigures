"""
Plots activation maps and activation curves of murine epicardial tissue.
"""

import math
import random
import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ScientificColourMaps5 as scm

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
    HEIGHT = 200
    WIDTH = 200
    # Allocate space for the Activation Map
    act_map = np.zeros(shape=(WIDTH, HEIGHT))
    # Spatial resolution (cm/px)
    resolution = 0.005  # 4 cm / 200 px

    # Convert conduction velocity from cm/s to px/s
    conduction_v = conduction_v / resolution
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
        act_map[ix, iy] = d / conduction_v
        # Convert time from s to ms
        act_map[ix, iy] = act_map[ix, iy] * 1000
    print('Isotropic act. map generated')
    return act_map


def plot_ActMap(axis, actMap):
    # Setup plot
    height, width, = actMap.shape[1], actMap.shape[0]
    # axis.axis('off')
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_yticks([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_xticklabels([])
    # axis.set_ylabel('Vm', fontsize=18)
    # axis.set_xlim(x_crop)
    # axis.set_ylim(y_crop)

    # Plot Activation Map
    img = axis.imshow(actMap, norm=cmap_norm, cmap=cmap_actMaps)

    return img


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
    # Frame rate (FPS)
    fps = 500
    # frames / ms
    fpms = fps / 1000
    # Number of frames
    frames = int((np.nanmax(actmap) - np.nanmin(actmap)) * (1 / fpms))

    # Bins for histogram calculation
    xbins = np.linspace(start=0, stop=actmap.max(), num=frames)
    hist = np.histogram(actmap, xbins)
    # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    # tissueact = 100 * np.cumsum(hist(tim,xbins))/allpts;

    # Calculate cumulative sum % of activation times
    tissueact = 100 * np.cumsum(hist[0]) / actmap.size
    # A array for plotting the activation curve
    x_axes = np.linspace(start=0, stop=actmap.max(), num=frames - 1)
    return x_axes, tissueact


# Setup the figure
fig = plt.figure()  # _ x _ inch page
# General layout
gs0 = fig.add_gridspec(2, 3)  # Overall: ? row, ? columns

# Setup Activation Map section
# Pacing section (both adult)
gsActMaps_Pacing = gs0[0].subgridspec(1, 2)  # 1 row, 2 columns
axActMap_Pacing_Fast = fig.add_subplot(gsActMaps_Pacing[0])
axActMap_Pacing_Slow = fig.add_subplot(gsActMaps_Pacing[1])
# Age comparison section
gsActMaps_Ages = gs0[3].subgridspec(2, 2)  # 2 rows, 2 columns
axActMap_Ages_PFast = fig.add_subplot(gsActMaps_Ages[0])
axActMap_Ages_PSlow = fig.add_subplot(gsActMaps_Ages[1])
axActMap_Ages_AFast = fig.add_subplot(gsActMaps_Ages[2])
axActMap_Ages_ASlow = fig.add_subplot(gsActMaps_Ages[3])
# axActCurve_Ages.set_title('Activation Constants', fontsize=12)
# Set activation map color map
cmap_actMaps = scm.lajolla

# Setup Activation Curve section
axActCurve_Pacing = fig.add_subplot(gs0[1])
axActCurve_Ages = fig.add_subplot(gs0[4])
axActCurve_Pacing.set_title('Activation Curves', fontsize=12)
axActCurve_Ages.set_xlabel('Time (ms)', fontsize=12)

# Setup Activation Constant section
axActConst_Pacing = fig.add_subplot(gs0[2])
axActConst_Ages = fig.add_subplot(gs0[5])
axActConst_Pacing.set_title('Activation Constants', fontsize=12)

# Generate all activation maps
# Pacing section (both adult)
actMap_Pacing_Fast = generate_ActMap(conduction_v=50)
actMap_Pacing_Slow = generate_ActMap(conduction_v=40)
# Age comparison section
# actMap_Ages_PFast = generate_ActMap(conduction_v=62)
# actMap_Ages_PSlow = generate_ActMap(conduction_v=60)
actMap_Ages_AFast = generate_ActMap(conduction_v=55)
actMap_Ages_ASlow = generate_ActMap(conduction_v=35)

# Import heart image
# heart = np.fliplr(np.rot90(plt.imread('data/20190322-pigb/06-300_RH237_0001.tif')))
# ret, heart_thresh = cv2.threshold(heart, 150, np.nan, cv2.THRESH_TOZERO)

# Import Activation Maps
# actMapIMPORT_Ages_PFast = np.loadtxt('data/20190717-rata/ActMap-01-250_Vm.csv',
#                                      delimiter=',', skiprows=0)
# actMapIMPORT_Ages_PSlow = np.loadtxt('data/20190717-rata/ActMap-03-150_Vm.csv',
#                                      delimiter=',', skiprows=0)
actMap_Ages_PFast = np.loadtxt('data/20190717-rata/ActMap-01-250_Vm.csv',
                               delimiter=',', skiprows=0)
actMap_Ages_PSlow = np.loadtxt('data/20190717-rata/ActMap-03-150_Vm.csv',
                               delimiter=',', skiprows=0)
# Crop imported activation maps
# h1_denom, h2_denom = 2.3, 1.18
# w1_denom, w2_denom = 3, 2
# h, w = actMap_Ages_PFast.shape
# h1, h2 = int(h / h1_denom), int(h / h2_denom)
# w1, w2 = int(w / w1_denom), int(w / w2_denom)
# actMap_Ages_PFast = actMap_Ages_PFast[h1:h2, w1:w2]
# h, w = actMap_Ages_PSlow.shape
# h1, h2 = int(h / h1_denom), int(h / h2_denom)
# w1, w2 = int(w / w1_denom), int(w / w2_denom)
# actMap_Ages_PSlow = actMap_Ages_PSlow[h1:h2, w1:w2]


actMaps = [actMap_Pacing_Slow, actMap_Pacing_Fast,
           actMap_Ages_PSlow, actMap_Ages_PFast, actMap_Ages_ASlow, actMap_Ages_AFast]

# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for actMap in actMaps:
    print(np.nanmax(actMap))
    actMapMax = max(actMapMax, np.nanmax(actMap))
# Round to the nearest X5.0?
actMapMax = round(actMapMax + 2.6, 1)
# Normalize across
# cmap_norm = colors.Normalize(vmin=0, vmax=round(actMapMax + 1.1, -1))
cmap_norm = colors.Normalize(vmin=0, vmax=actMapMax)

# Plot the activation maps
# Pacing section (both adult)
# axActMapsVm.set_title('Activation Maps', fontsize=12)
plot_map(axis=axActMap_Pacing_Fast, actmap=actMap_Pacing_Fast, cmap=cmap_actMaps, norm=cmap_norm)
plot_map(axis=axActMap_Pacing_Slow, actmap=actMap_Pacing_Slow, cmap=cmap_actMaps, norm=cmap_norm)
# Age comparison section
plot_map(axis=axActMap_Ages_PFast, actmap=actMap_Ages_PFast, cmap=cmap_actMaps, norm=cmap_norm)
plot_map(axis=axActMap_Ages_PSlow, actmap=actMap_Ages_PSlow, cmap=cmap_actMaps, norm=cmap_norm)
img_colormap = plot_ActMap(axis=axActMap_Ages_AFast, actMap=actMap_Ages_AFast)
plot_map(axis=axActMap_Ages_ASlow, actmap=actMap_Ages_ASlow, cmap=cmap_actMaps, norm=cmap_norm)

# Add colorbar (lower right of act. map)
ax_ins1 = inset_axes(axActMap_Ages_AFast,
                     width="80%", height="5%",  # % of parent_bbox width
                     loc=8,
                     bbox_to_anchor=(0.75, -0.25, 1, 1), bbox_transform=axActMap_Ages_AFast.transAxes,
                     borderpad=0)
cb1 = plt.colorbar(img_colormap, cax=ax_ins1, orientation="horizontal")
cb1.set_label('Activation Time (ms)', fontsize=6)
cb1.ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
cb1.ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
cb1.ax.tick_params(labelsize=6)

# Generate activation curves
# Pacing section (both adult)
actCurve_Fast_x, actCurve_Fast = generate_actcurve(actMap_Pacing_Fast, actMapMax)
actCurve_Slow_x, actCurve_Slow = generate_actcurve(actMap_Pacing_Slow, actMapMax)
# Age comparison section
actCurve_Ages_PFast_x, actCurve_Ages_PFast = generate_actcurve(actMap_Ages_PFast, actMapMax)
actCurve_Ages_PSlow_x, actCurve_Ages_PSlow = generate_actcurve(actMap_Ages_PSlow, actMapMax)
actCurve_Ages_AFast_x, actCurve_Ages_AFast = generate_actcurve(actMap_Ages_AFast, actMapMax)
actCurve_Ages_ASlow_x, actCurve_Ages_ASlow = generate_actcurve(actMap_Ages_ASlow, actMapMax)

# Plot Pacing activation curves
axActCurve_Pacing.set_ylabel('Tissue Activation (%)', fontsize=10)
axActCurve_Pacing.spines['top'].set_visible(False)
axActCurve_Pacing.spines['right'].set_visible(False)
axActCurve_Pacing.plot(actCurve_Fast_x, actCurve_Fast, 'r')
axActCurve_Pacing.plot(actCurve_Slow_x, actCurve_Slow, 'k')
axActCurve_Pacing.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed')
axActCurve_Pacing.xaxis.set_major_locator(ticker.MultipleLocator(5))
# Plot Ages activation curves
axActCurve_Ages.set_ylabel('Tissue Activation (%)', fontsize=10)
axActCurve_Ages.spines['top'].set_visible(False)
axActCurve_Ages.spines['right'].set_visible(False)
axActCurve_Ages.plot(actCurve_Ages_PFast_x, actCurve_Ages_PFast, 'r-.')
axActCurve_Ages.plot(actCurve_Ages_PSlow_x, actCurve_Ages_PSlow, 'k--', )
axActCurve_Ages.plot(actCurve_Ages_AFast_x, actCurve_Ages_AFast, 'r')
axActCurve_Ages.plot(actCurve_Ages_ASlow_x, actCurve_Ages_ASlow, 'k')
axActCurve_Ages.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed')
axActCurve_Ages.xaxis.set_major_locator(ticker.MultipleLocator(5))

# example_plot(axActCurve_Pacing)
# example_plot(axActCurve_Ages)
example_plot(axActConst_Pacing)
example_plot(axActConst_Ages)

fig.show()
fig.savefig('Developmental_ActivationCurves.svg')
print('Isotropic act. map plotted')
