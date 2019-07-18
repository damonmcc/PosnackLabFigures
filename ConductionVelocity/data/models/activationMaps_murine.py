"""
Generates model activation maps and activation curves of murine epicardial tissue.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ScientificColourMaps5 as scm


def gauss_2d(mu, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu, sigma)
    return x, y


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
    resolution = 0.005   # 4 cm / 200 px

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


def generate_ActCurve(actMap):
    # Frame rate (FPS)
    fps = 500
    # frames / ms
    fpms = fps / 1000
    # Number of frames
    frames = int((np.nanmax(actMap) - np.nanmin(actMap)) * (1 / fpms))

    # Bins for histogram calculation
    xbins = np.linspace(start=0, stop=actMap.max(), num=frames)
    hist = np.histogram(actMap, xbins)
    # a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    # tissueact = 100 * np.cumsum(hist(tim,xbins))/allpts;

    tissueact = 100 * np.cumsum(hist[0]) / actMap.size

    # A array for plotting the activation curve
    x_axes = np.linspace(start=0, stop=actMap.max(), num=frames-1)
    return x_axes, tissueact


# Setup the figure
fig = plt.figure()  # _ x _ inch page
# fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 3, width_ratios=[0.3, 0.3, 0.4])  # Overall: ? row, ? columns
# Set activation map color map
cmap_actMaps = scm.lajolla

# Generate all activation maps
actMap_Fast = generate_ActMap(conduction_v=70)
actMap_Slow = generate_ActMap(conduction_v=50)
actMaps = [actMap_Slow, actMap_Fast]

# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for map in actMaps:
    print(np.nanmax(map))
    actMapMax = max(actMapMax, np.nanmax(map))
# Normalize across
# cmap_norm = colors.Normalize(vmin=0, vmax=round(actMapMax + 1.1, -1))
cmap_norm = colors.Normalize(vmin=0, vmax=actMapMax)

# Plot the activation maps
# "Fast" conduction velocity actMap
axis_actMap_Fast = fig.add_subplot(gs0[0])
font = {'color':  'r',
        'size': 10,
        }
axis_actMap_Fast.set_title('"Fast" CV: 70 cm/s', fontdict=font)
axis_actMap_Fast.spines['right'].set_visible(False)
axis_actMap_Fast.spines['left'].set_visible(False)
axis_actMap_Fast.spines['top'].set_visible(False)
axis_actMap_Fast.spines['bottom'].set_visible(False)
# axis_actMap_Fast.set_yticks([])
# axis_actMap_Fast.set_yticklabels([])
# axis_actMap_Fast.set_xticks([])
# axis_actMap_Fast.set_xticklabels([])
img_actMap_Fast = axis_actMap_Fast.imshow(actMap_Fast, norm=cmap_norm, cmap=cmap_actMaps)
# "Slow" conduction velocity actMap
axis_actMap_Slow = fig.add_subplot(gs0[1])
font = {'color':  'k',
        'size': 10,
        }
axis_actMap_Slow.set_title('"Slow" CV: 50 cm/s', fontdict=font)
axis_actMap_Slow.spines['right'].set_visible(False)
axis_actMap_Slow.spines['left'].set_visible(False)
axis_actMap_Slow.spines['top'].set_visible(False)
axis_actMap_Slow.spines['bottom'].set_visible(False)
axis_actMap_Slow.set_yticks([])
axis_actMap_Slow.set_yticklabels([])
axis_actMap_Slow.set_xticks([])
axis_actMap_Slow.set_xticklabels([])
img_actMap_Slow = axis_actMap_Slow.imshow(actMap_Slow, norm=cmap_norm,  cmap=cmap_actMaps)

# Add colorbar (lower center of act. map)
ax_ins1 = inset_axes(axis_actMap_Slow, width="50%",  # width: 5% of parent_bbox width
                     height="3%",  # height : 80%
                     loc=8,
                     bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=axis_actMap_Slow.transAxes,
                     borderpad=0)
cb1 = plt.colorbar(img_actMap_Slow, cax=ax_ins1, orientation="horizontal")
cb1.set_label('Activation Time (ms)', fontsize=8)

# Generate activation curves
actCurve_Fast_x, actCurve_Fast = generate_ActCurve(actMap_Fast)
actCurve_Slow_x, actCurve_Slow = generate_ActCurve(actMap_Slow)
# Plot activation curves
axis_actCurve = fig.add_subplot(gs0[2])
axis_actCurve.set_ylabel('Tissue Activation (%)', fontsize=10)
axis_actCurve.spines['top'].set_visible(False)
axis_actCurve.spines['right'].set_visible(False)
axis_actCurve.plot(actCurve_Fast_x, actCurve_Fast, 'k')
axis_actCurve.plot(actCurve_Slow_x, actCurve_Slow, 'r')
axis_actCurve.hlines(50, xmin=0, xmax=actMapMax, linestyles='dashed')

fig.show()
fig.savefig('activationMaps_murine.svg')
print('Isotropic act. map plotted')
