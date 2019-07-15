"""
Generates model activation maps of murine epicardial tissue.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
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


# Dimensions of model data (px)
HEIGHT = 50
WIDTH = 50

# Allocate space for the Activation Map
actMap = np.zeros(shape=(WIDTH, HEIGHT))

# Spatial resolution (cm/px)
resolution = 0.01

# Number of frames ()
frame_num = 10

# Sampling frequency (Hz)
sample_f = 500

# Given conduction velocity (cm/s)
CV = 50

# Generate an isotropic activation map, radiating from the center
origin_x, origin_y = WIDTH / 2, HEIGHT / 2
# Assign an activation time to each pixel
for ix, iy in np.ndindex(actMap.shape):
    # Compute the distance from the center
    d = math.sqrt((abs(origin_x - ix) ** 2 + abs((origin_y - iy) ** 2)))
    # Assign the time associated with that distance from the point of activation
    actMap[ix, iy] = d / CV
    print(actMap[ix, iy])

print('Isotropic act. map generated')

# Plot the activation map
fig = plt.figure()  # _ x _ inch page
# fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.4])  # Overall: ? row, ? columns

axis_Iso = fig.add_subplot(gs0[0])
axis_Iso.spines['right'].set_visible(False)
axis_Iso.spines['left'].set_visible(False)
axis_Iso.spines['top'].set_visible(False)
axis_Iso.spines['bottom'].set_visible(False)
axis_Iso.set_yticks([])
axis_Iso.set_yticklabels([])
axis_Iso.set_xticks([])
axis_Iso.set_xticklabels([])
cmap_actMaps = scm.lajolla

actMap_Iso = axis_Iso.imshow(actMap, cmap=cmap_actMaps)
# Add colorbar (lower center of act. map)
ax_ins1 = inset_axes(axis_Iso, width="50%",  # width: 5% of parent_bbox width
                     height="3%",  # height : 80%
                     loc=8,
                     bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=axis_Iso.transAxes,
                     borderpad=0)
cb1 = plt.colorbar(actMap_Iso, cax=ax_ins1, orientation="horizontal")

#
# Generate an activation curve
# example_plot(fig.add_subplot(gs0[1]))
xbins = np.arange(0, actMap.max(), 0.01)
hist = np.histogram(actMap, xbins)
# a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
# tissueact = 100 * np.cumsum(hist(tim,xbins))/allpts;

tissueact = 100 * np.cumsum(hist[0])/actMap.size

axis_actCurve = fig.add_subplot(gs0[1])
axis_actCurve.plot(tissueact)

fig.show()
print('Isotropic act. map plotted')
