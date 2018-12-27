import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from scipy import stats
from matplotlib import ticker


def example_plot(axis):
    axis.plot([1, 2])
    axis.set_xlabel('x-label', fontsize=12)
    axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Title', fontsize=14)


fig = plt.figure(figsize=(8, 8))  # half 11 x 7 inch page
gs0 = fig.add_gridspec(2, 2)  # Overall: ? row, ? columns
gsActMaps = gs0[0].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Activation Maps
gsConDealys = gs0[1].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Conduction Delays

axCVMAX = fig.add_subplot(gs0[2])
axActTimes = fig.add_subplot(gs0[3])


for n in range(4):
    ax = fig.add_subplot(gsActMaps[n])
    example_plot(ax)
for n in range(4):
    ax = fig.add_subplot(gsConDealys[n])
    example_plot(ax)

example_plot(axCVMAX)
example_plot(axActTimes)

fig.show()
fig.savefig('MEHP_EP.svg')
