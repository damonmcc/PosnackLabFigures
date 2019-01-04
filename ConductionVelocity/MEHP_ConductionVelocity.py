import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from scipy import stats
from matplotlib import ticker

actMapsCTRL = [np.loadtxt('data/ActMap-20180619-rata-33.csv', delimiter=',', skiprows=0),
               np.loadtxt('data/ActMap-20180619-rata-28.csv', delimiter=',', skiprows=0),
               np.loadtxt('data/ActMap-20180619-rata-23.csv', delimiter=',', skiprows=0)]

actMapsMEHP = [np.loadtxt('data/ActMap-20180619-ratb-11.csv', delimiter=',', skiprows=0),
               np.loadtxt('data/ActMap-20180619-ratb-6.csv', delimiter=',', skiprows=0),
               np.loadtxt('data/ActMap-20180619-ratb-1.csv', delimiter=',', skiprows=0)]

def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapCTRL(axis, actMap):
    axis.imshow(actMap, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapMEHP(axis, actMap):
    axis.imshow(actMap, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


fig = plt.figure(figsize=(8, 8))  # half 11 x 7 inch page
gs0 = fig.add_gridspec(2, 2, height_ratios=[0.8, 0.2])  # Overall: ? row, ? columns
gsActMaps = gs0[0].subgridspec(3, 2, hspace=0.3)  # 3 rows, 2 columns for Activation Maps
gsConDealys = gs0[1].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Conduction Delays

axActMap140CTRL = fig.add_subplot(gsActMaps[0])
axActMap140MEHP = fig.add_subplot(gsActMaps[1])
axActMap190CTRL = fig.add_subplot(gsActMaps[2])
axActMap190MEHP = fig.add_subplot(gsActMaps[3])
axActMap240CTRL = fig.add_subplot(gsActMaps[4])
axActMap240MEHP = fig.add_subplot(gsActMaps[5])
axCVMAX = fig.add_subplot(gs0[2])
axActTimes = fig.add_subplot(gs0[3])

example_ActMapCTRL(axActMap140CTRL, actMapsCTRL[0])
example_ActMapMEHP(axActMap140MEHP, actMapsMEHP[0])
example_ActMapCTRL(axActMap190CTRL, actMapsCTRL[1])
example_ActMapMEHP(axActMap190MEHP, actMapsMEHP[1])
example_ActMapCTRL(axActMap240CTRL, actMapsCTRL[2])
example_ActMapMEHP(axActMap240MEHP, actMapsMEHP[2])

for n in range(4):
    ax = fig.add_subplot(gsConDealys[n])
    example_plot(ax)

example_plot(axCVMAX)
example_plot(axActTimes)

fig.show()
fig.savefig('MEHP_CV.svg')
