import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from scipy import stats
from matplotlib import ticker

colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'grey'
colorMEHP = 'black'

actMapsCTRL = {140: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-33.csv',
                                                  delimiter=',', skiprows=0))),
               190: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-28.csv',
                                                  delimiter=',', skiprows=0))),
               240: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-23.csv',
                                                  delimiter=',', skiprows=0)))}
actMapsMEHPbase = {140: np.loadtxt('data/ActMap-20180522-rata-15.csv',
                                   delimiter=',', skiprows=0),
                   190: np.loadtxt('data/ActMap-20180522-rata-10.csv',
                                   delimiter=',', skiprows=0),
                   240: np.loadtxt('data/ActMap-20180522-rata-05.csv',
                                   delimiter=',', skiprows=0)}
actMapsMEHP = {140: np.loadtxt('data/ActMap-20180522-rata-38.csv',
                               delimiter=',', skiprows=0),
               190: np.loadtxt('data/ActMap-20180522-rata-33.csv',
                               delimiter=',', skiprows=0),
               240: np.loadtxt('data/ActMap-20180522-rata-28.csv',
                               delimiter=',', skiprows=0)}

VmMEHP_base140 = np.genfromtxt('data/Signals/Signal-20180522-rata-15x55y140.csv', delimiter=',')
VmtMEHP_base140 = 1000 * np.genfromtxt('data/Signals/Signal-20180522-rata-15x55y140times.csv', delimiter=',')
VmMEHP_post140 = np.genfromtxt('data/Signals/Signal-20180522-rata-38x71y86.csv', delimiter=',')
VmtMEHP_post140 = 1000 * np.genfromtxt('data/Signals/Signal-20180522-rata-38x71y86times.csv', delimiter=',')


def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapCTRL(axis, actMap):
    axis.axis('off')
    axis.imshow(actMap, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapMEHP(axis, actMap):
    axis.axis('off')
    axis.imshow(actMap, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ConductionDelay(axis, dataBase, dataPost, times):
    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xlim([min(times), 560])
    axis.xaxis.set_major_locator(ticker.MultipleLocator(20))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('Vm (Normalized)', fontsize=12)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_ylim([0, 1.2])
    axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    shiftX = -15
    axis.plot(times, np.roll(dataBase, shiftX),
              color=colorBase, linewidth=2, label='Base')
    axis.plot(times, np.roll(dataPost, shiftX-20),
              color=colorPost, linewidth=2, label='Post')
    axis.legend(loc='upper right', ncol=1,
                prop={'size': 8}, numpoints=1, frameon=False)


fig = plt.figure(figsize=(8, 8))  # half 11 x 7 inch page
gs0 = fig.add_gridspec(2, 2, height_ratios=[0.8, 0.2])  # Overall: ? row, ? columns
gsActMaps = gs0[0].subgridspec(3, 2, hspace=0.3)  # 3 rows, 2 columns for Activation Maps
gsConDealys = gs0[1].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Conduction Delays

# Build Activation Map plots
ActMapTitleX = 0.1
ActMapTitleY = 1
axActMap140CTRL = fig.add_subplot(gsActMaps[0])
axActMap140MEHP = fig.add_subplot(gsActMaps[1])
axActMap140MEHP.text(ActMapTitleX, ActMapTitleY, 'PCL 140 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap140MEHP.transAxes)
axActMap190CTRL = fig.add_subplot(gsActMaps[2])
axActMap190MEHP = fig.add_subplot(gsActMaps[3])
axActMap190MEHP.text(ActMapTitleX, ActMapTitleY, 'PCL 190 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap190MEHP.transAxes)
axActMap240CTRL = fig.add_subplot(gsActMaps[4])
axActMap240MEHP = fig.add_subplot(gsActMaps[5])
axActMap240MEHP.text(ActMapTitleX, ActMapTitleY, 'PCL 240 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap240MEHP.transAxes)
axCVMAX = fig.add_subplot(gs0[2])
axActTimes = fig.add_subplot(gs0[3])

example_ActMapCTRL(axActMap140CTRL, actMapsMEHPbase[140])
example_ActMapMEHP(axActMap140MEHP, actMapsMEHP[140])
example_ActMapCTRL(axActMap190CTRL, actMapsMEHPbase[190])
example_ActMapMEHP(axActMap190MEHP, actMapsMEHP[190])
example_ActMapCTRL(axActMap240CTRL, actMapsMEHPbase[240])
example_ActMapMEHP(axActMap240MEHP, actMapsMEHP[240])

# Build Conduction Delay plots
axConDelaysCTRL = fig.add_subplot(gsConDealys[0])
axConDelaysMEHP = fig.add_subplot(gsConDealys[2])

example_ConductionDelay(axConDelaysCTRL, VmMEHP_base140, VmMEHP_post140, VmtMEHP_base140)
example_plot(axConDelaysMEHP)

# Fill rest with example plots
example_plot(fig.add_subplot(gsConDealys[1]))
example_plot(fig.add_subplot(gsConDealys[3]))
example_plot(axCVMAX)
example_plot(axActTimes)

fig.show()
fig.savefig('MEHP_CV.svg')
