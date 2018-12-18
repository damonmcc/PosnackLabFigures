import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from scipy import stats
from matplotlib import ticker


def example_plot(ax):
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=12)
    ax.set_ylabel('y-label', fontsize=12)
    ax.set_title('Title', fontsize=14)


fig = plt.figure(figsize=(11, 7))  # half 11 x 7 inch page.
gs0 = fig.add_gridspec(1, 2)  # Overall: 1 row, 2 columns
gs1 = gs0[0].subgridspec(2, 1)  # 2 rows for APD80 and VERP Before-After plot
gsAPD = gs0[1].subgridspec(2, 1)  # 2 rows for all APD spikes
gsAPD1 = gsAPD[0].subgridspec(1, 4)  # 4 columns for APD spikes
gsAPD2 = gsAPD[1].subgridspec(1, 4)  # 4 columns for APD spikes
colorBase = 'indianred'
colorPost = 'midnightblue'

# APD 80 plot
axAPD80 = fig.add_subplot(gs1[0])
APD80Data = pd.read_csv('data/APD_binned2.csv')
axAPD80.set_xlim([60, 290])
axAPD80.tick_params(axis='x', labelsize=12, which='both', direction='in')
axAPD80.xaxis.set_major_locator(ticker.MultipleLocator(50))
axAPD80.xaxis.set_minor_locator(ticker.MultipleLocator(10))
axAPD80.set_ylim([45, 81])
axAPD80.tick_params(axis='y', labelsize=12, which='both', direction='in')
axAPD80.yaxis.set_major_locator(ticker.MultipleLocator(5))
axAPD80.yaxis.set_minor_locator(ticker.MultipleLocator(1))
axAPD80.spines['right'].set_visible(False)
axAPD80.spines['top'].set_visible(False)
axAPD80.set_ylabel('APD80 (ms)', fontsize=14)
axAPD80.set_xlabel('Pacing Cycle Length (ms)', fontsize=16)
ctrl_legend = mlines.Line2D([], [], color='indianred', ls='solid', marker='o',
                            markersize=8, label='Ctrl')
mehp_legend = mlines.Line2D([], [], color='midnightblue', ls='dotted', marker='s',
                            markersize=8, label='MEHP')
axAPD80.legend(handles=[ctrl_legend, mehp_legend], loc='upper left',
               numpoints=1, fontsize=16)
axAPD80.errorbar(APD80Data.BCL, APD80Data.Ctrl_mean,
                 yerr=np.array(APD80Data.Ctrl_std) / np.sqrt(3),
                 ls='solid', color='indianred', marker='o', ms=8)
axAPD80.errorbar(APD80Data.BCL, APD80Data.MEHP_mean,
                 yerr=np.array(APD80Data.MEHP_std) / np.sqrt(3),
                 ls='dotted', color='midnightblue', marker='s', ms=8)

# VERP Before-After plots
axVERP = fig.add_subplot(gs1[1])
VERPData = pd.read_csv('data/mehp_verp.csv')
VERPtime = np.array([0, 5])  # before-after time array
# Control data
VERPctrlBASE = VERPData.base[(VERPData.group == 'ctrl')]
VERPctrlPOST = VERPData.post[(VERPData.group == 'ctrl')]
VERPctrl = [VERPctrlBASE, VERPctrlPOST]
# MEHP
VERPmehpBASE = VERPData.base[(VERPData.group == 'mehp')]
VERPmehpPOST = VERPData.post[(VERPData.group == 'mehp')]
VERPmehp = [VERPmehpBASE, VERPmehpPOST]
# Plotting
axVERP.set_xlim([0.8, 4.2])
axVERP.tick_params(axis='x', labelsize=12, rotation=45, which='both', direction='in')
axVERP.set_xticklabels(['', 'Baseline', '30 min', 'Baseline', '30 min'])
axVERP.xaxis.set_major_locator(ticker.MultipleLocator(1))
# axVERP.xaxis.set_minor_locator(ticker.MultipleLocator(10))
axVERP.set_ylim([50, 200])
axVERP.tick_params(axis='y', labelsize=12, which='both', direction='in')
axVERP.yaxis.set_major_locator(ticker.MultipleLocator(50))
axVERP.yaxis.set_minor_locator(ticker.MultipleLocator(10))
axVERP.spines['right'].set_visible(False)
axVERP.spines['top'].set_visible(False)
axVERP.plot([1, 2], VERPctrl, ls='solid', color=colorBase, marker='o', ms=8, mfc='w')
axVERP.plot([3, 4], VERPmehp, ls='solid', color=colorPost, marker='o', ms=8, mfc='w')



for n in range(4):
    ax = fig.add_subplot(gsAPD1[n])
    example_plot(ax)
for n in range(4):
    ax = fig.add_subplot(gsAPD2[n])
    example_plot(ax)

df = np.loadtxt('data/mehp_verp.csv', delimiter=',', skiprows=1, usecols=(2, 3))

fig.show()
fig.savefig('EP.svg')
