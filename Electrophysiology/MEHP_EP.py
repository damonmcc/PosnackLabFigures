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


fig = plt.figure(figsize=(11, 6))  # half 11 x 7 inch page
gs0 = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.7])  # Overall: 1 row, 2 columns
gs1 = gs0[0].subgridspec(2, 1, hspace=0.3)  # 2 rows for APD80 and VERP Before-After plots
gsBeforeAfter = gs1[1].subgridspec(1, 2, width_ratios=[0.5, 0.5], wspace=0.1)  # 2 columns for VERP Before-After plots
gsAPD = gs0[1].subgridspec(2, 1, hspace=0.4)  # 2 rows for all APD spikes
gsAPD1 = gsAPD[0].subgridspec(1, 4, wspace=0.5)  # 4 columns for APD spikes
gsAPD2 = gsAPD[1].subgridspec(1, 4, wspace=0.5)  # 4 columns for APD spikes
colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'grey'
colorMEHP = 'black'

# APD 80 plot
axAPD80 = fig.add_subplot(gs1[0])
APD80Data = pd.read_csv('data/APD_binned2.csv')
axAPD80.set_xlim([60, 290])
axAPD80.tick_params(axis='x', labelsize=10, which='both', direction='in')
axAPD80.xaxis.set_major_locator(ticker.MultipleLocator(50))
axAPD80.xaxis.set_minor_locator(ticker.MultipleLocator(10))
axAPD80.set_ylim([45, 81])
axAPD80.tick_params(axis='y', labelsize=10, which='both', direction='in')
axAPD80.yaxis.set_major_locator(ticker.MultipleLocator(5))
axAPD80.yaxis.set_minor_locator(ticker.MultipleLocator(1))
axAPD80.spines['right'].set_visible(False)
axAPD80.spines['top'].set_visible(False)
axAPD80.set_ylabel('APD80 (ms)', fontsize=12)
axAPD80.set_xlabel('Pacing Cycle Length (ms)', fontsize=12)
APD80CTRL_legend = mlines.Line2D([], [], color=colorCTRL, ls='solid', marker='o',
                                 markersize=8, label='Ctrl')
APD80MEHP_legend = mlines.Line2D([], [], color=colorMEHP, ls='dotted', marker='s',
                                 markersize=8, label='MEHP')
axAPD80.legend(handles=[APD80CTRL_legend, APD80MEHP_legend], loc='upper left',
               numpoints=1, fontsize=12, frameon=False)
axAPD80.errorbar(APD80Data.BCL, APD80Data.Ctrl_mean,
                 yerr=np.array(APD80Data.Ctrl_std) / np.sqrt(3),
                 ls='solid', color=colorCTRL, marker='o', ms=6)
axAPD80.errorbar(APD80Data.BCL, APD80Data.MEHP_mean,
                 yerr=np.array(APD80Data.MEHP_std) / np.sqrt(3),
                 ls='dotted', color=colorMEHP, marker='s', ms=6)

# VERP Before-After plots
VERPData = pd.read_csv('data/mehp_verp.csv')
VERPtime = np.array([0, 5])  # before-after time array
# Control data
axVERPctrl = fig.add_subplot(gsBeforeAfter[0])
VERPctrlBASE = VERPData.base[(VERPData.group == 'ctrl')]
VERPctrlPOST = VERPData.post[(VERPData.group == 'ctrl')]
VERPctrl = [VERPctrlBASE, VERPctrlPOST]
# MEHP
axVERPmhep = fig.add_subplot(gsBeforeAfter[1])
VERPmehpBASE = VERPData.base[(VERPData.group == 'mehp')]
VERPmehpPOST = VERPData.post[(VERPData.group == 'mehp')]
VERPmehp = [VERPmehpBASE, VERPmehpPOST]
# Plotting
# TODO add plot title
for idx, axis in enumerate([axVERPctrl, axVERPmhep]):
    axis.set_xlim([0.8, 2.2])
    # axis.margins(0.8, 2.2)
    axis.tick_params(axis='x', labelsize=10, rotation=45, which='both', direction='in')
    axis.set_xticklabels(['', 'Baseline', '30 min', 'Baseline', '30 min'])
    axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axVERP.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylim([50, 200])
    axis.tick_params(axis='y', labelsize=10, which='both', direction='in')
    axis.yaxis.set_major_locator(ticker.MultipleLocator(50))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

axVERPctrl.plot([1, 2], VERPctrl, ls='solid', color=colorCTRL, marker='o', ms=8, mfc='w')
axVERPmhep.get_yaxis().set_visible(False)
axVERPmhep.spines['left'].set_visible(False)
axVERPmhep.plot([1, 2], VERPmehp, ls='solid', color=colorMEHP, marker='o', ms=8, mfc='w')
VERPctrl_legend = mlines.Line2D([], [], color=colorCTRL, ls='solid',
                                markersize=8, label='Ctrl')
VERPmhep_legend = mlines.Line2D([], [], color=colorMEHP, ls='solid',
                                markersize=8, label='MEHP')
axVERPctrl.legend(handles=[VERPctrl_legend, VERPmhep_legend], loc='upper left',
                  numpoints=1, fontsize=12, frameon=False)

# APD Spike plots
APD = pd.read_csv('data/APD30_90_up90_MEHP_Reform2.csv')
# PCL = 140
# Action Potentials, Vm
axVm_140 = fig.add_subplot(gsAPD1[0])
VmCTRL_post140 = np.genfromtxt('data/ap_examples/20161221-rata-12-pcl140-ctrl-vm.csv', delimiter=',')
VmtCTRL_post140 = np.genfromtxt('data/ap_examples/20161221-rata-12-pcl140-ctrl-t.csv', delimiter=',')
VmMEHP_post140 = np.genfromtxt('data/ap_examples/20161222-ratb-13-pcl140-mehp-vm.csv', delimiter=',')
VmtMEHP_post140 = np.genfromtxt('data/ap_examples/20161222-ratb-13-pcl140-mehp-t.csv', delimiter=',')
# APD30
axAPD30_140 = fig.add_subplot(gsAPD1[1])
APD30CTRL_base140 = APD.APD30_140[(APD.group == 'ctrl') & (APD.context == 'base')]
APD30CTRL_post140 = APD.APD30_140[(APD.group == 'ctrl') & (APD.context == 'post')]
APD30MEHP_base140 = APD.APD30_140[(APD.group == 'mehp') & (APD.context == 'base')]
APD30MEHP_post140 = APD.APD30_140[(APD.group == 'mehp') & (APD.context == 'post')]
# APD90
axAPD90_140 = fig.add_subplot(gsAPD1[2])
APD90CTRL_base140 = APD.APD90_140[(APD.group == 'ctrl') & (APD.context == 'base')]
APD90CTRL_post140 = APD.APD90_140[(APD.group == 'ctrl') & (APD.context == 'post')]
APD90MEHP_base140 = APD.APD90_140[(APD.group == 'mehp') & (APD.context == 'base')]
APD90MEHP_post140 = APD.APD90_140[(APD.group == 'mehp') & (APD.context == 'post')]
# APDTri
axAPDtri_140 = fig.add_subplot(gsAPD1[3])
APDtriCTRL_base140 = APD.APDtri_140[(APD.group == 'ctrl') & (APD.context == 'base')]
APDtriCTRL_post140 = APD.APDtri_140[(APD.group == 'ctrl') & (APD.context == 'post')]
APDtriMEHP_base140 = APD.APDtri_140[(APD.group == 'mehp') & (APD.context == 'base')]
APDtriMEHP_post140 = APD.APDtri_140[(APD.group == 'mehp') & (APD.context == 'post')]
# PCL = 240
# Action Potentials, Vm
axVm_240 = fig.add_subplot(gsAPD2[0])
VmCTRL_post240 = np.genfromtxt('data/ap_examples/20161221-ratb-04-pcl240-vm.csv', delimiter=',')
VmtCTRL_post240 = np.genfromtxt('data/ap_examples/20161221-ratb-04-pcl240-t.csv', delimiter=',')
VmMEHP_post240 = np.genfromtxt('data/ap_examples/20180720-rata-27-pcl240-vm.csv', delimiter=',')
VmtMEHP_post240 = np.genfromtxt('data/ap_examples/20180720-rata-27-pcl240-t.csv', delimiter=',')
# APD30
axAPD30_240 = fig.add_subplot(gsAPD2[1])
APD30CTRL_base240 = APD.APD30_240[(APD.group == 'ctrl') & (APD.context == 'base')]
APD30CTRL_post240 = APD.APD30_240[(APD.group == 'ctrl') & (APD.context == 'post')]
APD30MEHP_base240 = APD.APD30_240[(APD.group == 'mehp') & (APD.context == 'base')]
APD30MEHP_post240 = APD.APD30_240[(APD.group == 'mehp') & (APD.context == 'post')]
# APD90
axAPD90_240 = fig.add_subplot(gsAPD2[2])
APD90CTRL_base240 = APD.APD90_240[(APD.group == 'ctrl') & (APD.context == 'base')]
APD90CTRL_post240 = APD.APD90_240[(APD.group == 'ctrl') & (APD.context == 'post')]
APD90MEHP_base240 = APD.APD90_240[(APD.group == 'mehp') & (APD.context == 'base')]
APD90MEHP_post240 = APD.APD90_240[(APD.group == 'mehp') & (APD.context == 'post')]
# APDTri
axAPDtri_240 = fig.add_subplot(gsAPD2[3])
APDtriCTRL_base240 = APD.APDtri_240[(APD.group == 'ctrl') & (APD.context == 'base')]
APDtriCTRL_post240 = APD.APDtri_240[(APD.group == 'ctrl') & (APD.context == 'post')]
APDtriMEHP_base240 = APD.APDtri_240[(APD.group == 'mehp') & (APD.context == 'base')]
APDtriMEHP_post240 = APD.APDtri_240[(APD.group == 'mehp') & (APD.context == 'post')]
# Plotting
axAPD90_140.set_title('PCL 140 ms', loc='left', fontsize=14)
axAPD90_240.set_title('PCL 240 ms', loc='left', fontsize=14)
# Action Potentials, Vm
for idx, axis in enumerate([axVm_140, axVm_240]):
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_xlim([0.8, 1.1])
    m = ['0', '75', '150', '225', '300']
    ticks = [0.8, 0.875, 0.95, 1.025, 1.1]
    axis.set_xticks(ticks)
    axis.set_xticklabels(m)
    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.set_ylim([0, 1.5])
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.set_ylabel('Vm (Normalized)', fontsize=12)
axVm_140.plot(VmtCTRL_post140, np.roll(VmCTRL_post140, 78),
              color=colorCTRL, linewidth=2, label='Ctrl')
axVm_140.plot(VmtMEHP_post140, np.roll(VmMEHP_post140, 49) + 0.03,
              color=colorMEHP, linewidth=2, linestyle='--', label='MEHP')
axVm_240.plot(VmtCTRL_post240, np.roll(VmCTRL_post240, 78),
              color=colorCTRL, linewidth=2, label='Ctrl')
axVm_240.plot(VmtMEHP_post240, np.roll(VmMEHP_post240, 49) + 0.03,
              color=colorMEHP, linewidth=2, linestyle='--', label='MEHP')
# Legend
VmCTRL_legend = mlines.Line2D([], [], color=colorMEHP, ls='dotted', marker='s',
                              markersize=8, label='Ctrl')
VmMEHP_legend = mlines.Line2D([], [], color=colorMEHP, ls='dotted', marker='s',
                              markersize=8, label='MEHP')
axVm_140.legend(loc='upper left', ncol=1,
                prop={'size': 8}, numpoints=1, frameon=False)
# APD30 plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPD30_140, axAPD30_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=14)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axVERP.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('APD30 (ms)', fontsize=12)
    axis.set_ylim([0, 30])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
for idx, data in enumerate([APD30CTRL_base140, APD30CTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30MEHP_base140, APD30MEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30CTRL_base240, APD30CTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30MEHP_base240, APD30MEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
# APD90 plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPD90_140, axAPD90_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=14)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('APD90 (ms)', fontsize=12)
    axis.set_ylim([0, 100])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
for idx, data in enumerate([APD90CTRL_base140, APD90CTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90MEHP_base140, APD90MEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90CTRL_base240, APD90CTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90MEHP_base240, APD90MEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx],
                    yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
# APD Tri. plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPDtri_140, axAPDtri_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=14)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axVERP.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('APD Tri. (ms)', fontsize=12)
    axis.set_ylim([0, 80])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
for idx, data in enumerate([APDtriCTRL_base140, APDtriCTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPDtri_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                     barWidth, color=colorsBP[idx],
                     yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                     error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APDtriMEHP_base140, APDtriMEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPDtri_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                     barWidth, color=colorsBP[idx],
                     yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                     error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APDtriCTRL_base240, APDtriCTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPDtri_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                     barWidth, color=colorsBP[idx],
                     yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                     error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APDtriMEHP_base240, APDtriMEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPDtri_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                     barWidth, color=colorsBP[idx],
                     yerr=np.std(data) / np.sqrt(len(data)), ecolor='k',
                     error_kw=dict(lw=1, capsize=4, capthick=1.0))
# Legend
APDbase_legend = mlines.Line2D([], [], color=colorBase, ls='solid',
                               linewidth=6, label='Baseline')
APDpost_legend = mlines.Line2D([], [], color=colorPost, ls='solid',
                               linewidth=6, label='30 min')
axAPDtri_140.legend(handles=[APDbase_legend, APDpost_legend],
                    loc='upper left', ncol=1,
                    prop={'size': 8}, numpoints=1, frameon=False)

# for n in range(4):
#     ax = fig.add_subplot(gsAPD1[n])
#     example_plot(ax)
# for n in range(4):
#     ax = fig.add_subplot(gsAPD2[n])
#     example_plot(ax)

fig.show()
fig.savefig('MEHP_EP.svg')
