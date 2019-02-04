import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch
import pandas as pd
from scipy import stats


def example_plot(axis):
    axis.plot([1, 2])
    axis.set_xlabel('x-label', fontsize=12)
    axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Title', fontsize=14)


# Common parameters
colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'grey'
colorMEHP = 'black'
fontSizeLeg = 8

fig = plt.figure(figsize=(11, 6))  # half 11 x 7 inch page
gs0 = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.7])  # Overall: 1 row, 2 columns
gs1 = gs0[0].subgridspec(2, 1, hspace=0.3)  # 2 rows for APD80 and VERP Before-After plots
gsBeforeAfter = gs1[1].subgridspec(1, 2, width_ratios=[0.5, 0.5], wspace=0.1)  # 2 columns for VERP Before-After plots
gsAPD = gs0[1].subgridspec(2, 1, hspace=0.5)  # 2 rows for all APD spikes
gsAPD1 = gsAPD[0].subgridspec(1, 4, wspace=0.6)  # 4 columns for APD spikes
gsAPD2 = gsAPD[1].subgridspec(1, 4, wspace=0.6)  # 4 columns for APD spikes


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
               numpoints=1, fontsize=8, frameon=False)
axAPD80.errorbar(APD80Data.BCL, APD80Data.Ctrl_mean,
                 yerr=np.array(APD80Data.Ctrl_std) / np.sqrt(3),
                 ls='solid', color=colorCTRL, marker='o', ms=6)
axAPD80.errorbar(APD80Data.BCL, APD80Data.MEHP_mean,
                 yerr=np.array(APD80Data.MEHP_std) / np.sqrt(3),
                 ls='dotted', color=colorMEHP, marker='s', ms=6)
yl = axAPD80.get_ylim()
yr = yl[1] - yl[0]
xl = axAPD80.get_xlim()
xr = xl[1] - xl[0]
axAPD80.text(xl[0] - (xr * 0.15), yl[1], 'A', ha='center', va='bottom', fontsize=12, fontweight='bold')
# =============================================================================

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
axVERPctrl.set_ylabel('VERP (ms)', fontsize=12)
VERPctrl_legend = mlines.Line2D([], [], color=colorCTRL, ls='solid',
                                markersize=8, label='Ctrl')
VERPmhep_legend = mlines.Line2D([], [], color=colorMEHP, ls='solid',
                                markersize=8, label='MEHP')
axVERPctrl.legend(handles=[VERPctrl_legend, VERPmhep_legend], loc='upper left',
                  numpoints=1, fontsize=8, frameon=False)
# Significant bars
yl = axVERPctrl.get_ylim()
yr = yl[1] - yl[0]
xl = axVERPctrl.get_xlim()
xr = xl[1] - xl[0]
ySig1 = yl[1]*0.89
ySig2 = yl[1]*0.94
# MEHP sig.
axVERPmhep.text(1.5, ySig1, '* p<0.05', ha='center', va='bottom', fontsize=8)
axVERPmhep.plot([1, 2], [ySig1, ySig1], "k-")
# Ctrl-MEHP Post sig.
axVERPmhep.text(1, ySig2, '* p<0.05', ha='center', va='bottom', fontsize=8)
con = ConnectionPatch(xyA=[2, ySig2], xyB=[2, ySig2], coordsA="data", coordsB="data",
                      axesA=axVERPmhep, axesB=axVERPctrl, arrowstyle="-")
axVERPmhep.add_artist(con)
# Figure letter
axVERPctrl.text(xl[0] - (xr * 0.35), yl[1], 'B', ha='center', va='bottom', fontsize=12, fontweight='bold')
# =============================================================================

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
axAPtri_140 = fig.add_subplot(gsAPD1[3])
APtriCTRL_base140 = APD.APDtri_140[(APD.group == 'ctrl') & (APD.context == 'base')]
APtriCTRL_post140 = APD.APDtri_140[(APD.group == 'ctrl') & (APD.context == 'post')]
APtriMEHP_base140 = APD.APDtri_140[(APD.group == 'mehp') & (APD.context == 'base')]
APtriMEHP_post140 = APD.APDtri_140[(APD.group == 'mehp') & (APD.context == 'post')]
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
axAPtri_240 = fig.add_subplot(gsAPD2[3])
APDtriCTRL_base240 = APD.APDtri_240[(APD.group == 'ctrl') & (APD.context == 'base')]
APDtriCTRL_post240 = APD.APDtri_240[(APD.group == 'ctrl') & (APD.context == 'post')]
APDtriMEHP_base240 = APD.APDtri_240[(APD.group == 'mehp') & (APD.context == 'base')]
APDtriMEHP_post240 = APD.APDtri_240[(APD.group == 'mehp') & (APD.context == 'post')]
# =============================================================================

# Plotting
# Titles
axAPD90_140.text(-0.5, -30, 'PCL 140 ms', ha='center', va='bottom', fontsize=14)
axAPD90_240.text(-0.5, -30, 'PCL 240 ms', ha='center', va='bottom', fontsize=14)

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
axVm_240.legend(loc='upper left', ncol=1,
                prop={'size': 8}, numpoints=1, frameon=False)
yl = axVm_140.get_ylim()
yr = yl[1] - yl[0]
xl = axVm_140.get_xlim()
xr = xl[1] - xl[0]
axVm_140.text(xl[0] - (xr * 0.4), yl[1], 'C', ha='center', va='bottom', fontsize=12, fontweight='bold')
yl = axVm_240.get_ylim()
yr = yl[1] - yl[0]
xl = axVm_240.get_xlim()
xr = xl[1] - xl[0]
axVm_240.text(xl[0] - (xr * 0.4), yl[1], 'D', ha='center', va='bottom', fontsize=12, fontweight='bold')
# =============================================================================

# APD30 plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPD30_140, axAPD30_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=12)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('APD30 (ms)', fontsize=12)
    axis.set_ylim([0, 40])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(5))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
for idx, data in enumerate([APD30CTRL_base140, APD30CTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30MEHP_base140, APD30MEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30CTRL_base240, APD30CTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD30MEHP_base240, APD30MEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD30_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
yl = axAPD30_140.get_ylim()
yr = yl[1] - yl[0]
xl = axAPD30_140.get_xlim()
xr = xl[1] - xl[0]
axAPD30_140.text(xl[0] - (xr * 0.35), yl[1], 'E', ha='center', va='bottom', fontsize=12, fontweight='bold')
yl = axAPD30_240.get_ylim()
yr = yl[1] - yl[0]
xl = axAPD30_240.get_xlim()
xr = xl[1] - xl[0]
axAPD30_240.text(xl[0] - (xr * 0.35), yl[1], 'F', ha='center', va='bottom', fontsize=12, fontweight='bold')
# =============================================================================

# APD90 plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPD90_140, axAPD90_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=12)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('APD90 (ms)', fontsize=12)
    axis.set_ylim([0, 120])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

# Significant bars
yl_140 = axAPD90_140.get_ylim()
yr_140 = yl_140[1] - yl_140[0]
xl_140 = axAPD90_140.get_xlim()
xr_140 = xl_140[1] - xl_140[0]
yl_240 = axAPD90_240.get_ylim()
yr_240 = yl_240[1] - yl_240[0]
xl_240 = axAPD90_240.get_xlim()
xr_240 = xl_240[1] - xl_240[0]
ySig1 = yl_240[1]*0.9
ySig2 = yl_240[1]*0.98
for idx, data in enumerate([APD90CTRL_base140, APD90CTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90MEHP_base140, APD90MEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90CTRL_base240, APD90CTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APD90MEHP_base240, APD90MEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPD90_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))

    # axAPD90_240.plot(np.linspace(barCenterTicks[1] - barWidth/2 + barOffset/2, barCenterTicks[1] + barOffset,
    #                              num=len(data)), data,
    #                  'o', color=colorsBP[idx], markersize=3, mfc='none')

# MEHP sig.
axAPD90_240.text(barCenterTicks[1], ySig1, '* p<0.05', ha='center', va='bottom', fontsize=8)
axAPD90_240.plot([barCenterTicks[1] - barWidth, barCenterTicks[1] + (barWidth)],
                 [ySig1, ySig1], "k-")
# Ctrl-MEHP Post sig.
axAPD90_240.text(barCenterTicks[1] - (barWidth/2), ySig2, '* p<0.05', ha='center', va='bottom', fontsize=8)
axAPD90_240.plot([barCenterTicks[0] + (barWidth/2), barCenterTicks[1] + (barWidth)],
                 [ySig2, ySig2], "k-")
# Figure letter
axAPD90_140.text(xl_140[0] - (xr_140 * 0.35), yl_140[1], 'G', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Figure letter
axAPD90_240.text(xl_240[0] - (xr_240 * 0.35), yl_240[1], 'H', ha='center', va='bottom', fontsize=12, fontweight='bold')
# =============================================================================

# AP Tri. plots
barWidth = 0.28
barGap = 0.02
barCenterTicks = [0.4, 1.2]
for idx, axis in enumerate([axAPtri_140, axAPtri_240]):
    axis.set_xlim([0, 1.6])
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xticks(barCenterTicks)
    axis.set_xticklabels(['Ctrl', 'MEHP'], rotation=0, fontsize=12)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # axVERP.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylabel('AP Tri. (ms)', fontsize=12)
    axis.set_ylim([0, 100])
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
for idx, data in enumerate([APtriCTRL_base140, APtriCTRL_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPtri_140.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APtriMEHP_base140, APtriMEHP_post140]):
    barOffset = idx * (barWidth + barGap)
    axAPtri_140.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APDtriCTRL_base240, APDtriCTRL_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPtri_240.bar(barCenterTicks[0] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))
for idx, data in enumerate([APDtriMEHP_base240, APDtriMEHP_post240]):
    barOffset = idx * (barWidth + barGap)
    axAPtri_240.bar(barCenterTicks[1] - (barWidth / 2) + barOffset, np.mean(data),
                    barWidth, color=colorsBP[idx], fill=True,
                    yerr=np.std(data) / np.sqrt(len(data)),
                    error_kw=dict(lw=1, capsize=4, capthick=1.0))

# Significant bars
yl_140 = axAPtri_140.get_ylim()
yr_140 = yl_140[1] - yl_140[0]
xl_140 = axAPtri_140.get_xlim()
xr_140 = xl_140[1] - xl_140[0]

yl_240 = axAPtri_240.get_ylim()
yr_240 = yl_240[1] - yl_240[0]
xl_240 = axAPtri_240.get_xlim()
xr_240 = xl_240[1] - xl_240[0]
ySig1 = yl_240[1]*0.9
ySig2 = yl_240[1]*0.98
# MEHP sig.
axAPtri_240.text(barCenterTicks[1], ySig1, '* p<0.05', ha='center', va='bottom', fontsize=8)
axAPtri_240.plot([barCenterTicks[1] - barWidth, barCenterTicks[1] + (barWidth)],
                 [ySig1, ySig1], "k-")
# Ctrl-MEHP Post sig.
axAPtri_240.text(barCenterTicks[1] - (barWidth/2), ySig2, '* p=0.05', ha='center', va='bottom', fontsize=8)
axAPtri_240.plot([barCenterTicks[0] + (barWidth/2), barCenterTicks[1] + (barWidth)],
                 [ySig2, ySig2], "k-")
# Figure letter
axAPtri_140.text(xl_140[0] - (xr_140 * 0.35), yl_140[1], 'I', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Figure letter
axAPtri_240.text(xl_240[0] - (xr_240 * 0.35), yl_240[1], 'J', ha='center', va='bottom', fontsize=12, fontweight='bold')
# Legend
APbase_legend = mlines.Line2D([], [], color=colorBase, ls='solid',
                              linewidth=6, label='Baseline')
APpost_legend = mlines.Line2D([], [], color=colorPost, ls='solid',
                              linewidth=6, label='30 min')
axAPD30_140.legend(handles=[APbase_legend, APpost_legend],
                   loc='upper left', ncol=1,
                   prop={'size': 8}, numpoints=1, frameon=False)
# =============================================================================

# for n in range(4):
#     ax = fig.add_subplot(gsAPD1[n])
#     example_plot(ax)
# for n in range(4):
#     ax = fig.add_subplot(gsAPD2[n])
#     example_plot(ax)

fig.show()
fig.savefig('MEHP_EP.svg')  # Vector, for loading into InkScape for touchup and further modification
fig.savefig('MEHP_EP.png')  # Raster, for web/presentations

# %% Descriptive & Comparative Statistics
# np.mean(cp_snrt)
# np.std(cp_snrt) / np.sqrt(len(cp_snrt))
#
# np.mean(mp_snrt)
# np.std(mp_snrt) / np.sqrt(len(mp_snrt))

print('\nVERP CTRL:')
print(stats.ttest_ind(VERPctrlBASE, VERPctrlPOST, axis=0, nan_policy='omit'))

print('\nVERP MEHP:')
print(stats.ttest_ind(VERPmehpBASE, VERPmehpPOST, axis=0, nan_policy='omit'))

print('\nVERP CTRL-MEHP:')
print(stats.ttest_ind(VERPctrlPOST, VERPmehpPOST, axis=0, nan_policy='omit'))

print('\nAPD90-240 CTRL:')
print(stats.ttest_ind(APD90CTRL_base240, APD90CTRL_post240, axis=0, nan_policy='omit'))

print('\nAPD90-240 MEHP:')
print(stats.ttest_ind(APD90MEHP_base240, APD90MEHP_post240, axis=0, nan_policy='omit'))

print('\nAPD90-240 CTRL-MEHP:')
print(stats.ttest_ind(APD90CTRL_post240, APD90MEHP_post240, axis=0, nan_policy='omit'))


print('\n# ======================================== #')
print('\nAP Tri-240 CTRL:')
print(stats.ttest_ind(APDtriCTRL_base240, APDtriCTRL_post240, axis=0, nan_policy='omit'))
print('\nAP Tri-240 MEHP:')
print(stats.ttest_ind(APDtriMEHP_base240, APDtriMEHP_post240, axis=0, nan_policy='omit'))
print('\nAP Tri-240 CTRL-MEHP:')
print(stats.ttest_ind(APDtriCTRL_post240, APDtriMEHP_post240, axis=0, nan_policy='omit'))

# print('\nVERP MEHP:')
# print(stats.ttest_ind(VERPmehpBASE, VERPmehpPOST, axis=0, nan_policy='omit'))
