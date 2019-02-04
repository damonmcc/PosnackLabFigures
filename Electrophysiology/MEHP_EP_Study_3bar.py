#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:12:13 2019

This Plot will have 4 panels:
    
    A. Example traces of SNRT
    B. SNRT
    C. WBCL
    D. AVNERP

@author: raf
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import rcParams

# Common parameters
colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'lightgrey'
colorMEHP = 'grey'
fontSizeLeg = 8

rcParams.update({'figure.autolayout': True})
# %% All Data Input
snrt = pd.read_csv('data/mehp_snrt_ngp.csv')
wbcl = pd.read_csv('data/mehp_wbcl_ngp.csv')
avnerp = pd.read_csv('data/mehp_avnerp_ngp.csv')
# Skip header and import columns: times (ms), ECG (mV)
ECGControl = np.genfromtxt('data/20180619-rata-T011840.txt', skip_header=8, usecols=(1, 2), skip_footer=4)
ECGMEHP = np.genfromtxt('data/20180720T012007-rata-ecg.txt', skip_header=8, usecols=(1, 2), skip_footer=4)
ECGControl = np.fliplr(ECGControl)
ECGMEHP = np.fliplr(ECGMEHP)
# %% Organize all the data frames. You can control the outliers by changing the value property (BOOL)
cb_snrt = snrt.base[snrt.group == 'control']
cp_snrt = snrt.post[snrt.group == 'control']
mb_snrt = snrt.base[snrt.group == 'MEHP']
mp_snrt = snrt.post[snrt.group == 'MEHP']

cb_wbcl = wbcl.base[(wbcl.group == 'control') & (wbcl.valid == 1)]
cp_wbcl = wbcl.post[(wbcl.group == 'control') & (wbcl.valid == 1)]
mb_wbcl = wbcl.base[(wbcl.group == 'MEHP') & (wbcl.valid == 1)]
mp_wbcl = wbcl.post[(wbcl.group == 'MEHP') & (wbcl.valid == 1)]

cb_avnerp = avnerp.base[(avnerp.group == 'control') & (avnerp.valid == 1)]
cp_avnerp = avnerp.post[(avnerp.group == 'control') & (avnerp.valid == 1)]
mb_avnerp = avnerp.base[(avnerp.group == 'MEHP') & (avnerp.valid == 1)]
mp_avnerp = avnerp.post[(avnerp.group == 'MEHP') & (avnerp.valid == 1)]
# %% Descriptive & Comparative Statistics
np.mean(cp_snrt)
np.std(cp_snrt) / np.sqrt(len(cp_snrt))

np.mean(mp_snrt)
np.std(mp_snrt) / np.sqrt(len(mp_snrt))

stats.ttest_ind(cp_snrt, mp_snrt, axis=0, nan_policy='omit')
stats.ttest_ind(cp_wbcl, mp_wbcl, axis=0, nan_policy='omit')
stats.ttest_ind(cp_avnerp, mp_avnerp, axis=0, nan_policy='omit')

# %% Main Figure Setup
fig = plt.figure(figsize=(6, 7))  # half an 8.5 x 11 page
gs0 = fig.add_gridspec(3, 1)  # Overall: 3 rows, 1 columns
gs1 = gs0[2].subgridspec(1, 3, hspace=0.3)  # split last row into 3 columns for barplots
axECGControl = fig.add_subplot(gs0[0])
axECGMEHP = fig.add_subplot(gs0[1])
axSNRT = fig.add_subplot(gs1[0])
axWBCL = fig.add_subplot(gs1[1])
axAVNERP = fig.add_subplot(gs1[2])
bC = 'indianred'  # control color or baseline color
tC = 'midnightblue'  # treatment color or time color
bC = tC  # both are timed
# %% Control and MEHP ECG traces
# axECGControl.text(-40, 96, 'A', ha='center', va='bottom', fontsize=16, fontweight='bold')
ECGwindow = 2.5

for idx, ax in enumerate([axECGControl, axECGMEHP]):
    # ax.tick_params(axis='x', labelsize=7, which='both', direction='in')
    # ax.tick_params(axis='y', labelsize=7, which='both', direction='in')
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    [s.set_visible(False) for s in ax.spines.values()]
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(bottom=False, left=False)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    # # for label in ax.xaxis.get_ticklabels():
    # #     label.set_rotation(45)
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    # ax.grid(True)

# Control ECG trace
axECGControl.set_ylabel('Ctrl', fontsize=12)
axECGControl.set_ylim([-1.5, 1.5])
ECGControlStart = 0.72  # ms after 00:51:12.502
axECGControl.set_xlim([ECGControlStart, ECGControlStart + ECGwindow])
ECGControlMin = np.min(ECGControl[:, 1])
# ECGControlAdjusted -= ECGControlMean
axECGControl.plot(ECGControl[:, 0], ECGControl[:, 1],
                  color=bC, linewidth=1.5)
# Draw lines to show SNRT lengths
ControlSNRT_start = 1.7
ControlSNRT_width = 0.379
ControlSNRT_end = ControlSNRT_start + ControlSNRT_width
ControlSNRT_Hash = ControlSNRT_start, ControlSNRT_end
ControlSNRT_HashHeight = 0.7
axECGControl.text(ControlSNRT_start + ControlSNRT_width / 2, ControlSNRT_HashHeight,
                  str(int(ControlSNRT_width * 1000)) + 'ms',
                  ha='center', va='bottom', fontsize=7, fontweight='bold')
axECGControl.plot([ControlSNRT_start, ControlSNRT_end],
                  [ControlSNRT_HashHeight, ControlSNRT_HashHeight],
                  "k-", linewidth=1)
axECGControl.plot([ControlSNRT_Hash, ControlSNRT_Hash],
                  [ControlSNRT_HashHeight - 0.1, ControlSNRT_HashHeight + 0.1],
                  "k-", linewidth=1)
# Draw arrows to show pacing spikes
paceArrowX = np.linspace(0.8, ControlSNRT_start, num=5)
paceArrowH = ControlSNRT_HashHeight * 0.8
[axECGControl.arrow(x, paceArrowH, 0, -0.2,
                    head_width=0.02, head_length=0.1, fc='k', ec='k') for x in paceArrowX]
# ECG Scale: mv and ms bars forming an L
CTRL_ECGScale = [0.15, 300 / 1000]  # 50 ms, 100 mV
CTRL_ECGScaleOrigin = [axECGControl.get_xlim()[1] - 0.4, axECGControl.get_ylim()[0] + 0.2]
CTRL_ECGScaleOriginPad = [0.02, 0.06]
# Time scale bar
axECGControl.plot([CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[0] + CTRL_ECGScale[0]],
                  [CTRL_ECGScaleOrigin[1], CTRL_ECGScaleOrigin[1]],
                  "k-", linewidth=1)
axECGControl.text(CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[1] - CTRL_ECGScaleOriginPad[1],
                  str(CTRL_ECGScale[0]) + 'ms',
                  ha='left', va='top', fontsize=7, fontweight='bold')
# Voltage scale bar
axECGControl.plot([CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[0]],
                  [CTRL_ECGScaleOrigin[1], CTRL_ECGScaleOrigin[1] + CTRL_ECGScale[1]],
                  "k-", linewidth=1)
axECGControl.text(CTRL_ECGScaleOrigin[0] - CTRL_ECGScaleOriginPad[0], CTRL_ECGScaleOrigin[1],
                  str(int(CTRL_ECGScale[1] * 1000)) + 'mV',
                  ha='right', va='bottom', fontsize=7, fontweight='bold')
# =============================================================================
yl = axECGControl.get_ylim()
yr = yl[1] - yl[0]
xl = axECGControl.get_xlim()
xr = xl[1] - xl[0]
axECGControl.text(xl[0] - (xr * 0.07), yr * 0.2, 'A', ha='center', va='bottom', fontsize=12, fontweight='bold')

# MEHP ECG trace
axECGMEHP.set_ylabel('MEHP', fontsize=12)
axECGMEHP.set_ylim([-0.5, 0.5])
ECGMEHPStart = 0.8  # ms after 00:51:13.502
axECGMEHP.set_xlim([ECGMEHPStart, ECGMEHPStart + ECGwindow])
axECGMEHP.plot(ECGMEHP[:, 0], ECGMEHP[:, 1],
               color=tC, linewidth=1.5)
# Draw lines to show SNRT lengths
MEHPSNRT_start = 1.78
MEHPSNRT_width = 0.644
MEHPSNRT_end = MEHPSNRT_start + MEHPSNRT_width
MEHPSNRT_Hash = MEHPSNRT_start, MEHPSNRT_end
MEHPSNRT_HashHeight = 0.25
axECGMEHP.text(MEHPSNRT_start + MEHPSNRT_width / 2, MEHPSNRT_HashHeight,
               str(int(MEHPSNRT_width * 1000)) + 'ms',
               ha='center', va='bottom', fontsize=7, fontweight='bold')
axECGMEHP.plot([MEHPSNRT_start, MEHPSNRT_end],
               [MEHPSNRT_HashHeight, MEHPSNRT_HashHeight],
               "k-", linewidth=1)
axECGMEHP.plot([MEHPSNRT_Hash, MEHPSNRT_Hash],
               [MEHPSNRT_HashHeight - 0.03, MEHPSNRT_HashHeight + 0.03],
               "k-", linewidth=1)
# Draw arrows to show pacing spikes
paceArrowX = np.linspace(0.86, MEHPSNRT_start, num=5)
paceArrowH = MEHPSNRT_HashHeight * 0.83
[axECGMEHP.arrow(x, paceArrowH, 0, -0.2 / 3,
                 head_width=0.02, head_length=0.1 / 3, fc='k', ec='k') for x in paceArrowX]
# ECG Scale: mv and ms bars forming an L
CTRL_ECGScale = [0.15, 100 / 1000]  # 50 ms, 100 mV
CTRL_ECGScaleOrigin = [axECGMEHP.get_xlim()[1] - 0.4, axECGMEHP.get_ylim()[0] + 0.2]
CTRL_ECGScaleOriginPad = [0.02, 0.02]
# Time scale bar
axECGMEHP.plot([CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[0] + CTRL_ECGScale[0]],
               [CTRL_ECGScaleOrigin[1], CTRL_ECGScaleOrigin[1]],
               "k-", linewidth=1)
axECGMEHP.text(CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[1] - CTRL_ECGScaleOriginPad[1],
               str(CTRL_ECGScale[0]) + 'ms',
               ha='left', va='top', fontsize=7, fontweight='bold')
# Voltage scale bar
axECGMEHP.plot([CTRL_ECGScaleOrigin[0], CTRL_ECGScaleOrigin[0]],
               [CTRL_ECGScaleOrigin[1], CTRL_ECGScaleOrigin[1] + CTRL_ECGScale[1]],
               "k-", linewidth=1)
axECGMEHP.text(CTRL_ECGScaleOrigin[0] - CTRL_ECGScaleOriginPad[0], CTRL_ECGScaleOrigin[1],
               str(int(CTRL_ECGScale[1] * 1000)) + 'mV',
               ha='right', va='bottom', fontsize=7, fontweight='bold')
# =============================================================================
yl = axECGMEHP.get_ylim()
yr = yl[1] - yl[0]
xl = axECGMEHP.get_xlim()
xr = xl[1] - xl[0]
axECGMEHP.text(xl[0] - (xr * 0.07), yr * 0.2, 'B', ha='center', va='bottom', fontsize=12, fontweight='bold')
# %% Bar Plots
width = 0.28
labels = ['Ctrl', 'MEHP']

# SNRT
axSNRT.bar(0.4, np.mean(cp_snrt), width, color=colorCTRL, fill=True,
           yerr=np.std(cp_snrt) / np.sqrt(len(cp_snrt)),
           error_kw=dict(lw=1, capsize=4, capthick=1.0))
axSNRT.bar(0.7, np.mean(mp_snrt), width, color=colorMEHP, fill=True,
           yerr=np.std(mp_snrt) / np.sqrt(len(mp_snrt)),
           error_kw=dict(lw=1, capsize=4, capthick=1.0))
# axSNRT.bar(0.4, np.mean(cp_snrt), width, edgecolor=bC, fill=False, yerr=np.std(cp_snrt) / np.sqrt(len(cp_snrt)),
#            ecolor=bC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axSNRT.bar(0.7, np.mean(mp_snrt), width, edgecolor=tC, color=tC, fill=False,
#            yerr=np.std(mp_snrt) / np.sqrt(len(mp_snrt)),
#            ecolor=tC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axSNRT.plot(np.linspace(0.35, 0.45, num=len(cp_snrt)), cp_snrt, 'o', color=bC, mfc='none')
# axSNRT.plot(np.linspace(0.65, 0.75, num=len(mp_snrt)), mp_snrt, 'o', color=tC, mfc='none')
axSNRT.set_ylim(bottom=0, top=602)
axSNRT.set_xlim(left=0.2, right=0.9)
axSNRT.spines['right'].set_visible(False)
axSNRT.spines['top'].set_visible(False)
axSNRT.set_xticks([0.4, 0.7])
axSNRT.set_xticklabels(labels, fontsize=9)
axSNRT.xaxis.set_ticks_position('bottom')
axSNRT.yaxis.set_ticks_position('left')
axSNRT.set_ylabel('SNRT (msec)', fontsize=9)
axSNRT.text(0.55, 602, '* p<0.05', ha='center', va='bottom', fontsize=8)
axSNRT.plot([0.4, 0.7], [596, 596], "k-", linewidth=2)
yl = axSNRT.get_ylim()
yr = yl[1] - yl[0]
xl = axSNRT.get_xlim()
xr = xl[1] - xl[0]
axSNRT.text(xl[0] - (xr * 0.5), yr * 0.96, 'C', ha='center', va='bottom', fontsize=12, fontweight='bold')

# WBCL
axWBCL.bar(0.4, np.mean(cp_wbcl), width, color=colorCTRL, fill=True,
           yerr=np.std(cp_wbcl) / np.sqrt(len(cp_wbcl)),
           error_kw=dict(lw=1, capsize=4, capthick=1.0))
axWBCL.bar(0.7, np.mean(mp_wbcl), width, color=colorMEHP, fill=True,
           yerr=np.std(mp_wbcl) / np.sqrt(len(mp_wbcl)),
           error_kw=dict(lw=1, capsize=4, capthick=1.0))
# axWBCL.bar(0.4, np.mean(cp_wbcl), width, edgecolor=bC, fill=False, yerr=np.std(cp_wbcl) / np.sqrt(len(cp_wbcl)),
#            ecolor=bC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axWBCL.bar(0.7, np.mean(mp_wbcl), width, edgecolor=tC, fill=False, yerr=np.std(mp_wbcl) / np.sqrt(len(mp_wbcl)),
#            ecolor=tC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axWBCL.plot(np.linspace(0.35, 0.45, num=len(cp_wbcl)), cp_wbcl, 'o', color=bC, mfc='none')
# axWBCL.plot(np.linspace(0.65, 0.75, num=len(mp_wbcl)), mp_wbcl, 'o', color=tC, mfc='none')
axWBCL.set_ylim(bottom=0, top=250)
axWBCL.set_xlim(left=0.2, right=0.9)
axWBCL.spines['right'].set_visible(False)
axWBCL.spines['top'].set_visible(False)
axWBCL.set_xticks([0.4, 0.7])
axWBCL.set_xticklabels(labels, fontsize=9)
axWBCL.xaxis.set_ticks_position('bottom')
axWBCL.yaxis.set_ticks_position('left')
axWBCL.set_ylabel('WBCL (msec)', fontsize=9)
axWBCL.text(0.55, 250, 'p=0.07', ha='center', va='bottom', fontsize=8)
axWBCL.plot([0.4, 0.7], [247, 247], "k-", linewidth=2)
yl = axWBCL.get_ylim()
yr = yl[1] - yl[0]
xl = axWBCL.get_xlim()
xr = xl[1] - xl[0]
axWBCL.text(xl[0] - (xr * 0.5), yr * 0.96, 'D', ha='center', va='bottom', fontsize=12, fontweight='bold')

# AVNERP
axAVNERP.bar(0.4, np.mean(cp_avnerp), width, color=colorCTRL, fill=True,
             yerr=np.std(cp_avnerp) / np.sqrt(len(cp_avnerp)),
             error_kw=dict(lw=1, capsize=4, capthick=1.0))
axAVNERP.bar(0.7, np.mean(mp_avnerp), width, color=colorMEHP, fill=True,
             yerr=np.std(mp_avnerp) / np.sqrt(len(mp_avnerp)),
             error_kw=dict(lw=1, capsize=4, capthick=1.0))
# axAVNERP.bar(0.7, np.mean(mp_avnerp), width, edgecolor=tC, color='b', fill=False,
#              yerr=np.std(mp_avnerp) / np.sqrt(len(mp_avnerp)),
#              ecolor=tC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axAVNERP.bar(0.4, np.mean(cp_avnerp), width, edgecolor=bC, fill=False, yerr=np.std(cp_avnerp) / np.sqrt(len(cp_avnerp)),
#              ecolor=bC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axAVNERP.bar(0.7, np.mean(mp_avnerp), width, edgecolor=tC, color='b', fill=False,
#              yerr=np.std(mp_avnerp) / np.sqrt(len(mp_avnerp)),
#              ecolor=tC, error_kw=dict(lw=1, capsize=4, capthick=2.0))
# axAVNERP.plot(np.linspace(0.35, 0.45, num=len(cp_avnerp)), cp_avnerp, 'o', color=bC, mfc='none')
# axAVNERP.plot(np.linspace(0.65, 0.75, num=len(mp_avnerp)), mp_avnerp, 'o', color=tC, mfc='none')
axAVNERP.set_ylim(bottom=0, top=250)
axAVNERP.set_xlim(left=0.2, right=0.9)
axAVNERP.spines['right'].set_visible(False)
axAVNERP.spines['top'].set_visible(False)
axAVNERP.set_xticks([0.4, 0.7])
axAVNERP.set_xticklabels(labels, fontsize=9)
axAVNERP.xaxis.set_ticks_position('bottom')
axAVNERP.yaxis.set_ticks_position('left')
axAVNERP.set_ylabel('AVNERP (msec)', fontsize=9)
axAVNERP.text(0.55, 250, '* p<0.05', ha='center', va='bottom', fontsize=8)
axAVNERP.plot([0.4, 0.7], [247, 247], "k-", linewidth=2)
yl = axAVNERP.get_ylim()
yr = yl[1] - yl[0]
xl = axAVNERP.get_xlim()
xr = xl[1] - xl[0]
axAVNERP.text(xl[0] - (xr * 0.5), yr * 0.96, 'E', ha='center', va='bottom', fontsize=12, fontweight='bold')

# %% Saving
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.8, hspace=0.9)
# fig.savefig('MEHP_AV_Properties.eps') #Vector, for loading into InkScape for touchup and further modification
# fig.savefig('MEHP_AV_Properties.pdf') #Vector, finalized, for uploading to journal/disseminate
plt.show()
fig.savefig('MEHP_EP_3bar.svg')  # Vector, for loading into InkScape for touchup and further modification
fig.savefig('MEHP_EP_3bar.png')  # Raster, for web/presentations
