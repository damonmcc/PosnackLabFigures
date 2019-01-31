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
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#%% All Data Input
snrt = pd.read_csv('data/mehp_snrt_ngp.csv')
wbcl = pd.read_csv('data/mehp_wbcl_ngp.csv')
avnerp = pd.read_csv('data/mehp_avnerp_ngp.csv')
# Skip header and import columns: times (ms), ECG (mV)
ECGControl = np.genfromtxt('traces/20180619-rata-T011840.txt', skip_header=8, usecols=(1, 2), skip_footer=4)
ECGMEHP = np.genfromtxt('traces/20180720T012007-rata-ecg.txt', skip_header=8, usecols=(1, 2), skip_footer=4)
ECGControl = np.fliplr(ECGControl)
ECGMEHP = np.fliplr(ECGMEHP)
#%% Organize all the data frames. You can control the outliers by changing the value property (BOOL)
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
#%% Descriptive & Comparative Statistics
np.mean(cp_snrt)
np.std(cp_snrt)/np.sqrt(len(cp_snrt))

np.mean(mp_snrt)
np.std(mp_snrt)/np.sqrt(len(mp_snrt))

stats.ttest_ind(cp_snrt,mp_snrt,axis=0, nan_policy='omit')
stats.ttest_ind(cp_wbcl,mp_wbcl,axis=0, nan_policy='omit')
stats.ttest_ind(cp_avnerp,mp_avnerp,axis=0, nan_policy='omit')

#%% Main Figure Setup
fig = plt.figure(figsize=(6, 7))  # half an 8.5 x 11 page
gs0 = fig.add_gridspec(3, 1)  # Overall: 3 rows, 1 columns
gs1 = gs0[2].subgridspec(1, 3, hspace=0.3)  # split last row into 3 columns for barplots
axECGControl = fig.add_subplot(gs0[0])
axECGMEHP = fig.add_subplot(gs0[1])
axSNRT = fig.add_subplot(gs1[0])
axWBCL = fig.add_subplot(gs1[1])
axAVNERP = fig.add_subplot(gs1[2])
baseColor = 'indianred'
timeColor = 'midnightblue'
#%% Control and MEHP ECG traces
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
axECGControl.set_ylabel('Ctrl', fontsize=9)
axECGControl.set_ylim([-1.5, 1.5])
ECGControlStart = 0.72  # ms after 00:51:12.502
axECGControl.set_xlim([ECGControlStart, ECGControlStart + ECGwindow])
ECGControlMin = np.min(ECGControl[:, 1])
# ECGControlAdjusted -= ECGControlMean
axECGControl.plot(ECGControl[:, 0], ECGControl[:, 1],
                  color=timeColor, linewidth=2)
# =============================================================================
# # Draw lines to show PR and QRS lengths
# ECGCorrection = 10  # lines seems to be too far ahead of desired start times
# ControlPR_start = 265 - ECGCorrection  # 00:51:12.767
# ControlPR_width = 44
# ControlPR_end = ControlPR_start + ControlPR_width
# ControlPR_Hash = ControlPR_start, ControlPR_end
# ControlPR_HashHeight = 0.5
# axECGControl.text(ControlPR_start + ControlPR_width / 2, ControlPR_HashHeight, str(ControlPR_width) + 'ms',
#                   ha='center', va='bottom', fontsize=7, fontweight='bold')
# axECGControl.plot([ControlPR_start, ControlPR_end],
#                   [ControlPR_HashHeight, ControlPR_HashHeight],
#                   "k-", linewidth=1)
# axECGControl.plot([ControlPR_Hash, ControlPR_Hash],
#                   [ControlPR_HashHeight - 0.1, ControlPR_HashHeight + 0.1],
#                   "k-", linewidth=1)
# ControlQRS_start = ControlPR_end
# ControlQRS_width = 20
# ControlQRS_end = ControlQRS_start + ControlQRS_width
# ControlQRS_Hash = ControlQRS_start, ControlQRS_end
# ControlQRS_HashHeight = 1.85
# axECGControl.text(ControlQRS_start + ControlQRS_width / 2, ControlQRS_HashHeight, str(ControlQRS_width) + 'ms',
#                   ha='center', va='bottom', fontsize=7, fontweight='bold')
# axECGControl.plot([ControlQRS_start, ControlQRS_start + ControlQRS_width],
#                   [ControlQRS_HashHeight, ControlQRS_HashHeight],
#                   "k-", linewidth=1)
# axECGControl.plot([ControlQRS_Hash, ControlQRS_Hash],
#                   [ControlQRS_HashHeight - 0.1, ControlQRS_HashHeight + 0.1],
#                   "k-", linewidth=1)
# yl = axECGControl.get_ylim()
# yr = yl[1] - yl[0]
# xl = axECGControl.get_xlim()
# xr = xl[1] - xl[0]
# axECGControl.text(xl[0] - (xr * 0.4), yr * 0.5, 'A', ha='center', va='bottom', fontsize=9, fontweight='bold')
# 
# =============================================================================
# MEHP ECG trace
axECGMEHP.set_ylabel('MEHP', fontsize=9)
axECGMEHP.set_ylim([-0.5, 0.5])
ECGMEHPStart = 0.8  # ms after 00:51:13.502
axECGMEHP.set_xlim([ECGMEHPStart, ECGMEHPStart + ECGwindow])
axECGMEHP.plot(ECGMEHP[:, 0], ECGMEHP[:, 1],
               color=timeColor, linewidth=2)
# Draw lines to show PR and QRS lengths
# =============================================================================
# MEHPPR_start = 114  # 00:51:12.589
# MEHPPR_width = 59
# MEHPPR_end = MEHPPR_start + MEHPPR_width
# MEHPPR_Hash = MEHPPR_start, MEHPPR_end
# MEHPPR_HashHeight = 0.5
# axECGMEHP.text(MEHPPR_start + MEHPPR_width / 2, MEHPPR_HashHeight, str(MEHPPR_width) + 'ms',
#                ha='center', va='bottom', fontsize=7, fontweight='bold')
# axECGMEHP.plot([MEHPPR_start, MEHPPR_end],
#                [MEHPPR_HashHeight, MEHPPR_HashHeight],
#                "k-", linewidth=1)
# axECGMEHP.plot([MEHPPR_Hash, MEHPPR_Hash],
#                [MEHPPR_HashHeight - 0.1, MEHPPR_HashHeight + 0.1],
#                "k-", linewidth=1)
# MEHPQRS_start = MEHPPR_end
# MEHPQRS_width = 25  # 00:51:12.663
# MEHPQRS_end = MEHPQRS_start + MEHPQRS_width
# MEHPQRS_Hash = MEHPQRS_start, MEHPQRS_end
# MEHPQRS_HashHeight = 1.85
# axECGMEHP.text(MEHPQRS_start + MEHPQRS_width / 2, MEHPQRS_HashHeight, str(MEHPQRS_width) + 'ms',
#                ha='center', va='bottom', fontsize=7, fontweight='bold')
# axECGMEHP.plot([MEHPQRS_start, MEHPQRS_start + MEHPQRS_width],
#                [MEHPQRS_HashHeight, MEHPQRS_HashHeight],
#                "k-", linewidth=1)
# axECGMEHP.plot([MEHPQRS_Hash, MEHPQRS_Hash],
#                [MEHPQRS_HashHeight - 0.1, MEHPQRS_HashHeight + 0.1],
#                "k-", linewidth=1)
# =============================================================================

# ECG Scale: mv and ms bars forming an L
# =============================================================================
# ECGScaleTime = [20, 500 / 1000]  # 20 ms, 500 mV
# ECGScaleOrigin = [axECGMEHP.get_xlim()[1] - 20, axECGMEHP.get_ylim()[0] + 0.3]
# ECGScaleOriginPad = [2, 0.2]
# # Time scale bar
# axECGMEHP.plot([ECGScaleOrigin[0], ECGScaleOrigin[0] + ECGScaleTime[0]],
#                [ECGScaleOrigin[1], ECGScaleOrigin[1]],
#                "k-", linewidth=1)
# axECGMEHP.text(ECGScaleOrigin[0], ECGScaleOrigin[1] - ECGScaleOriginPad[1],
#                str(ECGScaleTime[0]) + 'ms',
#                ha='left', va='top', fontsize=7, fontweight='bold')
# # Voltage scale bar
# axECGMEHP.plot([ECGScaleOrigin[0], ECGScaleOrigin[0]],
#                [ECGScaleOrigin[1], ECGScaleOrigin[1] + ECGScaleTime[1]],
#                "k-", linewidth=1)
# axECGMEHP.text(ECGScaleOrigin[0] - ECGScaleOriginPad[0], ECGScaleOrigin[1],
#                str(int(ECGScaleTime[1] * 1000)) + 'mV',
#                ha='right', va='bottom', fontsize=7, fontweight='bold')
# =============================================================================
#%% Bar Plots
width=0.28
labels = ['Ctrl', 'MEHP']

# SNRT
axSNRT.bar(0.4, np.mean(cp_snrt), width, color='k',fill=False, yerr=np.std(cp_snrt) / np.sqrt(len(cp_snrt)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axSNRT.bar(0.7, np.mean(mp_snrt), width, edgecolor='r', color='r', fill=False, yerr=np.std(mp_snrt) / np.sqrt(len(mp_snrt)),
          ecolor='r', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axSNRT.plot(np.linspace(0.35,0.45, num=len(cp_snrt)),cp_snrt, 'o', color='k', mfc='none')
axSNRT.plot(np.linspace(0.65,0.75, num=len(mp_snrt)),mp_snrt, 'o', color='r', mfc='none')
axSNRT.set_ylim(bottom=0, top=602)
axSNRT.set_xlim(left=0.2, right=0.9)
axSNRT.spines['right'].set_visible(False)
axSNRT.spines['top'].set_visible(False)
axSNRT.set_xticks([0.4, 0.7])
axSNRT.set_xticklabels(labels, fontsize=9)
axSNRT.xaxis.set_ticks_position('bottom')
axSNRT.yaxis.set_ticks_position('left')
axSNRT.set_ylabel('SNRT (msec)', fontsize=9)
axSNRT.text(0.55,602,'* p<0.05', ha='center', va='bottom', fontsize=8)
axSNRT.plot([0.4, 0.7], [596, 596], "k-", linewidth = 2)
yl = axSNRT.get_ylim()
yr = yl[1] - yl[0]
xl = axSNRT.get_xlim()
xr = xl[1] - xl[0]
axSNRT.text(xl[0] - (xr * 0.5), yr * 0.96, 'C', ha='center', va='bottom', fontsize=10, fontweight='bold')

# WBCL
axWBCL.bar(0.4, np.mean(cp_wbcl), width, color='k', fill=False, yerr=np.std(cp_wbcl) / np.sqrt(len(cp_wbcl)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axWBCL.bar(0.7, np.mean(mp_wbcl), width, edgecolor='r',fill=False, yerr=np.std(mp_wbcl) / np.sqrt(len(mp_wbcl)),
          ecolor='r', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axWBCL.plot(np.linspace(0.35,0.45, num=len(cp_wbcl)),cp_wbcl, 'o', color='k', mfc='none')
axWBCL.plot(np.linspace(0.65,0.75, num=len(mp_wbcl)),mp_wbcl, 'o', color='r', mfc='none')
axWBCL.set_ylim(bottom=0, top=250)
axWBCL.set_xlim(left=0.2, right=0.9)
axWBCL.spines['right'].set_visible(False)
axWBCL.spines['top'].set_visible(False)
axWBCL.set_xticks([0.4, 0.7])
axWBCL.set_xticklabels(labels, fontsize=9)
axWBCL.xaxis.set_ticks_position('bottom')
axWBCL.yaxis.set_ticks_position('left')
axWBCL.set_ylabel('WBCL (msec)', fontsize=9)
axWBCL.text(0.55,250,'p=0.07', ha='center', va='bottom', fontsize=8)
axWBCL.plot([0.4, 0.7], [247, 247], "k-", linewidth = 2)
yl = axWBCL.get_ylim()
yr = yl[1] - yl[0]
xl = axWBCL.get_xlim()
xr = xl[1] - xl[0]
axWBCL.text(xl[0] - (xr * 0.5), yr * 0.96, 'D', ha='center', va='bottom', fontsize=10, fontweight='bold')

# AVNERP
axAVNERP.bar(0.4, np.mean(cp_avnerp), width, color='k', fill=False, yerr=np.std(cp_avnerp) / np.sqrt(len(cp_avnerp)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axAVNERP.bar(0.7, np.mean(mp_avnerp), width, edgecolor='r', color='r', fill=False, yerr=np.std(mp_avnerp) / np.sqrt(len(mp_avnerp)),
          ecolor='r', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axAVNERP.plot(np.linspace(0.35,0.45, num=len(cp_avnerp)),cp_avnerp, 'o', color='k', mfc='none')
axAVNERP.plot(np.linspace(0.65,0.75, num=len(mp_avnerp)),mp_avnerp, 'o', color='r', mfc='none')
axAVNERP.set_ylim(bottom=0, top=250)
axAVNERP.set_xlim(left=0.2, right=0.9)
axAVNERP.spines['right'].set_visible(False)
axAVNERP.spines['top'].set_visible(False)
axAVNERP.set_xticks([0.4, 0.7])
axAVNERP.set_xticklabels(labels, fontsize=9)
axAVNERP.xaxis.set_ticks_position('bottom')
axAVNERP.yaxis.set_ticks_position('left')
axAVNERP.set_ylabel('AVNERP (msec)', fontsize=9)
axAVNERP.text(0.55,250,'* p<0.05', ha='center', va='bottom', fontsize=8)
axAVNERP.plot([0.4, 0.7], [247, 247], "k-", linewidth = 2)
yl = axAVNERP.get_ylim()
yr = yl[1] - yl[0]
xl = axAVNERP.get_xlim()
xr = xl[1] - xl[0]
axAVNERP.text(xl[0] - (xr * 0.5), yr * 0.96, 'E', ha='center', va='bottom', fontsize=10, fontweight='bold')

#%% Saving
plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9, wspace=0.8, hspace=0.9)
#fig.savefig('MEHP_AV_Properties.eps') #Vector, for loading into InkScape for touchup and further modification
#fig.savefig('MEHP_AV_Properties.pdf') #Vector, finalized, for uploading to journal/disseminate
#fig.savefig('MEHP_AV_Properties.png') #Raster, for web/presentations