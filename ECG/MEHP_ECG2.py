# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:24:59 2018
@author: Rafael Jaimes III, PhD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pandas
# from scipy import stats


fig = plt.figure(figsize=(11, 8))  # half 8.5 x 11 inch page.
baseColor = 'indianred'
timeColor = 'midnightblue'

gs0 = fig.add_gridspec(2, 3)  # Overall: Two rows, 3 columns
gs1 = gs0[0].subgridspec(2, 1)  # 2 rows for ECG traces
axECGControl = fig.add_subplot(gs1[0])
axECGMEHP = fig.add_subplot(gs1[1])


# Control and MEHP ECG traces
# axECGControl.text(-40, 96, 'A', ha='center', va='bottom', fontsize=16, fontweight='bold')
# Skip header and import columns: times (ms), ECG (mV)
ECGControl = np.genfromtxt('data/20171024-ratb_PR _length.txt',
                           skip_header=28, usecols=(1, 2), skip_footer=2)
ECGMEHP = np.genfromtxt('data/20171024-rata_PR_length.txt',
                        skip_header=27, usecols=(1, 2), skip_footer=2)

for idk, ax in enumerate([axECGControl, axECGMEHP]):
    ax.tick_params(axis='x', labelsize=7, which='both', direction='in')
    ax.tick_params(axis='y', labelsize=7, which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

axECGControl.set_ylabel('CTRL', fontsize=14)
axECGControl.set_ylim([-1, 1])
axECGControl.set_xlim([150, 450])
axECGControl.plot(ECGControl[:, 0], ECGControl[:, 1],
                  color=timeColor)

axECGMEHP.set_ylabel('MEHP', fontsize=14)
axECGMEHP.set_xlabel('Time (ms)')
axECGMEHP.set_ylim([-1, 1])
axECGMEHP.set_xlim([50, 350])
axECGMEHP.plot(ECGMEHP[:, 0], ECGMEHP[:, 1],
               color=timeColor)


# ECG data/statistics for bar plots
ecg = pandas.read_csv('data/PR_Data.csv')
# PR Times
cb_pr = ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'PR')]
cp_pr = ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'PR')]
cp_pr[18] = np.nan
mb_pr = ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'PR')]
mp_pr = ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'PR')]

# HR Times
cb_hr = ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'HR')]
cp_hr = ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'HR')]
mb_hr = ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'HR')]
mp_hr = ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'HR')]

# RR Times
cb_rr = ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'RR')]
cp_rr = ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'RR')]
mb_rr = ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'RR')]
mp_rr = ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'RR')]

# QT Times
cb_qt = ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'QTcF')]
cp_qt = ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'QTcF')]
mb_qt = ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'QTcF')]
mp_qt = ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'QTcF')]

# QRS Times
cb_qrs = ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'QRS')]
cp_qrs = ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'QRS')]
mb_qrs = ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'QRS')]
mp_qrs = ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'QRS')]

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# Descriptive Statistics
ctrl_base_pr = np.mean(cb_pr)
ctrl_post_pr = np.mean(cp_pr)
mehp_base_pr = np.mean(mb_pr)
mehp_post_pr = np.mean(mp_pr)

ctrl_base_qt = np.mean(cb_qt)
ctrl_post_qt = np.mean(cp_qt)
mehp_base_qtr = np.mean(mb_qt)
mehp_post_qt = np.mean(mp_qt)

# HR Plot
axHR = fig.add_subplot(gs0[0, 1])
width = 0.28
b1 = axHR.bar(0.4 - width / 2, np.mean(cb_hr), width, color=baseColor, yerr=np.std(cb_hr) / np.sqrt(len(cb_hr)),
              ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Ctrl')
b2 = axHR.bar(0.7 - width / 2, np.mean(cp_hr), width, color=timeColor, yerr=np.std(cp_hr) / np.sqrt(len(cp_hr)),
              ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Ctrl')
b3 = axHR.bar(1.2 - width / 2, np.mean(mb_hr), width, color=baseColor, yerr=np.std(mb_hr) / np.sqrt(len(mb_hr)),
              ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='MEHP')
b4 = axHR.bar(1.5 - width / 2, np.mean(mp_hr), width, color=timeColor, yerr=np.std(mp_hr) / np.sqrt(len(mp_hr)),
              ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='MEHP')
plt.ylim([0, 250])
plt.xlim([0, 1.6])
axHR.spines['right'].set_visible(False)
axHR.spines['top'].set_visible(False)
labels = ['Ctrl', 'MEHP']
plt.xticks([0.4, 1.2], labels, rotation=0, fontsize=14)
axHR.xaxis.set_ticks_position('bottom')
axHR.yaxis.set_ticks_position('left')
axHR.set_ylabel('HR (bpm)', fontsize=14)
# ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
# ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl = axHR.get_ylim()
yr = yl[1] - yl[0]
xl = axHR.get_xlim()
xr = xl[1] - xl[0]
# axHR.text(xl[0] - (xr * 0.33), yr * 0.9, 'B', ha='center', va='bottom', fontsize=16, fontweight='bold')


# RR Plot
axRR = fig.add_subplot(gs0[0, 2])
axRR.bar(0.4 - width / 2, np.mean(cb_rr), width, color=baseColor, yerr=np.std(cb_rr) / np.sqrt(len(cb_rr)),
         ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axRR.bar(0.7 - width / 2, np.mean(cp_rr), width, color=timeColor, yerr=np.std(cp_rr) / np.sqrt(len(cp_rr)),
         ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axRR.bar(1.2 - width / 2, np.mean(mb_rr), width, color=baseColor, yerr=np.std(mb_rr) / np.sqrt(len(mb_rr)),
         ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axRR.bar(1.5 - width / 2, np.mean(mp_rr), width, color=timeColor, yerr=np.std(mp_rr) / np.sqrt(len(mp_rr)),
         ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim([0, 450])
plt.xlim([0, 1.6])
axRR.spines['right'].set_visible(False)
axRR.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0, fontsize=14)
axRR.xaxis.set_ticks_position('bottom')
axRR.yaxis.set_ticks_position('left')
axRR.set_ylabel('RR (ms)', fontsize=14)
# ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
# ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl = axRR.get_ylim()
yr = yl[1] - yl[0]
xl = axRR.get_xlim()
xr = xl[1] - xl[0]
plt.legend((b1[0], b2[0]), ('Baseline', '30 min'), bbox_to_anchor=(1, 1.1), loc='upper right', ncol=1,
           prop={'size': 12}, numpoints=1, frameon=False)
# axRR.text(xl[0] - (xr * 0.33), yr * 0.96, 'C', ha='center', va='bottom', fontsize=16, fontweight='bold')

# PR Plot
axPR = fig.add_subplot(gs0[1, 0])
axPR.bar(0.4 - width / 2, np.mean(cb_pr), width, color=baseColor, yerr=np.std(cb_pr) / np.sqrt(4), ecolor='k',
         error_kw=dict(lw=1, capsize=4, capthick=1.0))
axPR.bar(0.7 - width / 2, np.mean(cp_pr), width, color=timeColor, yerr=np.std(cp_pr) / np.sqrt(4), ecolor='k',
         error_kw=dict(lw=1, capsize=4, capthick=1.0))
axPR.bar(1.2 - width / 2, np.mean(mb_pr), width, color=baseColor, yerr=np.std(mb_pr) / np.sqrt(4), ecolor='k',
         error_kw=dict(lw=1, capsize=4, capthick=1.0))
axPR.bar(1.5 - width / 2, np.mean(mp_pr), width, color=timeColor, yerr=np.std(mp_pr) / np.sqrt(4), ecolor='k',
         error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim([0, 80])
plt.xlim([0, 1.6])
axPR.spines['right'].set_visible(False)
axPR.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0, fontsize=14)
axPR.xaxis.set_ticks_position('bottom')
axPR.yaxis.set_ticks_position('left')
axPR.set_ylabel('PR (ms)', fontsize=14)
# ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
# ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl = axPR.get_ylim()
yr = yl[1] - yl[0]
xl = axPR.get_xlim()
xr = xl[1] - xl[0]
# axPR.text(xl[0] - (xr * 0.33), yr * 0.96, 'D', ha='center', va='bottom', fontsize=16, fontweight='bold')
axPR.text(1.2, 76, '*', ha='center', va='center', fontsize=16)
axPR.plot([0.7 - width / 2, 1.5 - width / 2], [68, 68], "k-", linewidth=2)
axPR.plot([1.2 - width / 2, 1.5 - width / 2], [73, 73], "k-", linewidth=2)

# QRS Plot
axQRS = fig.add_subplot(gs0[1, 1])
axQRS.bar(0.4 - width / 2, np.mean(cb_qrs), width, color=baseColor, yerr=np.std(cb_qrs) / np.sqrt(len(cb_qrs)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQRS.bar(0.7 - width / 2, np.mean(cp_qrs), width, color=timeColor, yerr=np.std(cp_qrs) / np.sqrt(len(cp_qrs)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQRS.bar(1.2 - width / 2, np.mean(mb_qrs), width, color=baseColor, yerr=np.std(mb_qrs) / np.sqrt(len(mb_qrs)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQRS.bar(1.5 - width / 2, np.mean(mp_qrs), width, color=timeColor, yerr=np.std(mp_qrs) / np.sqrt(len(mp_qrs)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim([0, 30])
plt.xlim([0, 1.6])
axQRS.spines['right'].set_visible(False)
axQRS.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0, fontsize=14)
axQRS.xaxis.set_ticks_position('bottom')
axQRS.yaxis.set_ticks_position('left')
axQRS.set_ylabel('QRS Width (ms)', fontsize=14)
# ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
# ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl = axQRS.get_ylim()
yr = yl[1] - yl[0]
xl = axQRS.get_xlim()
xr = xl[1] - xl[0]
# axQRS.text(xl[0] - (xr * 0.33), yr * 0.96, 'E', ha='center', va='bottom', fontsize=16, fontweight='bold')

# stats.ttest_ind(cp_pr,mp_pr,axis=0, nan_policy='omit')

# QTc Plot
axQTc = fig.add_subplot(gs0[1, 2])
axQTc.bar(0.4 - width / 2, np.mean(cb_qt), width, color=baseColor, yerr=np.std(cb_qt) / np.sqrt(len(cb_qt)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQTc.bar(0.7 - width / 2, np.mean(cp_qt), width, color=timeColor, yerr=np.std(cp_qt) / np.sqrt(len(cp_qt)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQTc.bar(1.2 - width / 2, np.mean(mb_qt), width, color=baseColor, yerr=np.std(mb_qt) / np.sqrt(len(mb_qt)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
axQTc.bar(1.5 - width / 2, np.mean(mp_qt), width, color=timeColor, yerr=np.std(mp_qt) / np.sqrt(len(mp_qt)),
          ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim([0, 305])
plt.xlim([0, 1.6])
axQTc.spines['right'].set_visible(False)
axQTc.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0, fontsize=14)
axQTc.xaxis.set_ticks_position('bottom')
axQTc.yaxis.set_ticks_position('left')
axQTc.set_ylabel('QTc (ms)', fontsize=14)
# ax1.text(0.55,292,'*', ha='center', va='bottom', fontsize=20)
# ax1.plot([0.4, 0.7], [292, 292], "k-", linewidth = 4)
yl = axQTc.get_ylim()
yr = yl[1] - yl[0]
xl = axQTc.get_xlim()
xr = xl[1] - xl[0]
# axQTc.text(xl[0] - (xr * 0.33), yr * 0.96, 'F', ha='center', va='bottom', fontsize=16, fontweight='bold')

# Timeline figure
# img = mpimg.imread('TimelineFigure.png')
# ax = plt.imshow(img)
# ax.axis('off')

plt.tight_layout()
plt.show()
# plt.savefig('MEHP_ECG_wQRS.svg')






