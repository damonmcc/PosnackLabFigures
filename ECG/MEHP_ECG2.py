# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:24:59 2018
@author: Rafael Jaimes III, PhD
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
# from scipy import stats


fig= plt.figure(figsize=(11,5.5)) # half 8.5 x 11 inch page.
baseColor='indianred'
timeColor='midnightblue'

# Timeline figure
#img=mpimg.imread('TimelineFigure.png')
#ax = plt.imshow(img)
#ax.axis('off')
ax = fig.add_subplot(2, 3, 1)
ax.plot(range(0,100), range(0,100))
ax.text(-40,96,'A',ha='center',va='bottom',fontsize=16,fontweight='bold')


ecg = pandas.read_csv('PR_Data.csv')

# PR Times
cb_pr=ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'PR')]
cp_pr=ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'PR')]
cp_pr[18]=np.nan;
mb_pr=ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'PR')]
mp_pr=ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'PR')]

# HR Times
cb_hr=ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'HR')]
cp_hr=ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'HR')]
mb_hr=ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'HR')]
mp_hr=ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'HR')]

# RR Times
cb_rr=ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'RR')]
cp_rr=ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'RR')]
mb_rr=ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'RR')]
mp_rr=ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'RR')]

# QT Times
cb_qt=ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'QTcF')]
cp_qt=ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'QTcF')]
mb_qt=ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'QTcF')]
mp_qt=ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'QTcF')]

# QRS Times
cb_qrs=ecg.base[(ecg.group == 'Ctrl') & (ecg.parameter == 'QRS')]
cp_qrs=ecg.post[(ecg.group == 'Ctrl') & (ecg.parameter == 'QRS')]
mb_qrs=ecg.base[(ecg.group == 'MEHP') & (ecg.parameter == 'QRS')]
mp_qrs=ecg.post[(ecg.group == 'MEHP') & (ecg.parameter == 'QRS')]

plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

# Descriptive Statistics
ctrl_base_pr=np.mean(cb_pr)
ctrl_post_pr=np.mean(cp_pr)
mehp_base_pr=np.mean(mb_pr)
mehp_post_pr=np.mean(mp_pr)

ctrl_base_qt=np.mean(cb_qt)
ctrl_post_qt=np.mean(cp_qt)
mehp_base_qtr=np.mean(mb_qt)
mehp_post_qt=np.mean(mp_qt)


ax = fig.add_subplot(2, 3, 2)
width=0.28
b1=ax.bar(0.4-width/2,np.mean(cb_hr), width, color=baseColor,yerr=np.std(cb_hr)/np.sqrt(len(cb_hr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Ctrl')
b2=ax.bar(0.7-width/2,np.mean(cp_hr),width, color=timeColor,yerr=np.std(cp_hr)/np.sqrt(len(cp_hr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Ctrl')
b3=ax.bar(1.2-width/2,np.mean(mb_hr), width, color=baseColor,yerr=np.std(mb_hr)/np.sqrt(len(mb_hr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='MEHP')
b4=ax.bar(1.5-width/2,np.mean(mp_hr), width, color=timeColor,yerr=np.std(mp_hr)/np.sqrt(len(mp_hr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0), label='MEHP')
plt.ylim(ymin=0,ymax=250)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
labels=['Ctrl', 'MEHP']
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('HR (bpm)',fontsize=14)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.9,'B',ha='center',va='bottom',fontsize=16,fontweight='bold')
plt.legend((b1[0], b2[0]),('Baseline', '30 min'),bbox_to_anchor=(1.1, 1.4),loc='upper right',ncol=1, prop={'size':14},numpoints=1,frameon=False)

ax = fig.add_subplot(2, 3, 3)
ax.bar(0.4-width/2,np.mean(cb_rr), width, color=baseColor,yerr=np.std(cb_rr)/np.sqrt(len(cb_rr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp_rr),width, color=timeColor,yerr=np.std(cp_rr)/np.sqrt(len(cp_rr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb_rr), width, color=baseColor,yerr=np.std(mb_rr)/np.sqrt(len(mb_rr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp_rr), width, color=timeColor,yerr=np.std(mp_rr)/np.sqrt(len(mp_rr)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=450)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('RR (ms)',fontsize=14)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'C',ha='center',va='bottom',fontsize=16,fontweight='bold')

ax = fig.add_subplot(2, 3, 4)
ax.bar(0.4-width/2,np.mean(cb_pr), width, color=baseColor,yerr=np.std(cb_pr)/np.sqrt(4),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp_pr),width, color=timeColor,yerr=np.std(cp_pr)/np.sqrt(4),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb_pr), width, color=baseColor,yerr=np.std(mb_pr)/np.sqrt(4),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp_pr), width, color=timeColor,yerr=np.std(mp_pr)/np.sqrt(4),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=80)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('PR (ms)',fontsize=14)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'D',ha='center',va='bottom',fontsize=16,fontweight='bold')
ax.text(1.2,76,'*',ha='center',va='center',fontsize=16)
ax.plot([0.7-width/2, 1.5-width/2], [68, 68], "k-",linewidth=2)
ax.plot([1.2-width/2, 1.5-width/2], [73, 73], "k-",linewidth=2)

ax = fig.add_subplot(2, 3, 5)
ax.bar(0.4-width/2,np.mean(cb_qrs), width, color=baseColor,yerr=np.std(cb_qrs)/np.sqrt(len(cb_qrs)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp_qrs),width, color=timeColor, yerr=np.std(cp_qrs)/np.sqrt(len(cp_qrs)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb_qrs), width, color=baseColor, yerr=np.std(mb_qrs)/np.sqrt(len(mb_qrs)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp_qrs), width, color=timeColor, yerr=np.std(mp_qrs)/np.sqrt(len(mp_qrs)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=30)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('QRS Width (ms)',fontsize=14)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'E',ha='center',va='bottom',fontsize=16,fontweight='bold')

#stats.ttest_ind(cp_pr,mp_pr,axis=0, nan_policy='omit')

ax = fig.add_subplot(2, 3, 6)
ax.bar(0.4-width/2,np.mean(cb_qt), width, color=baseColor,yerr=np.std(cb_qt)/np.sqrt(len(cb_qt)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp_qt),width, color=timeColor, yerr=np.std(cp_qt)/np.sqrt(len(cp_qt)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb_qt), width, color=baseColor, yerr=np.std(mb_qt)/np.sqrt(len(mb_qt)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp_qt), width, color=timeColor, yerr=np.std(mp_qt)/np.sqrt(len(mp_qt)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=305)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('QTc (ms)',fontsize=14)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'F',ha='center',va='bottom',fontsize=16,fontweight='bold')


plt.tight_layout()
plt.show()
# plt.savefig('MEHP_ECG_wQRS.svg')