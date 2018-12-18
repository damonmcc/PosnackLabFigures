# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:24:59 2018
@author: Rafael Jaimes III, PhD
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
import scipy.io as spio

oaps = pandas.read_csv('APD30_90_up90_MEHP_Reform2.csv')
baseColor='indianred'
timeColor='midnightblue'

# APD30, PCL = 140
cb140_apd30=oaps.APD30_140[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp140_apd30=oaps.APD30_140[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb140_apd30=oaps.APD30_140[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp140_apd30=oaps.APD30_140[(oaps.group == 'mehp') & (oaps.context == 'post')]

# APD90, PCL = 140
cb140_apd90=oaps.APD90_140[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp140_apd90=oaps.APD90_140[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb140_apd90=oaps.APD90_140[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp140_apd90=oaps.APD90_140[(oaps.group == 'mehp') & (oaps.context == 'post')]

# APDTri, PCL =140
cb140_apdtri=oaps.APDtri_140[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp140_apdtri=oaps.APDtri_140[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb140_apdtri=oaps.APDtri_140[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp140_apdtri=oaps.APDtri_140[(oaps.group == 'mehp') & (oaps.context == 'post')]

# APD30, PCL =240
cb240_apd30=oaps.APD30_240[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp240_apd30=oaps.APD30_240[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb240_apd30=oaps.APD30_240[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp240_apd30=oaps.APD30_240[(oaps.group == 'mehp') & (oaps.context == 'post')]

# APD90, PCL =240
cb240_apd90=oaps.APD90_240[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp240_apd90=oaps.APD90_240[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb240_apd90=oaps.APD90_240[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp240_apd90=oaps.APD90_240[(oaps.group == 'mehp') & (oaps.context == 'post')]

# APD Tri, PCL =240
cb240_apdtri=oaps.APDtri_240[(oaps.group == 'ctrl') & (oaps.context == 'base')]
cp240_apdtri=oaps.APDtri_240[(oaps.group == 'ctrl') & (oaps.context == 'post')]
mb240_apdtri=oaps.APDtri_240[(oaps.group == 'mehp') & (oaps.context == 'base')]
mp240_apdtri=oaps.APDtri_240[(oaps.group == 'mehp') & (oaps.context == 'post')]

# Example Action Potentials
cp240vm=np.genfromtxt('ap_examples/20161221-ratb-04-pcl240-vm.csv', delimiter=',')
cp240t=np.genfromtxt('ap_examples/20161221-ratb-04-pcl240-t.csv', delimiter=',')

cp140vm=np.genfromtxt('ap_examples/20161221-rata-12-pcl140-ctrl-vm.csv', delimiter=',')
cp140t=np.genfromtxt('ap_examples/20161221-rata-12-pcl140-ctrl-t.csv', delimiter=',')

mp240vm=np.genfromtxt('ap_examples/20180720-rata-27-pcl240-vm.csv', delimiter=',')
mp240t=np.genfromtxt('ap_examples/20180720-rata-27-pcl240-t.csv', delimiter=',')

mp140vm=np.genfromtxt('ap_examples/20161222-ratb-13-pcl140-mehp-vm.csv', delimiter=',')
mp140t=np.genfromtxt('ap_examples/20161222-ratb-13-pcl140-mehp-t.csv', delimiter=',')

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)

fig= plt.figure(figsize=(11,5.5)) # half 8.5 x 11 inch page.

ax = fig.add_subplot(2, 4, 1)
plt.plot(cp140t,np.roll(cp140vm,78),color=timeColor,linewidth=2,label='Ctrl')
ax=plt.gca()
plt.plot(mp140t,np.roll(mp140vm,49)+0.03,color=timeColor,linewidth=2,linestyle='--',label='MEHP')
ax.tick_params(axis='x', which='both',bottom='on',top='off')
ax.tick_params(axis='y', which='both',right='off',left='on')
plt.xlim(xmin=0.8,xmax=1.1)
m=['0', '75', '150', '225', '300']
ticks=[0.8, 0.875, 0.95, 1.025, 1.1]
ax.set_xticks(ticks)
ax.set_xticklabels(m)
plt.ylim(ymin=0,ymax=1.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Counts',fontsize=12)
plt.xlabel('Time (ms)',fontsize=12)
plt.ylabel('Vm (Normalized)',fontsize=12)
plt.legend(bbox_to_anchor=(1.2, 1),loc='upper right',ncol=1, prop={'size':10},numpoints=1,frameon=False)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.4),yr*0.98,'A',ha='center',va='bottom',fontsize=16,fontweight='bold')
ax.set_title('PCL 140 ms',fontsize=12)

ax = fig.add_subplot(2, 4, 5)
plt.plot(cp240t,cp240vm,color=timeColor,linewidth=2,label='Ctrl')
ax=plt.gca()
plt.plot(mp240t,np.roll(mp240vm,110),color=timeColor,linewidth=2,linestyle='--',label='MEHP')
ax.tick_params(axis='x', which='both',bottom='on',top='off')
ax.tick_params(axis='y', which='both',right='off',left='on')
plt.xlim(xmin=0.9,xmax=1.2)
m=['0', '75', '150', '225', '300']
ticks=[0.9, 0.975, 1.05, 1.125, 1.2]
ax.set_xticks(ticks)
ax.set_xticklabels(m)
plt.ylim(ymin=0,ymax=1.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.ylabel('Counts',fontsize=12)
plt.xlabel('Time (ms)',fontsize=12)
plt.ylabel('Vm (Normalized)',fontsize=12)
plt.legend(bbox_to_anchor=(1.2, 0.9),loc='upper right',ncol=1, prop={'size':10},numpoints=1,frameon=False)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.4),yr*0.98,'E',ha='center',va='bottom',fontsize=16,fontweight='bold')
ax.set_title('PCL 240 ms',fontsize=12)

ax = fig.add_subplot(2, 4, 2)
width=0.28
b1=ax.bar(0.4-width/2,np.mean(cb140_apd30), width, color=baseColor,yerr=np.std(cb240_apd30)/np.sqrt(len(cb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
b2=ax.bar(0.7-width/2,np.mean(cp140_apd30),width, color=timeColor,yerr=np.std(cb240_apd30)/np.sqrt(len(cb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb140_apd30), width, color=baseColor,yerr=np.std(cb240_apd30)/np.sqrt(len(cb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp140_apd30), width, color=timeColor,yerr=np.std(cb240_apd30)/np.sqrt(len(cb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=30)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
labels=['Ctrl', 'MEHP']
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('APD30 (ms)',fontsize=12)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.41),yr*0.98,'B',ha='center',va='bottom',fontsize=16,fontweight='bold')
plt.legend((b1[0], b2[0]),('Baseline', '30 min'),bbox_to_anchor=(0.9, 1.2),loc='upper right',ncol=1, prop={'size':10},numpoints=1,frameon=False)

ax = fig.add_subplot(2, 4, 3)
ax.bar(0.4-width/2,np.mean(cb140_apd90), width, color=baseColor, yerr=np.std(cb140_apd90)/np.sqrt(len(cb140_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp140_apd90),width, color=timeColor, yerr=np.std(cp140_apd90)/np.sqrt(len(cp140_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb140_apd90), width, color=baseColor, yerr=np.std(mb140_apd90)/np.sqrt(len(mb140_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp140_apd90), width, color=timeColor, yerr=np.std(mp140_apd90)/np.sqrt(len(mp140_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=100)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
labels=['Ctrl', 'MEHP']
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('APD90 (ms)',fontsize=12)
ax.set_title('PCL 140 ms',fontsize=12)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.50),yr*0.96,'C',ha='center',va='bottom',fontsize=16,fontweight='bold')

ax = fig.add_subplot(2, 4, 4)
ax.bar(0.4-width/2,np.mean(cb140_apdtri), width,color=baseColor,yerr=np.std(cb140_apdtri)/np.sqrt(len(cb140_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp140_apdtri),width, color=timeColor,yerr=np.std(cp140_apdtri)/np.sqrt(len(cp140_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb140_apdtri), width, color=baseColor,yerr=np.std(mb140_apdtri)/np.sqrt(len(mb140_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp140_apdtri), width, color=timeColor,yerr=np.std(mp140_apdtri)/np.sqrt(len(mp140_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=80)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('AP Tri. (ms)',fontsize=12)
ax.set_title('PCL 140 ms',fontsize=12)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'D',ha='center',va='bottom',fontsize=16,fontweight='bold')

ax = fig.add_subplot(2, 4, 6)
ax.bar(0.4-width/2,np.mean(cb240_apd30), width,color=baseColor,yerr=np.std(cb240_apd30)/np.sqrt(len(cb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp240_apd30),width, color=timeColor,yerr=np.std(cp240_apd30)/np.sqrt(len(cp240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb240_apd30), width, color=baseColor,yerr=np.std(mb240_apd30)/np.sqrt(len(mb240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp240_apd30), width, color=timeColor,yerr=np.std(mp240_apd30)/np.sqrt(len(mp240_apd30)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=30)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('APD30 (ms)',fontsize=12)
ax.set_title('PCL 240 ms',fontsize=12)
#ax1.text(0.55,292,'*',ha='center',va='bottom',fontsize=20)
#ax1.plot([0.4, 0.7], [292, 292], "k-",linewidth=4)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'F',ha='center',va='bottom',fontsize=16,fontweight='bold')

ax = fig.add_subplot(2, 4, 7)
ax.bar(0.4-width/2,np.mean(cb240_apd90), width, color=baseColor,yerr=np.std(cb240_apd90)/np.sqrt(len(cb240_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(0.7-width/2,np.mean(cp240_apd90),width, color=timeColor,yerr=np.std(cp240_apd90)/np.sqrt(len(cp240_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.2-width/2,np.mean(mb240_apd90), width, color=baseColor,yerr=np.std(mb240_apd90)/np.sqrt(len(mb240_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
ax.bar(1.5-width/2,np.mean(mp240_apd90), width, color=timeColor,yerr=np.std(mp240_apd90)/np.sqrt(len(mp240_apd90)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1.0))
plt.ylim(ymin=0,ymax=120)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('APD90 (ms)',fontsize=12)
ax.set_title('PCL 240 ms',fontsize=12)
ax.text(1.2,111,'*',ha='center',va='center',fontsize=16)
ax.plot([1.2-width/2, 1.2+width/2], [101, 101], "k-",linewidth=2)
ax.plot([0.7-width/2, 1.2+width/2], [107, 107], "k-",linewidth=2)

yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.50),yr*0.96,'G',ha='center',va='bottom',fontsize=16,fontweight='bold')

ax = fig.add_subplot(2, 4, 8)
ax.bar(0.4-width/2,np.mean(cb240_apdtri), width, color=baseColor,yerr=np.std(cb240_apdtri)/np.sqrt(len(cb240_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1))
ax.bar(0.7-width/2,np.mean(cp240_apdtri),width, color=timeColor,yerr=np.std(cp240_apdtri)/np.sqrt(len(cp240_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1))
ax.bar(1.2-width/2,np.mean(mb240_apdtri), width, color=baseColor,yerr=np.std(mb240_apdtri)/np.sqrt(len(mb240_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1))
ax.bar(1.5-width/2,np.mean(mp240_apdtri), width, color=timeColor,yerr=np.std(mp240_apdtri)/np.sqrt(len(mp240_apdtri)),ecolor='k', error_kw=dict(lw=1, capsize=4, capthick=1))
plt.ylim(ymin=0,ymax=91)
plt.xlim(xmin=0,xmax=1.6)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks([0.4, 1.2], labels, rotation=0,fontsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('AP Tri. (ms)',fontsize=12)
ax.set_title('PCL 240 ms',fontsize=12)
ax.text(1.2,84,'*',ha='center',va='center',fontsize=16)
ax.plot([1.2-width/2, 1.2+width/2], [76, 76], "k-",linewidth=2)
ax.plot([0.7-width/2, 1.2+width/2], [80, 80], "k-",linewidth=2)
yl=ax.get_ylim()
yr=yl[1]-yl[0]
xl=ax.get_xlim()
xr=xl[1]-xl[0]
ax.text(xl[0]-(xr*0.33),yr*0.96,'H',ha='center',va='bottom',fontsize=16,fontweight='bold')

#h_pad = height between edges of adjacent subplots, w_pad width between edges of adjacent subplots
#pad = padding between the figure edge and the edges of subplots, as a fraction of the font-size
plt.rcParams.update({'font.size': 12})
plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=2)
#plt.savefig('MEHP_OAPS_APD.png', dpi=300)
#plt.savefig('MEHP_OAPS_APD.pdf')