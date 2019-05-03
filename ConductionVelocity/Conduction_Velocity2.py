import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rotImage import rotImage
from matplotlib.patches import Circle
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy import stats
import cv2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


plt.rcParams.update({'font.size': 9})
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10)
#fig= plt.figure(figsize=(8.5,11))
#fig= plt.figure(figsize=(8.5,5.5)) # Half page figure
fig = plt.figure(num=1,clear=True,figsize=(9,8)) #figsize(width, height)

baseColor='indianred'
timeColor='midnightblue'
jet_norm = colors.Normalize(vmin=0, vmax=20)


# Activation Maps
# Baseline PCL 140
data = np.loadtxt('ActMaps/ActMap-20180619-rata-14.csv',delimiter=',',skiprows=0)
ax = fig.add_subplot(3, 3, 1)
#rotData=rotImage(data,180)
#ax.set_ylim(360-180, 0)
rotData=cv2.flip(data,0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
im1 = plt.imshow(rotData[2:,:-2],norm=jet_norm,cmap="jet", aspect=1)
plt.title('Ctrl Base'), plt.xticks([]), plt.yticks([])
r1 = Circle((126, 256-160), 5, fc='none', ec='w', lw=2) #Draw ROI Circle
r2 = Circle((180, 256-128), 5, fc='none', ec='k', lw=2) #Draw ROI Circle
ax.text(140, 256-155,'R1',ha='left',va='bottom',fontsize=10, color='w',fontweight='bold') # stim point
ax.text(187, 256-123,'R2',ha='left',va='bottom',fontsize=10, color='k',fontweight='bold') # distal point
ax.add_artist(r1)
ax.add_artist(r2)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.01), yr * 0.01, 'A', ha='center', va='bottom', fontsize=16, fontweight='bold')

axins1 = inset_axes(ax,
                    width="90%",  # width = 100% of parent_bbox width
                    height="6%",  # height : 6%
                    loc=8,
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0
                    )

#cb1 = fig.colorbar(im1, cax=axins1, cmap=im1.cmap, orientation='horizontal', anchor=(0.0, 0.5))
cb1 = plt.colorbar(im1, cax=axins1, orientation="horizontal")
cb1.set_label('Activation Time (msec)', fontsize=10)


data = np.loadtxt('ActMaps/ActMap-20180619-rata-33.csv',delimiter=',',skiprows=0)
ax = fig.add_subplot(3, 3, 2)
#rotData=rotImage(data,180)
rotData=cv2.flip(data,0)
#ax.set_ylim(360-180, 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
im2 = plt.imshow(rotData[2:,:-2],norm=jet_norm,cmap="jet", aspect=1)
plt.title('30 min Ctrl'), plt.xticks([]), plt.yticks([])
r1 = Circle((126, 256-160), 5, fc='none', ec='w', lw=2) #Draw ROI Circle
r2 = Circle((180, 256-128), 5, fc='none', ec='k', lw=2) #Draw ROI Circle
ax.text(140, 256-155,'R1',ha='left',va='bottom',fontsize=10, color='w',fontweight='bold') # stim point
ax.text(187, 256-123,'R2',ha='left',va='bottom',fontsize=10, color='k',fontweight='bold') # distal point
ax.add_artist(r1)
ax.add_artist(r2)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.01), yr * 0.01, 'B', ha='center', va='bottom', fontsize=16, fontweight='bold')

r1=(126,173) # Pacing Site
r2=(180,128) # Distal Site
dist=np.sqrt((r2[0]-r1[0])**2+(r2[1]-r1[1])**2)

# Baseline PCL 200
data = np.loadtxt('ActMaps/ActMap-20180910-06-140.csv',delimiter=',',skiprows=0)
ax = fig.add_subplot(3, 3, 4)
#rotData=rotImage(data,180)
im3 = plt.imshow(data[::,:],norm=jet_norm,cmap="jet", aspect=1)
#ax.set_ylim(384-180, 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title('Pre-MEHP Base'), plt.xticks([]), plt.yticks([])
r1 = Circle((162, 168), 5, fc='none', ec='w', lw=2) #Draw ROI Circle
r2 = Circle((195, 107), 5, fc='none', ec='k', lw=2) #Draw ROI Circle
ax.add_artist(r1)
ax.add_artist(r2)
ax.text(166,175,'R1',ha='left',va='top',fontsize=10, color='white',fontweight='bold') # stim point
ax.text(199,112,'R2',ha='left',va='top',fontsize=10, color='k',fontweight='bold') # distal point
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.01), yr * 0.01, 'D', ha='center', va='bottom', fontsize=16, fontweight='bold')

data = np.loadtxt('ActMaps/ActMap-20180910-23-140.csv',delimiter=',',skiprows=0)
ax = fig.add_subplot(3, 3, 5)
rotData=rotImage(data,180)
plt.imshow(data[:,:],norm=jet_norm,cmap="jet", aspect=1)
#ax.set_ylim(360-180, 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title('30 min MEHP'), plt.xticks([]), plt.yticks([])
r1 = Circle((210, 170), 5, fc='none', ec='w', lw=2) #Draw ROI Circle
r2 = Circle((221, 101), 5, fc='none', ec='k', lw=2) #Draw ROI Circle
ax.add_artist(r1)
ax.add_artist(r2)
ax.text(215,175,'R1',ha='left',va='top',fontsize=10, color='w',fontweight='bold') # stim point
ax.text(226,106,'R2',ha='left',va='top',fontsize=10, color='k',fontweight='bold') # distal point
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.01), yr * 0.01, 'E', ha='center', va='bottom', fontsize=16, fontweight='bold')
#%%
# Example Traces
ctrl_vm_traces = np.loadtxt('Ctrl_Lang_CV_AP_Example.csv',delimiter=',',skiprows=4)
time=ctrl_vm_traces[:,0]

ax = fig.add_subplot(3,3,3)
plt.plot()
#fs=887
#fe=915
fs=660
fe=685
mmin=np.min(ctrl_vm_traces[fs:fe,1])
mmax=np.max(ctrl_vm_traces[fs:fe,1])
plt.plot((time[fs:fe]-time[fs])*1000-2,((ctrl_vm_traces[fs:fe,1])-mmin)/(mmax-mmin),color='gray', ls='-.', linewidth=2,label='Base R1')
mmin=np.min(ctrl_vm_traces[fs:fe,2])
mmax=np.max(ctrl_vm_traces[fs:fe,2])
plt.plot((time[fs:fe]-time[fs])*1000-2,((ctrl_vm_traces[fs:fe,2])-mmin)/(mmax-mmin),color='gray', linewidth=2,label='Base R2')
fs=80
fe=110
mmin=np.min(ctrl_vm_traces[fs:fe,3])
mmax=np.max(ctrl_vm_traces[fs:fe,3])
plt.plot((time[fs:fe]-time[fs])*1000-4,((ctrl_vm_traces[fs:fe,3])-mmin)/(mmax-mmin),color='k', ls=':', linewidth=2,label='30min R1')
mmin=np.min(ctrl_vm_traces[fs:fe,4])
mmax=np.max(ctrl_vm_traces[fs:fe,4])
plt.plot((time[fs:fe]-time[fs])*1000-4,((ctrl_vm_traces[fs:fe,4])-mmin)/(mmax-mmin),color='k', linewidth=2,label='30min R2')
ax=plt.gca()
ax.tick_params(axis='x', which='both',bottom=True,top=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_xlim(-5, 20)
plt.xlabel('Conduction Time (msec)',fontsize=11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend(loc='upper left',ncol=1, prop={'size':9},numpoints=1,frameon=False, bbox_to_anchor=(-0.21, 1))
ax.annotate("Pace", xy=(-0.1, 0.2), xytext=(-5, 0.4), arrowprops=dict(facecolor='black', shrink=0.05))
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.15), yr * 0.9, 'C', ha='center', va='bottom', fontsize=16, fontweight='bold')

mehp_vm_traces = np.loadtxt('MEHP_Lang_CV_AP_examples_Base.csv',delimiter=',',skiprows=4)
mehp_vm_traces2 = np.loadtxt('MEHP_Lang_CV_AP_examples2.csv',delimiter=',',skiprows=4)
mehp_vm_traces2_t = mehp_vm_traces2[0:,0]

ax = fig.add_subplot(3,3,6)
plt.plot()
fs=887
fe=915
mmin=np.min(mehp_vm_traces[fs:fe,1])
mmax=np.max(mehp_vm_traces[fs:fe,1])
plt.plot((time[fs:fe]-time[fs])*1000-7,((mehp_vm_traces[fs:fe,1])-mmin)/(mmax-mmin),color='gray', ls='-.', linewidth=2,label='Base R1')
mmin=np.min(mehp_vm_traces[fs:fe,2])
mmax=np.max(mehp_vm_traces[fs:fe,2])
plt.plot((time[fs:fe]-time[fs])*1000-7,((mehp_vm_traces[fs:fe,2])-mmin)/(mmax-mmin),color='gray', linewidth=2,label='Base R2')
fs=800
fe=880
mmin=np.min(mehp_vm_traces2[fs:fe,1])
mmax=np.max(mehp_vm_traces2[fs:fe,1])
plt.plot((mehp_vm_traces2_t[fs:fe]-mehp_vm_traces2_t[fs])*1000-14,((mehp_vm_traces2[fs:fe,1])-mmin)/(mmax-mmin),color='k', ls=':', linewidth=2,label='MEHP R1')
mmin=np.min(mehp_vm_traces2[fs:fe,2])
mmax=np.max(mehp_vm_traces2[fs:fe,2])
plt.plot((mehp_vm_traces2_t[fs:fe]-mehp_vm_traces2_t[fs])*1000-14,((mehp_vm_traces2[fs:fe,2])-mmin)/(mmax-mmin),color='k', linewidth=2,label='MEHP R2')
ax=plt.gca()
ax.tick_params(axis='x', which='both',bottom=True,top=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_xlim(-5, 20)
plt.xlabel('Conduction Time (msec)',fontsize=11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend(loc='upper left',ncol=1, prop={'size':9},numpoints=1,frameon=False, bbox_to_anchor=(-0.21, 1))
ax.annotate("Pace", xy=(-0.1, 0.1), xytext=(-5, 0.3), arrowprops=dict(facecolor='black', shrink=0.05))
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.15), yr * 0.9, 'F', ha='center', va='bottom', fontsize=16, fontweight='bold')
#%%
# Compiled Data for Bar Plots
cvs = pd.read_csv('MEHP_langendorff_cv.csv')
ctrlBase240=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'base') & (cvs.pcl == 240)]
ctrlBase140=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'base') & (cvs.pcl == 140)]
ctrlPost240=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'post') & (cvs.pcl == 240)]
ctrlPost140=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'post') & (cvs.pcl == 140)]

mehpBase240=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'base') & (cvs.pcl == 240)]
mehpBase140=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'base') & (cvs.pcl == 140)]
mehpPost240=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'post') & (cvs.pcl == 240)]
mehpPost140=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'post') & (cvs.pcl == 140)]

ctrlBase240sem=np.std(ctrlBase240)/np.sqrt(len(ctrlBase240));
ctrlBase140sem=np.std(ctrlBase140)/np.sqrt(len(ctrlBase140));
ctrlPost240sem=np.std(ctrlPost240)/np.sqrt(len(ctrlPost240));
ctrlPost140sem=np.std(ctrlPost140)/np.sqrt(len(ctrlPost140));

mehpBase240sem=np.std(mehpBase240)/np.sqrt(len(mehpBase240));
mehpBase140sem=np.std(mehpBase140)/np.sqrt(len(mehpBase140));
mehpPost240sem=np.std(mehpPost240)/np.sqrt(len(mehpPost240));
mehpPost140sem=np.std(mehpPost140)/np.sqrt(len(mehpPost140));

ax = fig.add_subplot(3, 3, 7)
width=0.35
ind = np.arange(2) # time axis has two components: baseline, and post/30min
base = ax.bar(ind - width/2, [np.mean(ctrlBase240), np.mean(mehpBase240)],
                                width, yerr=[ctrlBase240sem, mehpBase240sem], color=baseColor, 
                                error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Baseline')
post = ax.bar(ind + width/2, [np.mean(ctrlPost240), np.mean(mehpPost240)],
                                width, yerr=[ctrlPost240sem, mehpPost240sem], color=timeColor, 
                                error_kw=dict(lw=1, capsize=4, capthick=1.0), label='30 Min')
ax.set_xticks(ind)
ax.set_xticklabels(('Ctrl', 'MEHP'),fontsize=12)
#ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.4), ncol=1)
ax.set_ylabel('CV (cm/s) @ PCL=240 ms',fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, 1.5)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2), ncol=2)
ax.text(1,88,'* p < 0.05',ha='center',va='center',fontsize=10)
ax.plot((ind + width/2), [79, 79], "k-",linewidth=2)
ax.plot([(ind[1] - width/2), (ind[1] + width/2)], [83, 83], "k-",linewidth=2)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.3), yr * 0.97, 'G', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax = fig.add_subplot(3, 3, 8)
width=0.35
ind = np.arange(2) # time axis has two components: baseline, and post/30min
base = ax.bar(ind - width/2, [np.mean(ctrlBase140), np.mean(mehpBase140)],
                                width, yerr=[ctrlBase140sem, mehpBase140sem], color=baseColor, 
                                error_kw=dict(lw=1, capsize=4, capthick=1.0), label='Baseline')
post = ax.bar(ind + width/2, [np.mean(ctrlPost140), np.mean(mehpPost140)],
                                width, yerr=[ctrlPost140sem, mehpPost140sem], color=timeColor, 
                                error_kw=dict(lw=1, capsize=4, capthick=1.0), label='30 Min')
ax.set_xticks(ind)
ax.set_xticklabels(('Ctrl', 'MEHP'),fontsize=12)
ax.set_ylabel('CV (cm/s) @ PCL=140 ms',fontsize=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, 1.5)
ax.text(1,88,'* p < 0.05',ha='center',va='center',fontsize=10)
ax.plot((ind + width/2), [79, 79], "k-",linewidth=2)
ax.plot([(ind[1] - width/2), (ind[1] + width/2)], [83, 83], "k-",linewidth=2)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.3), yr * 0.97, 'H', ha='center', va='bottom', fontsize=16, fontweight='bold')

#h_pad = height between edges of adjacent subplots, w_pad width between edges of adjacent subplots
#pad = padding between the figure edge and the edges of subplots, as a fraction of the font-size
plt.tight_layout(pad=1, w_pad=1, h_pad=1)

#%% Comparative Statistics
stats.ttest_ind(ctrlPost240,mehpPost240,axis=0, nan_policy='omit') # significant
stats.ttest_ind(mehpBase240,mehpPost240,axis=0, nan_policy='omit') # significant
stats.ttest_ind(ctrlPost140,mehpPost140,axis=0, nan_policy='omit') # almost significant
stats.ttest_ind(mehpBase140,mehpPost140,axis=0, nan_policy='omit') # not significant
#%% Descriptive Statistics
np.mean(ctrlPost140)
np.std(ctrlPost140)/np.sqrt(len(ctrlPost140))

np.mean(mehpPost140)
np.std(mehpPost140)/np.sqrt(len(ctrlPost140))

np.mean(ctrlPost240)
np.std(ctrlPost240)/np.sqrt(len(ctrlPost240))

np.mean(mehpPost240)
np.std(mehpPost240)/np.sqrt(len(ctrlPost240))
#plt.savefig('MEHP_CV.svg')