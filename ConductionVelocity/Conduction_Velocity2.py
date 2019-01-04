import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import PIL

from rotImage import rotImage
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 8})
plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
# Region of Interest coordinates, with 180 degree rotation in mind
imX = 384
imY = 256
imXLim = [0, imY]
imYLim = [imX-38, 40]

# imXLim = [20, imY-45]
# imYLim = [imX-58, 135]
roiX = 84  # Adjusted for image rotation
roiY = 300  # Adjusted for image rotation
roiR = 5
# fig= plt.figure(figsize=(8.5,11))
fig = plt.figure(figsize=(8.5, 5.5))  # Half page figure

# Activation Maps
data = np.loadtxt('ActMaps/ActMap-4_Left-5x5.csv', delimiter=',', skiprows=0)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)

heartLeft = np.transpose(plt.imread('ActMaps/4_Left.tif'))
heartRight = np.transpose(plt.imread('ActMaps/4_Right.tif'))
# rotHeart = np.transpose(heartLeft)

min1 = data.min()
max1 = data.max()
ax = fig.add_subplot(2, 4, 1)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)
plt.imshow(heartLeft, cmap="bone")
im1 = plt.imshow(rotData, cmap="jet")
norm1 = mpl.colors.Normalize(vmin=min1, vmax=max1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title('Vm, 5x5 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])

data = np.loadtxt('ActMaps/ActMap-4_Left-9x9.csv', delimiter=',', skiprows=0)
ax = fig.add_subplot(2, 4, 2)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)
plt.imshow(heartLeft, cmap="bone")
im2 = plt.imshow(rotData, cmap=im1.cmap)
# plt.imshow(rotData[180:,:],cmap="jet", aspect=1.3)
# ax.set_ylim(360-180, 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title('Vm, 9x9 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)


data = np.loadtxt('ActMaps/ActMap-4_Left-15x15.csv', delimiter=',', skiprows=0)
ax = fig.add_subplot(2, 4, 3)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)
plt.imshow(heartLeft, cmap="bone")
plt.imshow(rotData, cmap=im1.cmap)
# plt.imshow(rotData[180:,:],cmap="jet", aspect=1.3)
# ax.set_ylim(360-180, 0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.title('Vm, 15x15 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)


data = np.loadtxt('ActMaps/ActMap-4_Right-5x5.csv', delimiter=',', skiprows=0)
ax = fig.add_subplot(2, 4, 5)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)
plt.imshow(heartRight, cmap="bone")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.imshow(rotData, cmap=im1.cmap)
plt.title('Ca, 5x5 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)


data = np.loadtxt('ActMaps/ActMap-4_Right-9x9.csv', delimiter=',', skiprows=0)
ax = fig.add_subplot(2, 4, 6)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)
plt.imshow(heartRight, cmap="bone")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
im6 = plt.imshow(rotData, cmap=im1.cmap)
plt.title('Ca, 9x9 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)


data = np.loadtxt('ActMaps/ActMap-4_Right-15x15.csv', delimiter=',', skiprows=0)
ax = fig.add_subplot(2, 4, 7)
ax.set_xlim(imXLim)
ax.set_ylim(imYLim)
rotData = np.transpose(data)
rotData = np.ma.masked_where(rotData == 0, rotData)
plt.imshow(heartRight, cmap="bone")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.imshow(rotData, cmap=im1.cmap)
plt.title('Ca, 15x15 Box Blur', fontweight='semibold'), plt.xticks([]), plt.yticks([])
p = Circle((roiX, roiY), roiR, fc='none', ec='k', lw=2)  # Draw ROI Circle
ax.add_artist(p)

axins1 = inset_axes(ax,
                    width="90%",  # width = 100% of parent_bbox width
                    height="3%",  # height : 3%
                    loc=8,
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0
                    )
cb1 = plt.colorbar(im2, cax=axins1, orientation="horizontal")
# cb1 = fig.colorbar(im1, cax=None, cmap=im1.cmap, orientation='horizontal', anchor=(0.0, 0.5))
cb1.set_label('Activation Time (ms)', fontsize=7)
# cb1.set_ticks(np.linspace(0, math.ceil(max1), num=10))

# Signal Traces
time = np.loadtxt('VmTraces/Signals-4_Left-5x5times.csv', delimiter=',', skiprows=0)
vmTimeAdj = time - time.min()
vm5x5 = np.loadtxt('VmTraces/Signals-4_Left-5x5.csv', delimiter=',', skiprows=0)
vm9x9 = np.loadtxt('VmTraces/Signals-4_Left-9x9.csv', delimiter=',', skiprows=0)
vm15x15 = np.loadtxt('VmTraces/Signals-4_Left-15x15.csv', delimiter=',', skiprows=0)
time = np.loadtxt('CaTraces/Signals-4_Right-5x5times.csv', delimiter=',', skiprows=0)
caTimeAdj = time - time.min()
ca5x5 = np.loadtxt('CaTraces/Signals-4_Right-5x5.csv', delimiter=',', skiprows=0)
ca9x9 = np.loadtxt('CaTraces/Signals-4_Right-9x9.csv', delimiter=',', skiprows=0)
ca15x15 = np.loadtxt('CaTraces/Signals-4_Right-15x15.csv', delimiter=',', skiprows=0)

ax = fig.add_subplot(2, 4, 4)
plt.plot(vmTimeAdj *1000, ((vm5x5-np.min(vm5x5))/(np.max(vm5x5)-np.min(vm5x5))), color='r', linewidth=1, label='5x5')
plt.plot(vmTimeAdj *1000, ((vm9x9-np.min(vm9x9))/(np.max(vm9x9)-np.min(vm9x9))), color='g', linewidth=1, label='9x9')
plt.plot(vmTimeAdj *1000, ((vm15x15-np.min(vm15x15))/(np.max(vm15x15)-np.min(vm15x15))), color='b', linewidth=1, label='15x15')
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom='on', top=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.get_yaxis().set_visible(False)
plt.xlabel('Time (ms)', fontsize=8)
plt.ylabel('Vm Fluorescence @ ROI', fontsize=8, fontweight='semibold')
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
plt.legend(loc='upper right', ncol=1, prop={'size': 7}, numpoints=1, frameon=False)  # bbox_to_anchor=(-0.02, 1.05)

ax = fig.add_subplot(2, 4, 8)
plt.plot(caTimeAdj*1000, ca5x5, color='r', linewidth=1, label='5x5')
plt.plot(caTimeAdj*1000, ca9x9, color='g', linewidth=1, label='9x9')
plt.plot(caTimeAdj*1000, ca15x15, color='b', linewidth=1, label='15x15')
ax = plt.gca()
ax.tick_params(axis='x', which='both', bottom='on', top='off')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.get_yaxis().set_visible(False)
plt.xlabel('Time (ms)', fontsize=8)
# ax.set_ylabel(labelpad=0)
plt.ylabel('Ca Fluorescence @ ROI', fontsize=8, fontweight='semibold')
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
plt.legend(loc='upper right', ncol=1, prop={'size': 7}, numpoints=1, frameon=False)

# # Compiled Data for Bar Plots
# cvs = pd.read_csv('MEHP_langendorff_cv.csv')
# ctrlBase240=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'base') & (cvs.pcl == 240)]
# ctrlBase140=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'base') & (cvs.pcl == 140)]
# ctrlPost240=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'post') & (cvs.pcl == 240)]
# ctrlPost140=cvs.cv[(cvs.group == 'ctrl') & (cvs.context == 'post') & (cvs.pcl == 140)]
#
# mehpBase240=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'base') & (cvs.pcl == 240)]
# mehpBase140=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'base') & (cvs.pcl == 140)]
# mehpPost240=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'post') & (cvs.pcl == 240)]
# mehpPost140=cvs.cv[(cvs.group == 'mehp') & (cvs.context == 'post') & (cvs.pcl == 140)]
#
# ctrlBase240sem=np.std(ctrlBase240)/np.sqrt(len(ctrlBase240));
# ctrlBase140sem=np.std(ctrlBase140)/np.sqrt(len(ctrlBase140));
# ctrlPost240sem=np.std(ctrlPost240)/np.sqrt(len(ctrlPost240));
# ctrlPost140sem=np.std(ctrlPost140)/np.sqrt(len(ctrlPost140));
#
# mehpBase240sem=np.std(mehpBase240)/np.sqrt(len(mehpBase240));
# mehpBase140sem=np.std(mehpBase140)/np.sqrt(len(mehpBase140));
# mehpPost240sem=np.std(mehpPost240)/np.sqrt(len(mehpPost240));
# mehpPost140sem=np.std(mehpPost140)/np.sqrt(len(mehpPost140));
#
# ax = fig.add_subplot(2, 4, 4)
# width=0.35
# ind = np.arange(2) # time axis has two components: baseline, and post/30min
# base = ax.bar(ind - width/2, [np.mean(ctrlBase240), np.mean(mehpBase240)],
#                                 width, yerr=[ctrlBase240sem, mehpBase240sem], color='green', label='Baseline')
# post = ax.bar(ind + width/2, [np.mean(ctrlPost240), np.mean(mehpPost240)],
#                                 width, yerr=[ctrlPost240sem, mehpPost240sem], color='blue', label='30 Min')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Ctrl', 'MEHP'))
# ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.2), ncol=1)
# ax.set_ylabel('CV (cm/s) @ PCL=240 ms',fontsize=11)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim(0, 80)
# ax.set_xlim(-0.5, 1.5)
#
# ax = fig.add_subplot(2, 4, 8)
# width=0.35
# ind = np.arange(2) # time axis has two components: baseline, and post/30min
# base = ax.bar(ind - width/2, [np.mean(ctrlBase140), np.mean(mehpBase140)],
#                                 width, yerr=[ctrlBase140sem, mehpBase140sem], color='green', label='Baseline')
# post = ax.bar(ind + width/2, [np.mean(ctrlPost140), np.mean(mehpPost140)],
#                                 width, yerr=[ctrlPost140sem, mehpPost140sem], color='blue', label='30 Min')
# ax.set_xticks(ind)
# ax.set_xticklabels(('Ctrl', 'MEHP'))
# ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.2), ncol=1)
# ax.set_ylabel('CV (cm/s) @ PCL=140 ms',fontsize=11)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim(0, 80)
# ax.set_xlim(-0.5, 1.5)

# h_pad = height between edges of adjacent subplots, w_pad width between edges of adjacent subplots
# pad = padding between the figure edge and the edges of subplots, as a fraction of the font-size
plt.tight_layout(pad=0.5)
plt.savefig('BoxBlur_Isochrones.pdf')