import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
from scipy import stats
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

bc = 'indianred'
tc = 'midnightblue'

# df = pandas.read_csv('data/rhythm_score.csv')
# ctrl_base=df.corr_verp[(df.group == 'c') & (df.time == 'b')]
# ctrl_post=df.corr_verp[(df.group == 'c') & (df.time == 'p')]
# mehp_base=df.corr_verp[(df.group == 'm') & (df.time == 'b')]
# mehp_post=df.corr_verp[(df.group == 'm') & (df.time == 'p')]

c1_verp = np.array([81, 79])
c2_verp = np.array([82, 81])
c3_verp = np.array([80.5, 90])
c4_verp = np.array([79, 60])

m1_verp = np.array([71, 100])
m2_verp = np.array([90, 170])
m3_verp = np.array([70, 90])
m4_verp = np.array([69, 120])
m5_verp = np.array([81, 150])
m6_verp = np.array([80, 80])
m7_verp = np.array([79, 98])
m8_verp = np.array([66, 106])
m9_verp = np.array([87, 140])

# Compiling for averaged data
mehp_verp = np.vstack([m1_verp, m2_verp, m3_verp, m4_verp, m5_verp, m6_verp, m7_verp, m8_verp, m9_verp])
ctrl_verp = np.vstack([c1_verp, c2_verp, c3_verp, c4_verp])
# Mean MEHP VERP post
np.mean(mehp_verp[:, 1])
np.mean(ctrl_verp[:, 1])

time = np.array([1, 2])  # before and after

ls = 'solid'
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
fig = plt.figure(figsize=(10, 6))  # half 8.5 x 11 inch page.

ax = fig.add_subplot(2, 4, 1)

ax.plot(time, c1_verp, ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, c2_verp, ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, c3_verp, ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, c4_verp, ls=ls, color=bc, marker='o', ms=8, mfc='w')

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('VERP (ms)', fontsize=14)
ax.set_title('Control', fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
# labels=['Baseline', '30 min']
plt.xticks(time, [], rotation=45, fontsize=14)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.5), yr * 1.3, 'A', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax = fig.add_subplot(2, 4, 2)

ax.plot(time, m1_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m2_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m3_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m4_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m5_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m6_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m7_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m8_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, m9_verp, ls=ls, color=tc, marker='o', ms=8, mfc='w')

# quick stats
stats.ttest_rel(mehp_verp[0:10, 0], mehp_verp[0:10, 1], axis=0)
ax.text(1.5, 190, 'p=0.002', ha='center', va='center', fontsize=14)

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('none')
ax.set_title('60 $\mu$M MEHP', fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
# labels=['Baseline', '30 min']
plt.xticks(time, [], rotation=45, fontsize=14)

ax = fig.add_subplot(2, 4, 3)
averp = np.loadtxt('data/mehp_averp.csv', delimiter=',', skiprows=1, usecols=(2, 3))

ax.plot(time, averp[0], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[1], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[2], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[3], ls=ls, color=bc, marker='o', ms=8, mfc='w')

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('AVNERP (ms)', fontsize=14)
# ax.set_title('Control',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
# labels=['Baseline', '30 min']
plt.xticks(time, [], rotation=45, fontsize=14)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.5), yr * 1.3, 'B', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax = fig.add_subplot(2, 4, 4)

ax.plot(time, averp[4] + 1, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[5] + 2, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[6], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[7], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[8], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, averp[9], ls=ls, color=tc, marker='o', ms=8, mfc='w')

# quick stats
stats.ttest_rel(averp[4:10, 0], averp[4:10, 1], axis=0)
ax.text(1.5, 190, 'p=0.0008', ha='center', va='center', fontsize=14)
# ax.plot(time, [650, 650], "k-",linewidth=2)

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('none')
# ax.set_title('60 $\mu$M MEHP',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
# labels=['Baseline', '30 min']
plt.xticks(time, [], rotation=45, fontsize=14)

ax = fig.add_subplot(2, 4, 5)
wbcl = np.loadtxt('data/mehp_wbcl.csv', delimiter=',', skiprows=1, usecols=(2, 3))

ax.plot(time, wbcl[0], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[1], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[2], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[3] + 2, ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[4], ls=ls, color=bc, marker='o', ms=8, mfc='w')

# stats.ttest_rel(wbcl[0:5,0],wbcl[0:5,1],axis=0)
# ax.text(1.5,680,'p=0.25',ha='center',va='center',fontsize=14)
# ax.plot(time, [650, 650], "k-",linewidth=2)

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('WBCL (ms)', fontsize=14)
# ax.set_title('Control',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
labels = ['Baseline', '30 min']
plt.xticks(time, labels, rotation=45, fontsize=14)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.5), yr * 1.3, 'C', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax = fig.add_subplot(2, 4, 6)

ax.plot(time, wbcl[5], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[6], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[7] + 1, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[8] + 2, ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[9], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[10], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, wbcl[11], ls=ls, color=tc, marker='o', ms=8, mfc='w')

# quick stats
stats.ttest_rel(wbcl[5:12, 0], wbcl[5:12, 1], axis=0)
ax.text(1.5, 205, 'p=0.007', ha='center', va='center', fontsize=14)
# ax.plot(time, [210, 210], "k-",linewidth=2)

plt.ylim(ymin=50, ymax=200)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('none')
# ax.set_title('60 $\mu$M MEHP',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
labels = ['Baseline', '30 min']
plt.xticks(time, labels, rotation=45, fontsize=14)

####################
# Adding in the SNRT DATA now

snrt = np.loadtxt('data/mehp_snrt.csv', delimiter=',', skiprows=1, usecols=(2, 3))

ax = fig.add_subplot(2, 4, 7)

ax.plot(time, snrt[1], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, snrt[3], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, snrt[4], ls=ls, color=bc, marker='o', ms=8, mfc='w')
ax.plot(time, snrt[5], ls=ls, color=bc, marker='o', ms=8, mfc='w')

plt.ylim(ymin=200, ymax=620)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('SNRT (ms)', fontsize=14)
# ax.set_title('Control',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
labels = ['Baseline', '30 min']
plt.xticks(time, labels, rotation=45, fontsize=14)
yl = ax.get_ylim()
yr = yl[1] - yl[0]
xl = ax.get_xlim()
xr = xl[1] - xl[0]
ax.text(xl[0] - (xr * 0.5), yr * 1.45, 'D', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax = fig.add_subplot(2, 4, 8)

ax.plot(time, snrt[0], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, snrt[2], ls=ls, color=tc, marker='o', ms=8, mfc='w')
ax.plot(time, snrt[6], ls=ls, color=tc, marker='o', ms=8, mfc='w')

stats.ttest_rel(snrt[[0, 2, 6], 0], snrt[[0, 2, 6], 1], axis=0)

ax.text(1.5, 680, 'p=0.25', ha='center', va='center', fontsize=14)
# ax.plot(time, [650, 650], "k-",linewidth=2)

plt.ylim(ymin=200, ymax=660)
plt.xlim(xmin=0, xmax=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('none')
# ax.set_title('60 $\mu$M MEHP',fontsize=14)
# ax.set_xlabel('Pacing Cycle Length (ms)',fontsize=14)
labels = ['Baseline', '30 min']
plt.xticks(time, labels, rotation=45, fontsize=14)

plt.rcParams.update({'font.size': 12})
plt.tight_layout(pad=0, w_pad=0, h_pad=2)
# fig.savefig('VERP.png')
# fig.savefig('VERP.svg')
# fig.savefig('EP_study.svg')

fig.show()