import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.patches import Circle
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set(style="white", color_codes=True)
from sklearn.preprocessing import normalize
import matplotlib.lines as mlines
import pandas as pd
from scipy import stats

# Set Colors
colorBase = 'indianred'
colorPost = 'midnightblue'
colorsBP = [colorBase, colorPost]
colorCTRL = 'grey'
colorMEHP = 'black'
exposure_colors = ['black', 'red']
exposure_pal = sns.color_palette(exposure_colors)
context_colors = ['0.8', '0.45']
context_pal = sns.color_palette(context_colors)

# Load activation maps
actMapsCTRLpost = {140: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-33.csv',
                                                      delimiter=',', skiprows=0))),
                   190: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-28.csv',
                                                      delimiter=',', skiprows=0))),
                   240: np.fliplr(np.rot90(np.loadtxt('data/ActMap-20180619-rata-23.csv',
                                                      delimiter=',', skiprows=0)))}
actMapsMEHPbase = {140: np.loadtxt('data/ActMap-20180522-rata-15.csv',
                                   delimiter=',', skiprows=0),
                   190: np.loadtxt('data/ActMap-20180522-rata-10.csv',
                                   delimiter=',', skiprows=0),
                   240: np.loadtxt('data/ActMap-20180522-rata-05.csv',
                                   delimiter=',', skiprows=0)}
actMapsMEHPpost = {140: np.loadtxt('data/ActMap-20180522-rata-38.csv',
                                   delimiter=',', skiprows=0),
                   190: np.loadtxt('data/ActMap-20180522-rata-33.csv',
                                   delimiter=',', skiprows=0),
                   240: np.loadtxt('data/ActMap-20180522-rata-28.csv',
                                   delimiter=',', skiprows=0)}
# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for key, value in actMapsMEHPbase.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
for key, value in actMapsMEHPpost.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
# Create normalization range for all activation maps
jet_norm = colors.Normalize(vmin=0, vmax=actMapMax)

# Load signal data
VmMEHP_base140 = np.genfromtxt('data/Signals/Signal-20180522-rata-15x55y140.csv', delimiter=',')
VmtMEHP_base140 = 1000 * np.genfromtxt('data/Signals/Signal-20180522-rata-15x55y140times.csv', delimiter=',')
VmMEHP_post140 = np.genfromtxt('data/Signals/Signal-20180522-rata-38x71y86.csv', delimiter=',')
VmtMEHP_post140 = 1000 * np.genfromtxt('data/Signals/Signal-20180522-rata-38x71y86times.csv', delimiter=',')
# Load signal data, columns: time (s), fluorescence (norm)
# Pair with stim activation time used for activation maps
ConDelayMEHP_base140 = {'data': np.genfromtxt('data/Signals/20180522-rata-15x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.117}
ConDelayMEHP_post140 = {'data': np.genfromtxt('data/Signals/20180522-rata-38x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.153}
ConDelayMEHP_base190 = {'data': np.genfromtxt('data/Signals/20180522-rata-10x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.169}
ConDelayMEHP_post190 = {'data': np.genfromtxt('data/Signals/20180522-rata-33x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.322}
ConDelayMEHP_base240 = {'data': np.genfromtxt('data/Signals/20180522-rata-05x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.194}
ConDelayMEHP_post240 = {'data': np.genfromtxt('data/Signals/20180522-rata-28x214y212.csv',
                                              delimiter=','),
                        'act_start': 1.213}
# Load signal data for 3 PCLs, columns: time (s), fluorescence (norm)
ConDelayMEHP_base = {140: {'data': np.genfromtxt('data/Signals/20180522-rata-05x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.194},
                     190: {'data': np.genfromtxt('data/Signals/20180522-rata-10x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.169},
                     240: {'data': np.genfromtxt('data/Signals/20180522-rata-15x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.117}}
ConDelayMEHP_post = {140: {'data': np.genfromtxt('data/Signals/20180522-rata-28x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.213},
                     190: {'data': np.genfromtxt('data/Signals/20180522-rata-33x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.322},
                     240: {'data': np.genfromtxt('data/Signals/20180522-rata-38x226y191.csv',
                                                 delimiter=','),
                           'act_start': 1.153}}
# Create region of interest (ROI), point data: r = 7
roi = {'x': 226,
       'y': 191,
       'r': 7}

# Load and calculate Conduction Velocity max values
ConMax = pd.read_csv('data/APD_binned2.csv')



def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapBase(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMapPost(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ConductionDelay(axis, base, post):
    stim_base = int(base['act_start'] * 1000)
    data_base = base['data'][:, 1]
    data_base = np.interp(data_base, (data_base.min(), data_base.max()), (0, 1))
    stim_post = int(post['act_start'] * 1000)
    data_post = post['data'][:, 1]
    data_post = np.interp(data_post, (data_post.min(), data_post.max()), (0, 1))

    times_base = (base['data'][:, 0]) * 1000  # seconds to ms
    times_base[:] = [x - stim_base for x in times_base]
    times_post = (post['data'][:, 0]) * 1000  # seconds to ms
    times_post[:] = [x - stim_post for x in times_post]
    time_window = 50

    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xlim([0, time_window])
    axis.xaxis.set_major_locator(ticker.MultipleLocator(10))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    # axis.set_ylabel('Vm (Normalized)', fontsize=12)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_ylim([0, 1.1])
    axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.plot(times_base, data_base,
              color=context_colors[0], linewidth=2, label='Base')
    axis.plot(times_post, data_post,
              color=context_colors[1], linewidth=2, label='Post')


def example_ConductionMax(axis, data):
    axis.set_xlim([60, 290])
    axis.tick_params(axis='x', labelsize=10, which='both', direction='in')
    axis.xaxis.set_major_locator(ticker.MultipleLocator(50))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    axis.set_ylim([45, 81])
    axis.tick_params(axis='y', labelsize=10, which='both', direction='in')
    axis.yaxis.set_major_locator(ticker.MultipleLocator(5))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.set_ylabel('Max CV (cm/s)', fontsize=12)
    axis.set_xlabel('Pacing Cycle Length (ms)', fontsize=12)
    APD80CTRL_legend = mlines.Line2D([], [], color=colorCTRL, ls='solid', marker='o',
                                     markersize=8, label='Ctrl')
    APD80MEHP_legend = mlines.Line2D([], [], color=colorMEHP, ls='dotted', marker='s',
                                     markersize=8, label='MEHP')
    axis.legend(handles=[APD80CTRL_legend, APD80MEHP_legend], loc='upper left',
                numpoints=1, fontsize=12, frameon=False)
    axis.errorbar(data.BCL, data.Ctrl_mean,
                  yerr=np.array(data.Ctrl_std) / np.sqrt(3),
                  ls='solid', color=colorCTRL, marker='o', ms=6)
    axis.errorbar(data.BCL, data.MEHP_mean,
                  yerr=np.array(data.MEHP_std) / np.sqrt(3),
                  ls='dotted', color=colorMEHP, marker='s', ms=6)
    sns.relplot(x="timepoint", y="signal", kind="line", ci="sd",
                data=data, ax=axis, palette=context_colors);


fig = plt.figure(figsize=(8, 8))  # _ x _ inch page
gs0 = fig.add_gridspec(2, 2, height_ratios=[0.8, 0.2])  # Overall: ? row, ? columns
gsActMaps = gs0[0].subgridspec(3, 2, hspace=0.3)  # 3 rows, 2 columns for Activation Maps
gsConDealys = gs0[1].subgridspec(3, 2, hspace=0.3)  # 3 rows, 2 columns for Conduction Delays

# Build Activation Map plots
ActMapTitleX = 0.1
ActMapTitleY = 1
axActMap140base = fig.add_subplot(gsActMaps[0])
axActMap140post = fig.add_subplot(gsActMaps[1])
axActMap140post.text(ActMapTitleX, ActMapTitleY, 'PCL 140 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap140post.transAxes)
axActMap140base.set_title('Base', weight='semibold')
axActMap140post.set_title('Post', weight='semibold')
axActMap190base = fig.add_subplot(gsActMaps[2])
axActMap190post = fig.add_subplot(gsActMaps[3])
axActMap190post.text(ActMapTitleX, ActMapTitleY, 'PCL 190 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap190post.transAxes)
axActMap240base = fig.add_subplot(gsActMaps[4])
axActMap240post = fig.add_subplot(gsActMaps[5])
axActMap240post.text(ActMapTitleX, ActMapTitleY, 'PCL 240 ms', ha='right', va='bottom',
                     weight='semibold', transform=axActMap240post.transAxes)
axCVMAX = fig.add_subplot(gs0[2])
axActTimes = fig.add_subplot(gs0[3])

# Build Activation Map plots
example_ActMapPost(axActMap140base, actMapsMEHPbase[140])
example_ActMapPost(axActMap140post, actMapsMEHPpost[140])
example_ActMapPost(axActMap190base, actMapsMEHPbase[190])
example_ActMapPost(axActMap190post, actMapsMEHPpost[190])
example_ActMapPost(axActMap240base, actMapsMEHPbase[240])
example_ActMapPost(axActMap240post, actMapsMEHPpost[240])
# cb1 = plt.colorbar(actMapsMEHPbase[240], cax=axActMap240base, orientation="horizontal")
# # cb1 = fig.colorbar(im1, cax=None, cmap=im1.cmap, orientation='horizontal', anchor=(0.0, 0.5))
# cb1.set_label('Activation Time (ms)', fontsize=7)
# cb1.set_ticks(np.linspace(0, math.ceil(max1), num=10))

# Build Conduction Delay plots
axConDelays140 = fig.add_subplot(gsConDealys[0])
axConDelays190 = fig.add_subplot(gsConDealys[2])
axConDelays240 = fig.add_subplot(gsConDealys[4])
example_ConductionDelay(axConDelays140, ConDelayMEHP_base[140], ConDelayMEHP_post[140])
example_ConductionDelay(axConDelays190, ConDelayMEHP_base[190], ConDelayMEHP_post[190])
example_ConductionDelay(axConDelays240, ConDelayMEHP_base[240], ConDelayMEHP_post[240])
axConDelays140.legend(loc='upper left', ncol=1,
                      prop={'size': 8}, numpoints=1, frameon=False)



# Fill rest with example plots
example_plot(fig.add_subplot(gsConDealys[1]))
example_plot(fig.add_subplot(gsConDealys[3]))
example_plot(fig.add_subplot(gsConDealys[5]))
example_plot(axCVMAX)
example_plot(axActTimes)

fig.show()
fig.savefig('MEHP_CV.svg')
