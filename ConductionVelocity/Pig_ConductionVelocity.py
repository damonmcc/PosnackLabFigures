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
roi = {'x': 155,
       'y': 255,
       'r': 10}

# Load and calculate Conduction Velocity max values
# ConMax = pd.read_csv('data/APD_binned2.csv')

def plot_heart(axis, heart_image):
    axis.imshow(heart_image)

def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def example_ActMap_Vm(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)
    return img


def example_ActMap_Ca(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)
    return img


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
    axis.set_ylabel('Vm (Normalized)', fontsize=12)
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


# Build figure
fig = plt.figure(figsize=(8, 8))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 2)  # Overall: ? row, ? columns
# Build Heart section
gsHeart = gs0[0].subgridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.3)
axImage = fig.add_subplot(gsHeart[0])
gsActMaps = gsHeart[1].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Activation Maps
# Build Traces section
gsTraces = gs0[1].subgridspec(2, 1, hspace=0.3)  # 2 rows, 1 column for Activation Maps
axTracesSlow = fig.add_subplot(gsTraces[0])
axTracesFast = fig.add_subplot(gsTraces[1])



# Build Activation Map plots
ActMapTitleX = 0.1
ActMapTitleY = 1

# Fill rest with example plots
example_plot(axImage)
example_plot(fig.add_subplot(gsActMaps[0]))
example_plot(fig.add_subplot(gsActMaps[1]))
example_plot(fig.add_subplot(gsActMaps[2]))
example_plot(fig.add_subplot(gsActMaps[3]))

example_plot(axTracesSlow)
example_plot(axTracesFast)

# Show and save figure
fig.show()
fig.savefig('Pig_CV.svg')
