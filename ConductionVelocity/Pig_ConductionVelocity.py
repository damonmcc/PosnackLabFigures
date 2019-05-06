import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.patches import Circle, Wedge
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



def plot_heart(axis, heart_image):
    axis.axis('off')
    axis.imshow(heart_image, cmap='bone')

def example_ActMap_Vm(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    # ros_wedge = Wedge((ros['x'], ros['y']), ros['r'], 225, 45, fc='none', ec='k', lw=2)
    # axis.add_artist(ros_wedge)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Vm Title', fontsize=14)
    return img


def example_ActMap_Ca(axis, actMap):
    axis.axis('off')
    roi_circle = Circle((roi['x'], roi['y']), roi['r'], fc='none', ec='k', lw=2)
    axis.add_artist(roi_circle)
    img = axis.imshow(actMap, norm=jet_norm, cmap="jet")
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    axis.set_title('Ca Title', fontsize=14)
    return img


def example_Coupling(axis, Vm, Ca):
    act_end = int(2.5 * 1000)   # seconds to ms

    stim_Vm = int(Vm['act_start'] * 1000)
    data_Vm = Vm['data'][:, 1]
    # data_Vm = np.interp(data_Vm, (data_Vm.min(), data_Vm.max()), (0, 1))
    stim_Ca = int(Ca['act_start'] * 1000)
    data_Ca = -1 + Ca['data'][:, 1]
    # data_Ca = np.interp(data_Ca, (data_Ca.min(), data_Ca.max()), (0, 1))

    times_Vm = (Vm['data'][:, 0]) * 1000  # seconds to ms
    # times_Vm[:] = [x - stim_Vm for x in times_Vm]
    times_Ca = (Ca['data'][:, 0]) * 1000  # seconds to ms
    # times_Ca[:] = [x - stim_Ca for x in times_Ca]
    time_window = 500

    axis.set_xlabel('Time (ms)', fontsize=12)
    axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    axis.set_xlim([stim_Vm, stim_Vm + time_window])
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(10))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    axis.set_ylabel('Normalized Fluorescence', fontsize=12)
    axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    axis.set_ylim([-1.1, 1.1])
    axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.plot(times_Vm, data_Vm,
              color=context_colors[0], linewidth=2, label='Base')
    axis.plot(times_Ca, data_Ca,
              color=context_colors[1], linewidth=2, label='Post')


def example_plot(axis):
    axis.plot([1, 2])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


# Build figure
fig = plt.figure(figsize=(8, 8))  # _ x _ inch page
gs0 = fig.add_gridspec(1, 2)  # Overall: ? row, ? columns

# Build heart section
gsHeart = gs0[0].subgridspec(2, 1, hspace=0.3)
axImage = fig.add_subplot(gsHeart[0])

# Build Activation Map section
gsActMaps = gsHeart[1].subgridspec(2, 2, hspace=0.3)  # 2 rows, 2 columns for Activation Maps
axActMapsVm = fig.add_subplot(gsActMaps[0]), fig.add_subplot(gsActMaps[2])
# axActMapsVm[0].set_ylabel('PCL 250 ms')
# axActMapsVm[1].set_ylabel('PCL 350 ms')
axActMapsCa = fig.add_subplot(gsActMaps[1]), fig.add_subplot(gsActMaps[3])
# Create region of interest (ROI)
# X and Y flipped and subtracted from W and H, due to image rotation
roi = {'y': 640-155,
       'x': 512-255,
       'r': 20}
# Create Region of Stimulation (ROS)
ros = {'y': 640-140,
       'x': 512-360,
       'r': 20}

# Build Traces section
gsTraces = gs0[1].subgridspec(2, 1, hspace=0.3)  # 2 rows, 1 column for Activation Maps
axTraces = fig.add_subplot(gsTraces[0]), fig.add_subplot(gsTraces[1])

# Build Activation Map section
ActMapTitleX = 0.1
ActMapTitleY = 1


# Import heart image
heart = np.rot90(plt.imread('data/20190322-piga/01-350_RH237_0001.tif'))

# Import Activation Maps
actMapsVm = {250: np.fliplr(np.rot90(np.loadtxt('data/20190322-piga/ActMaps/ActMap-11-250_RH237.csv',
                                                delimiter=',', skiprows=0))),
             350: np.fliplr(np.rot90(np.loadtxt('data/20190322-piga/ActMaps/ActMap-01-350_RH237.csv',
                                                delimiter=',', skiprows=0)))}
actMapsCa = {250: np.fliplr(np.rot90(np.loadtxt('data/20190322-piga/ActMaps/ActMap-11-250_Rhod-2.csv',
                                                delimiter=',', skiprows=0))),
             350: np.fliplr(np.rot90(np.loadtxt('data/20190322-piga/ActMaps/ActMap-01-350_Rhod-2.csv',
                                                delimiter=',', skiprows=0)))}
# Determine max value across all activation maps
actMapMax = 0
print('Activation Map max values:')
for key, value in actMapsVm.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
for key, value in actMapsCa.items():
    print(np.nanmax(value))
    actMapMax = max(actMapMax, np.nanmax(value))
# Create normalization range for all activation maps
print('Activation Maps max value: ', actMapMax)
jet_norm = colors.Normalize(vmin=0, vmax=actMapMax)


# Import Traces
# TODO change to allow reference by PCL, like ActMapsVm and actMapsCa
# Load signal data, columns: time (s), fluorescence (norm)
# Pair with stim activation time used for activation maps
TraceVm_250 = {'data': np.genfromtxt('data/20190322-piga/ActMaps/Signals-11-250_RH237.csv',
                                              delimiter=','),
                        'act_start': 0.78}
TraceVm_350 = {'data': np.genfromtxt('data/20190322-piga/ActMaps/Signals-01-350_RH237.csv',
                                              delimiter=','),
                        'act_start': 0.78}
TraceCa_250 = {'data': np.genfromtxt('data/20190322-piga/ActMaps/Signals-11-250_Rhod-2.csv',
                                              delimiter=','),
                        'act_start': 0.78}
TraceCa_350 = {'data': np.genfromtxt('data/20190322-piga/ActMaps/Signals-01-350_Rhod-2.csv',
                                              delimiter=','),
                        'act_start': 0.78}


# Plot heart image
plot_heart(axis=axImage, heart_image=heart)
# Plot activation maps
example_ActMap_Vm(axis=axActMapsVm[0], actMap=actMapsVm[350])
example_ActMap_Ca(axis=axActMapsCa[0], actMap=actMapsCa[350])

example_ActMap_Vm(axis=axActMapsVm[1], actMap=actMapsVm[250])
example_ActMap_Ca(axis=axActMapsCa[1], actMap=actMapsCa[250])

# Plot Traces
example_Coupling(axis=axTraces[0], Vm=TraceVm_350, Ca=TraceCa_350)
example_Coupling(axis=axTraces[1], Vm=TraceVm_250, Ca=TraceCa_250)

# Fill rest with example plots
# example_plot(axImage)
# example_plot(fig.add_subplot(gsActMaps[0]))
# example_plot(fig.add_subplot(gsActMaps[1]))
# example_plot(fig.add_subplot(gsActMaps[2]))
# example_plot(fig.add_subplot(gsActMaps[3]))

# example_plot(axTracesSlow)
# example_plot(axTracesFast)


# Show and save figure
fig.show()
fig.savefig('Pig_CV.svg')
