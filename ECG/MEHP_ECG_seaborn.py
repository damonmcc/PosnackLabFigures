import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import datetime as dt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set(style="white", color_codes=True)
exposure_colors = ['black', 'red']
exposure_pal = sns.color_palette(exposure_colors)
context_colors = ['0.8', '0.45']
context_pal = sns.color_palette(context_colors)


def plot_example(axis):
    axis.plot([1, 2])
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


def plot_timeseries(axis, data, x, y):
    print("Length of time array: " + str(len(x)))
    x_seconds = []
    x_delta = []
    for i in range(len(x)):
        try:
            x_seconds.append(dt.datetime.strptime(x[i], '%M:%S.%f'))
        except ValueError:
            x_seconds.append(dt.datetime.strptime(x[i], '%H:%M:%S.%f'))
        print("From %s  to  %s" % (x[i], x_seconds[i]))
    for i in range(len(x_seconds)):
        x_delta.append((x_seconds[i]-t0).total_seconds())
        print("From %s  to  %s  with RR %s" % (x_seconds[i], x_delta[i], data.RR[i]))
    sns.swarmplot(x=x_delta, y=y, data=data, hue='context', palette=context_colors, ax=axis)
    # axis.set_xlim([min(x_seconds), max(x_seconds)])
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axis.get_legend().set_visible(False)
    sns.despine()


def plot_swarm_box(axis, data, x, y):
    sns.boxplot(x=x, y=y, data=data, hue='context', whis=np.inf,    # choice for whis?
                palette=context_colors, dodge=True, ax=axis)
    sns.swarmplot(x=x, y=y, data=data, hue='context',
                  size=2, color='0.2', dodge=True, ax=axis)
    axis.get_legend().set_visible(False)
    sns.despine()


# Creat figure and layout grids
fig = plt.figure(figsize=(8, 8))  # _ x _ inch page
gs0 = fig.add_gridspec(3, 1, height_ratios=[0.2, 0.3, 0.5])  # Overall: 3 rows, 1 column
gsHR = gs0[1].subgridspec(1, 4, hspace=1.8)  # 1 row, 4 columns for Hear Rate plots
gsQRS = gs0[2].subgridspec(2, 3, hspace=0.3)  # 2 rows, 3 columns for QRS plots

# Create all axes
axHR_HR = fig.add_subplot(gsHR[0])
axHR_HRBox = fig.add_subplot(gsHR[1])
axHR_HRV = fig.add_subplot(gsHR[2])
axHR_HRVBox = fig.add_subplot(gsHR[3])
# Load data
tips = sns.load_dataset('tips')
hrv = pd.read_csv('data/ecg_hrv.csv')
t0 = dt.datetime(1900, 1, 1)
# Plot data
plot_timeseries(axHR_HR, hrv, hrv.time, 'RR')
plot_swarm_box(axHR_HRBox, hrv, 'group', 'RR')
plot_swarm_box(axHR_HRVBox, hrv, 'group', 'SDNN')

for n in range(6):
    ax = fig.add_subplot(gsQRS[n])
    plot_example(ax)

print(stats.ttest_ind(hrv.SDNN[(hrv.group == 'mehp') & (hrv.context == 'post')],
                      hrv.SDNN[(hrv.group == 'ctrl') & (hrv.context == 'post')], axis=0))
fig.show()