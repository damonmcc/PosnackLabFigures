#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from decimal import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

fig = plt.figure(figsize=(8, 5))  # _ x _ inch page

# Common parameters
baseColor = 'indianred'
timeColor = 'midnightblue'

# colorBase = 'indianred'
# colorPost = 'midnightblue'
# colorsBP = [colorBase, colorPost]
colorCTRL = 'lightgrey'
colorMEHP = 'grey'
fontSize_TraceYLabel = 8


def plot_Traces(axis, data, time_start=0.0, idx_start=None, time_window=None, time_end=None, idx_end=None):
    # Setup data and time (ms) variables
    data_Vm = []
    traces_count = data.shape[1] - 1  # Number of columns after time column
    times = (data[:, 0])  # ms
    time_start, time_window = time_start * 1000, time_window * 1000  # seconds to ms
    # Find index of first value after start time
    for idx, time in enumerate(times):
        if time < time_start:
            pass
        else:
            idx_start = idx
            break

    if time_window:
        # Find index of first value after end time
        for idx, time in enumerate(times):
            if time < time_start + time_window:
                pass
            else:
                idx_end = idx
                break
        # Convert possibly strange floats to rounded Decimals
        time_start, time_window = Decimal(time_start).quantize(Decimal('.001'), rounding=ROUND_UP), \
                                  Decimal(time_window).quantize(Decimal('.001'), rounding=ROUND_UP)
        # Slice time array based on indices
        times = times[idx_start:idx_end]
        time_end = time_start + time_window
        axis.set_xlim([float(time_start), float(time_end)])

    trace = data[:, 1]
    # if time_window:
    #     trace = trace[idx_start:idx_end]
    # Normalize each trace
    data_min, data_max = np.nanmin(trace), np.nanmax(trace)
    print('*** Normalizing')
    print('** from min-max ', data_min, '-', data_max)
    # print('** to min-max ', data_min, '-', data_max)
    trace = np.interp(trace, (data_min, data_max), (0, 1))
    # data_Vm.append(trace)

    # Prepare axis for plotting
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.tick_params(axis='x', which='both', direction='in', bottom=True, top=False)
    # axis.tick_params(axis='y', which='both', direction='in', right=False, left=True)
    # axis.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(20))

    axis.set_ylim([0, 1.1])

    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # axis.set_xlabel('Time (ms)', fontsize=12)
    # axis.set_ylabel('Norm. Fluor., Vm', fontsize=12)
    # axis.set_title('Normalized Fluorescence ', fontsize=12)

    # Scale: Time and Normalized Voltage bars forming an L
    ECGScaleTime = [1000, 250 / 1000]  # 1 s, 0.2 Normalized V.
    ECGScaleOrigin = [axis.get_xlim()[1] - 1 * ECGScaleTime[0], axis.get_ylim()[0] + 0.01]
    # ECGScaleOrigin = [axis.get_xlim()[1] - 20, axis.get_ylim()[0] + 0.3]
    ScaleOriginPad = [2, 0.05]
    # Time scale bar
    axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0] + ECGScaleTime[0]],
              [ECGScaleOrigin[1], ECGScaleOrigin[1]],
              "k-", linewidth=1)
    axis.text(ECGScaleOrigin[0], ECGScaleOrigin[1] - ScaleOriginPad[1],
              str(ECGScaleTime[0] / 1000) + ' s',
              ha='left', va='top', fontsize=6, fontweight='bold')
    # # Voltage scale bar
    # axis.plot([ECGScaleOrigin[0], ECGScaleOrigin[0]],
    #           [ECGScaleOrigin[1], ECGScaleOrigin[1] + ECGScaleTime[1]], "k-", linewidth=1)
    # axis.text(ECGScaleOrigin[0] - ScaleOriginPad[0], ECGScaleOrigin[1],
    #           str(ECGScaleTime[1]),
    #           ha='right', va='bottom', fontsize=6, fontweight='bold')

    # Plot the trace
    axis.plot(data[:, 0], trace,
              color=baseColor, linewidth=1, label='TraceLabel')
    # axis.plot(times, trace,
    #           color=baseColor, linewidth=2, label='Base')


def example_plot(axis):
    axis.plot([1, 2])
    axis.set_xticks([])
    axis.set_xticklabels([])
    axis.set_yticks([])
    axis.set_yticklabels([])
    # axis.set_xlabel('x-label', fontsize=12)
    # axis.set_ylabel('y-label', fontsize=12)
    # axis.set_title('Title', fontsize=14)


# Setup grids and axes
gs0 = fig.add_gridspec(2, 2)  # Overall: Two rows, 2 columns

gs1 = gs0[0].subgridspec(2, 1, hspace=0.3)  # 2 rows for ECG traces
gs2 = gs0[1].subgridspec(2, 1, hspace=0.3)  # 2 rows for ECG traces
gs3 = gs0[2].subgridspec(2, 1, hspace=0.3)  # 2 rows for ECG traces
gs4 = gs0[3].subgridspec(2, 1, hspace=0.3)  # 2 rows for ECG traces

axECG_SNRT1 = fig.add_subplot(gs1[0])
axECG_SNRT2 = fig.add_subplot(gs1[1])
axECG_SNRT1.set_title('SNRT?')

axECG_VERP1 = fig.add_subplot(gs2[0])
axECG_VERP2 = fig.add_subplot(gs2[1])
axECG_VERP1.set_title('VERP')
axECG_VERP1.set_ylabel('S1 450 : S1 300\nCapture', fontsize=fontSize_TraceYLabel)
axECG_VERP2.set_ylabel('S1 450 : S1 250\nNo Capture', fontsize=fontSize_TraceYLabel)

axECG_WBCL1 = fig.add_subplot(gs3[0])
axECG_WBCL2 = fig.add_subplot(gs3[1])
axECG_WBCL1.set_title('WBCL')
axECG_WBCL1.set_ylabel('S1 250\nCapture', fontsize=fontSize_TraceYLabel)
axECG_WBCL2.set_ylabel('S1 205\nNo Capture', fontsize=fontSize_TraceYLabel)

axECG_AVNERP1 = fig.add_subplot(gs4[0])
axECG_AVNERP2 = fig.add_subplot(gs4[1])
axECG_AVNERP1.set_title('AVNERP')
axECG_AVNERP1.set_ylabel('S1 450 : S1 200\nCapture', fontsize=fontSize_TraceYLabel)
axECG_AVNERP2.set_ylabel('S1 450 : S1 199\nNo Capture', fontsize=fontSize_TraceYLabel)


# Import ECG Traces

# Skip header and import columns: times (ms), ECG (mV)
ECG_VERP1 = np.genfromtxt('data/20190517-pigA_LV2.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_VERP2 = np.genfromtxt('data/20190517-pigA_LV3.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)

ECG_WBCL1 = np.genfromtxt('data/20190517-pigA_RA1.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_WBCL2 = np.genfromtxt('data/20190517-pigA_RA3.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)

ECG_AVNERP1 = np.genfromtxt('data/20190517-pigA_RA5.txt',
                            skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_AVNERP2 = np.genfromtxt('data/20190517-pigA_RA6.txt',
                            skip_header=31, usecols=(0, 1), skip_footer=2)

# Plot Traces
timeWindow = 6
plot_Traces(axis=axECG_VERP1, data=ECG_VERP1,
            time_start=1, time_window=timeWindow)
plot_Traces(axis=axECG_VERP2, data=ECG_VERP2,
            time_start=2.5, time_window=timeWindow)

plot_Traces(axis=axECG_WBCL1, data=ECG_WBCL1,
            time_start=2.2, time_window=timeWindow)
plot_Traces(axis=axECG_WBCL2, data=ECG_WBCL2,
            time_start=1.1, time_window=timeWindow)

plot_Traces(axis=axECG_AVNERP1, data=ECG_AVNERP1,
            time_start=2.5, time_window=timeWindow)
plot_Traces(axis=axECG_AVNERP2, data=ECG_AVNERP2,
            time_start=1.6, time_window=timeWindow)

# Fill rest with example plots
example_plot(fig.add_subplot(gs1[0]))
example_plot(fig.add_subplot(gs1[1]))
# example_plot(fig.add_subplot(gs3[0]))
# example_plot(fig.add_subplot(gs3[1]))
# example_plot(fig.add_subplot(gs4[0]))
# example_plot(fig.add_subplot(gs4[1]))

# example_plot(axECGVERP1)
# example_plot(ECGVERP2)


# Show and save figure
fig.show()
fig.savefig('Pig_CV.svg')
