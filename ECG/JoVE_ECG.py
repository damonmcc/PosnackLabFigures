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
colorPace = 'midnightblue'
colorCapture = 'indianred'

# colorBase = 'indianred'
# colorPost = 'midnightblue'
# colorsBP = [colorBase, colorPost]
colorCTRL = 'lightgrey'
colorTace = 'grey'
fontsize_TraceTitle = 10
fontSize_TraceYLabel = 8
fontsize_PaceLabel = 5
# Scale: Time and Normalized Voltage bars forming an L
ECGScaleTime = [500, 250 / 1000]  # 0.5 s, 0.2 Normalized V.


def plot_Traces(axis, data, time_start=0.0, idx_start=None, time_window=None, scale=None):
    # Setup data and time (ms) variables
    # traces_count = data.shape[1] - 1  # Number of columns after time column
    times = (data[:, 0])  # ms
    trace = data[:, 1]
    idx_end = len(trace - 1)
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
        time_start_dec = Decimal(time_start).quantize(Decimal('.001'), rounding=ROUND_UP)
        time_window_dec = Decimal(time_window).quantize(Decimal('.001'), rounding=ROUND_UP)
        # Slice time array based on indices
        # times = times[idx_start:idx_end]
        time_end = time_start_dec + time_window_dec

        # Limit x-axis to times of idx_start and (idx_start + time_window)
        axis.set_xlim([float(time_start_dec), float(time_end)])

    # if time_window:
    #     trace = trace[idx_start:idx_end]
    # Normalize each trace
    data_min, data_max = np.nanmin(trace[idx_start:idx_end]), np.nanmax(trace[idx_start:idx_end])
    print('*** Normalizing')
    print('** from min-max ', data_min, '-', data_max)
    # print('** to min-max ', data_min, '-', data_max)
    trace = np.interp(trace, (data_min, data_max), (0, 0.9))
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

    # axis.set_ylim([0.3, 1.1])

    # axis.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    # axis.set_xlabel('Time (ms)', fontsize=12)
    # axis.set_ylabel('Norm. Fluor., Vm', fontsize=12)
    # axis.set_title('Normalized Fluorescence ', fontsize=12)

    # Time scale bar
    if scale:
        ecg_scale_origin = [axis.get_xlim()[1] - 1 * ECGScaleTime[0], axis.get_ylim()[0] + 0.01]
        # ecg_scale_origin = [axis.get_xlim()[1] - 20, axis.get_ylim()[0] + 0.3]
        scale_origin_pad = [2, 0.05]
        axis.plot([ecg_scale_origin[0], ecg_scale_origin[0] + ECGScaleTime[0]],
                  [ecg_scale_origin[1], ecg_scale_origin[1]],
                  "k-", linewidth=1)
        axis.text(ecg_scale_origin[0], ecg_scale_origin[1] - scale_origin_pad[1],
                  str(ECGScaleTime[0]) + ' ms',
                  ha='left', va='top', fontsize=6, fontweight='bold')
        # # Voltage scale bar
        # axis.plot([ecg_scale_origin[0], ecg_scale_origin[0]],
        #           [ecg_scale_origin[1], ecg_scale_origin[1] + ECGScaleTime[1]], "k-", linewidth=1)
        # axis.text(ecg_scale_origin[0] - scale_origin_pad[0], ecg_scale_origin[1],
        #           str(ECGScaleTime[1]),
        #           ha='right', va='bottom', fontsize=6, fontweight='bold')

    # Plot the trace
    axis.plot(data[:, 0], trace,
              color=colorTace, linewidth=1, label='TraceLabel')
    # axis.plot(times, trace,
    #           color=baseColor, linewidth=2, label='Base')


def plot_paces(axis, data, s1_num, s1_pcl, s2_pcl=None, time_start=0.0, time_window=None,
               site=None, capture=None):
    times = (data[:, 0])  # ms
    trace = data[:, 1]
    idx_start = 0
    idx_end = len(trace - 1)
    time_start = time_start * 1000  # seconds to ms
    time_end = times[-1]

    # Find index of first time value after first pace
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
                time_end = times[idx_end]
                break
    # Normalize each trace
    data_min, data_max = np.nanmin(trace[idx_start:idx_end]), np.nanmax(trace[idx_start:idx_end])
    print('*** Normalizing')
    print('** from min-max ', data_min, '-', data_max)
    # print('** to min-max ', data_min, '-', data_max)
    trace_norm = np.interp(trace, (data_min, data_max), (0, 0.9))

    # Draw arrows to show S1-S1 pacing spikes
    pace_start = time_start
    pace_end = pace_start + (s1_num * s1_pcl)
    # paceArrowL = -0.12
    paceArrowL = -0.04
    paceArrowHeadL = 0.05
    paceArrowX = np.linspace(pace_start, pace_end, num=s1_num, endpoint=False)
    # paceArrowY = trace_norm[idx_start] - paceArrowL + (2 * paceArrowHeadL)
    paceArrowY = 1
    paceArrowHeadW = (ECGScaleTime[0] * time_window) / 80
    [axis.arrow(x, paceArrowY, 0, paceArrowL,
                head_width=paceArrowHeadW, head_length=paceArrowHeadL,
                fc=colorPace, ec=colorPace) for x in paceArrowX]

    paceLabel_Y = 1.1

    if site is 'LV':
        capture_gap = 100
    else:
        capture_gap = 250

    if not s2_pcl:
        axis.text(0, paceLabel_Y, 'S1-S1: ' + str(s1_pcl) + ' ms', ha='left',
                  fontsize=fontsize_PaceLabel, transform=axis.transAxes)
        if capture is not None:
            if capture:
                capture_text = 'C'
            else:
                capture_text = 'NC'

            capture_x = paceArrowX[-1] + capture_gap
            capture_x_frac = (capture_x - axis.get_xlim()[0]) / (time_window * 1000)
            capture_y = 0.95
            capture_y_frac = (capture_y - axis.get_ylim()[0]) / (axis.get_ylim()[1] - axis.get_ylim()[0])
            axis.text(capture_x_frac, capture_y_frac, capture_text, ha='center',
                      fontsize=fontsize_PaceLabel, transform=axis.transAxes)
            axis.arrow(capture_x, 1, 0, paceArrowL,
                       head_width=paceArrowHeadW, head_length=paceArrowHeadL,
                       fc=colorCapture, ec=colorCapture)

    else:
        # Draw arrow to show S1-S2 pacing spike
        pace_start = paceArrowX[-1] + s2_pcl
        idx_pace = 0
        # Find index of first value after last pace time
        for idx, time in enumerate(times):
            if time < pace_start:
                pass
            else:
                idx_pace = idx
                break
        axis.text(0, paceLabel_Y, 'S1-S1: ' + str(s1_pcl) + ' ms\n'
                                                            'S1-S2: ' + str(s2_pcl) + ' ms',
                  ha='left', fontsize=fontsize_PaceLabel, transform=axis.transAxes)

        # paceArrowY = trace_norm[idx_pace] - paceArrowL + (2 * paceArrowHeadL)
        paceArrowY = 1
        axis.arrow(pace_start, paceArrowY, 0, paceArrowL,
                   head_width=paceArrowHeadW, head_length=paceArrowHeadL,
                   fc=colorPace, ec=colorPace)

        if capture is not None:
            if capture:
                capture_text = 'C'
            else:
                capture_text = 'NC'

            capture_x = pace_start + capture_gap
            capture_x_frac = (capture_x - axis.get_xlim()[0]) / (time_window * 1000)
            capture_y = 0.95
            capture_y_frac = (capture_y - axis.get_ylim()[0]) / (axis.get_ylim()[1] - axis.get_ylim()[0])
            axis.text(capture_x_frac, capture_y_frac, capture_text, ha='center',
                      fontsize=fontsize_PaceLabel, transform=axis.transAxes)
            axis.arrow(capture_x, capture_y, 0, paceArrowL,
                       head_width=paceArrowHeadW, head_length=paceArrowHeadL,
                       fc=colorCapture, ec=colorCapture)



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

axECG_NSR1 = fig.add_subplot(gs1[0])
axECG_NSR1.set_title('Sinus Rhythm', fontsize=fontsize_TraceTitle)
axECG_NSR2 = fig.add_subplot(gs1[1])
axECG_NSR2.set_title('Epicardial Pacing', fontsize=fontsize_TraceTitle)

axECG_VERP1 = fig.add_subplot(gs2[0])
axECG_VERP2 = fig.add_subplot(gs2[1])
axECG_VERP1.set_title('VERP', fontsize=fontsize_TraceTitle)

axECG_WBCL1 = fig.add_subplot(gs3[0])
axECG_WBCL2 = fig.add_subplot(gs3[1])
axECG_WBCL1.set_title('WBCL', fontsize=fontsize_TraceTitle)

axECG_AVNERP1 = fig.add_subplot(gs4[0])
axECG_AVNERP2 = fig.add_subplot(gs4[1])
axECG_AVNERP1.set_title('AVNERP', fontsize=fontsize_TraceTitle)

# Import ECG Traces

# Skip header and import columns: times (ms), ECG (mV)
ECG_NSR1 = np.genfromtxt('data/20190517-pigA_NSR1.txt',
                         skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_NSR2 = np.genfromtxt('data/20190517-pigA_LV0.txt',
                         skip_header=31, usecols=(0, 1), skip_footer=2)

ECG_VERP1 = np.genfromtxt('data/20190517-pigA_LV2.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_VERP2 = np.genfromtxt('data/20190517-pigA_LV3.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)

ECG_WBCL1 = np.genfromtxt('data/20190517-pigA_RA2.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_WBCL2 = np.genfromtxt('data/20190517-pigA_RA3.txt',
                          skip_header=31, usecols=(0, 1), skip_footer=2)

ECG_AVNERP1 = np.genfromtxt('data/20190517-pigA_RA5.txt',
                            skip_header=31, usecols=(0, 1), skip_footer=2)
ECG_AVNERP2 = np.genfromtxt('data/20190517-pigA_RA6.txt',
                            skip_header=31, usecols=(0, 1), skip_footer=2)

# Plot Traces
# NSR/Pacing
timeWindow = 5
traceStart = 1
paceStart = 1
plot_Traces(axis=axECG_NSR1, data=ECG_NSR1,
            time_start=traceStart, time_window=timeWindow)

traceStart = 4.3
plot_Traces(axis=axECG_NSR2, data=ECG_NSR2,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_NSR2, data=ECG_NSR2,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=9, s1_pcl=400, site='LV', capture=True)

# VERP
timeWindow = 5
traceStart = 1.43
paceStart = 0.5
plot_Traces(axis=axECG_VERP1, data=ECG_VERP1,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_VERP1, data=ECG_VERP1,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=8, s1_pcl=450, s2_pcl=300, site='LV', capture=True)

traceStart = 1.96
plot_Traces(axis=axECG_VERP2, data=ECG_VERP2,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_VERP2, data=ECG_VERP2,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=8, s1_pcl=450, s2_pcl=250, site='LV', capture=False)

# WBCL
timeWindow = 5
traceStart = 2.8
paceStart = 1
plot_Traces(axis=axECG_WBCL1, data=ECG_WBCL1,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_WBCL1, data=ECG_WBCL1,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=6, s1_pcl=250, capture=True)

traceStart = 1.95
plot_Traces(axis=axECG_WBCL2, data=ECG_WBCL2,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_WBCL2, data=ECG_WBCL2,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=6, s1_pcl=205, capture=False)

# AVNERP
timeWindow = 5
traceStart = 2.43
paceStart = 0.5
plot_Traces(axis=axECG_AVNERP1, data=ECG_AVNERP1,
            time_start=traceStart, time_window=timeWindow)
plot_paces(axis=axECG_AVNERP1, data=ECG_AVNERP1,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=6, s1_pcl=450, s2_pcl=200, capture=True)

traceStart = 3.9
plot_Traces(axis=axECG_AVNERP2, data=ECG_AVNERP2,
            time_start=traceStart, time_window=timeWindow, scale=True)
plot_paces(axis=axECG_AVNERP2, data=ECG_AVNERP2,
           time_start=traceStart + paceStart, time_window=timeWindow,
           s1_num=6, s1_pcl=450, s2_pcl=199, capture=False)

# Fill rest with example plots
# example_plot(fig.add_subplot(gs1[0]))
# example_plot(fig.add_subplot(gs1[1]))
# example_plot(fig.add_subplot(gs3[0]))
# example_plot(fig.add_subplot(gs3[1]))
# example_plot(fig.add_subplot(gs4[0]))
# example_plot(fig.add_subplot(gs4[1]))
# example_plot(axECGVERP1)
# example_plot(ECGVERP2)


# Show and save figure
fig.show()
fig.savefig('JoVE_ECG.svg')
