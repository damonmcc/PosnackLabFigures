import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

fig, axes = plt.subplots(3, sharex=True)
xlim = [150, 450]
ylim = [0, 1]

fileVm240 = "data/20180803-rata/Voltage/04-240_Vm_15x15.csv"
fileCa240 = "data/20180803-rata/Calcium/04-240_Ca_15x15.csv"
file240time = "data/20180803-rata/Voltage/04-240_Vm_15x15times.csv"
dataVm240 = np.loadtxt(fileVm240, delimiter=',', skiprows=0)
dataCa240 = np.loadtxt(fileCa240, delimiter=',', skiprows=0)
time240 = np.loadtxt(file240time, delimiter=',', skiprows=0)


for ax in axes.flat:
    ax.tick_params(axis='x', which='both', direction='in')
    ax.tick_params(axis='y', which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.set_xlim(xlim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.set_ylim(ylim)

for idx, ax in enumerate(axes.flat, start=1):
    ax.plot(time240*1000, dataVm240, color='r', linewidth=1, label='Vm')
    ax.plot(time240*1000, dataCa240, color='y', linewidth=1, label='Cm')
    ax.set_ylabel('Normalized\nFluorescence #'+str(idx), fontsize=7, fontweight='bold')

plt.xlabel('Time (ms)', fontsize=8, fontweight='bold')
legend_lines = [Line2D([0], [0], color='r', lw=1),
                Line2D([0], [0], color='y', lw=1)]
fig.legend(legend_lines, ['Vm', 'Ca'],
           loc='upper right', ncol=1, prop={'size': 8}, numpoints=1, frameon=False)
plt.show()
