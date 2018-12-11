import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))  # Half-page figure


# Images, 640 x 512 px
# RH237 dye, Voltage, Left
# heartVmImage = Image.open('data/20181109-pigb/Voltage/12-300_Vm_0001.tif').rotate(90)
# axes[0, 0].imshow(np.array(heartVmImage), cmap='bone')
heartVmImage = np.rot90(plt.imread('data/20181109-pigb/Voltage/12-300_Vm_0001.tif'))
axes[0, 0].axis('off')
axes[0, 0].imshow(heartVmImage, cmap='bone')
axes[0, 0].set_title('RH237 Staining, Vm', fontsize=7, fontweight='bold')

# RHod-2 dye, Calcium, Right
heartCaImage = np.rot90(plt.imread('data/20181109-pigb/Calcium/12-300_Ca_0001.tif'))
axes[1, 0].axis('off')
axes[1, 0].imshow(heartVmImage, cmap='bone')
axes[1, 0].set_title('Rhod-2 Staining, Va', fontsize=7, fontweight='bold')

# Region of Interest circles, adjusted for rotation
roiXY = (300, 200)
roiR = 20
roiCircleVm = Circle(roiXY, roiR, fc='none', ec='k', lw=2)
roiCircleCa = Circle(roiXY, roiR, fc='none', ec='k', lw=2)
axes[0, 0].add_patch(roiCircleVm)
axes[1, 0].add_patch(roiCircleCa)


xlim = [150, 450]
ylim = [0, 1]
for ax in axes[:, 1]:
    ax.tick_params(axis='x', which='both', direction='in')
    ax.tick_params(axis='y', which='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.show()
