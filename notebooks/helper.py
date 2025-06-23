import numpy as np
import matplotlib.pyplot as plt
from hmrGC.image3d import trim_zeros
from phantominator import shepp_logan
from matplotlib.patches import Rectangle
import os
import copy

def plot_images(arr, cmap, planes, voxelSize_mm, position_3d, limits, filename='',
                fig_name=0, plot_cmap=True, patch=None):
    val_patch = []
    for plane in planes:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax1)
        
        values = copy.deepcopy(arr)
        
        voxelSize_mm = voxelSize_mm
        cor_aspect = voxelSize_mm[2]/voxelSize_mm[0]
        sag_aspect = voxelSize_mm[2]/voxelSize_mm[1]
        trans_aspect = voxelSize_mm[1]/voxelSize_mm[0]
        if plane == 'coronal':
            values = np.transpose(values, [0, 2, 1])
            values = np.flip(values, axis=[1])
            aspect = cor_aspect
            position = arr.shape[0]-position_3d[2]
        elif plane == 'sagittal':
            values = np.transpose(values, [1, 2, 0])
            values = np.flip(values, axis=1)
            aspect = sag_aspect
            position = position_3d[1]
        elif plane == 'axial':
            values = np.transpose(values, [2, 0, 1])
            aspect = trans_aspect
            position = position_3d[0]

        values, _ = trim_zeros(values[position])
        values = np.squeeze(values)

        if limits is None:
            limits = [0,  np.percentile(values, 99)]

        im1 = ax1.imshow(values, vmin=limits[0], vmax=limits[1],
                         cmap=cmap, aspect=aspect)
        if patch:
            for i in range(len(patch)):
                x_coord = patch[i][0][0]
                x_size = patch[i][1]
                y_coord = patch[i][0][1]
                y_size = patch[i][2]
                values_rect = values[y_coord:y_coord+y_size, x_coord:x_coord+x_size]
                val_patch.append((np.mean(values_rect), np.std(values_rect)))
                rect = Rectangle(patch[i][0],x_size,y_size,linewidth=2,edgecolor='r',facecolor='none')
                ax1.add_patch(rect)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        if os.path.exists(f'./figures/{fig_name}') is False:
            os.makedirs(f'./figures/{fig_name}')
        plt.savefig(f'./figures/{fig_name}/{filename}_{plane}.png')
        plt.show()

    if plot_cmap:
        fig, ax1 = plt.subplots(1, 1, figsize=(3,2))

        im1 = ax1.imshow(values, vmin=limits[0], vmax=limits[1], cmap=cmap,
                         aspect=aspect)
        im1.set_visible(False)
        plt.axis('off')
        cbar = plt.colorbar(im1, ax=ax1,  orientation="horizontal")
        plt.savefig(f'./figures/{fig_name}/{filename}_cmap.eps')
        plt.close()
    return val_patch

def water_fat_shepp_logan():
    M0, T1, T2 = shepp_logan((128,128,20), MR=True)
    water = copy.deepcopy(M0)
    fat = copy.deepcopy(water)
    mask = (M0 == 0)

    ## Assign new values to fat and water
    fat[water < 0.8] = 0.0
    fat[np.abs(M0 - 0.12) < 1e-6] = 2.5/2
    fat[np.abs(M0 - 0.617) < 1e-6] = 20/2
    fat[np.abs(M0 - 0.822) < 1e-6] = 2.5/2
    fat[np.abs(M0 - 0.852) < 1e-6] = 5/2
    fat[np.abs(M0 - 0.93) < 1e-6] = 2.5/2
    fat[np.abs(M0 - 0.98) < 1e-6] = 10/2
    fat[np.abs(M0 - 1.185) < 1e-6] = 15/2
    fat_fraction = fat / (water+fat)
    fat = fat_fraction
    water = (1 - fat_fraction)
    water[mask] = 0
    fat[mask] = 0
    return water, fat, mask