# Vertically interpolate cloud cover into cloud area fraction
# For more documentation see cloud_area_fraction.ipynb

# Takes 160 minutes for DYAMOND data (ran on a 900GB compute node)

import os
import xarray as xr
import numpy as np

SOURCE = 'DYAMOND'       

# Define all paths
grid_path = '~/bd1179_work/DYAMOND/grid_extpar'

# Load files (ds_zh_lr = ds_zhalf_lowres)
ds_zh_lr = xr.open_dataset(os.path.join(grid_path, 'zghalf_icon-a_capped_upsampled_R02B05_DYAMOND.nc'))
ds_zh_hr = xr.open_dataset(os.path.join(grid_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid.nc'))  
# Extract values
zh_lr = ds_zh_lr.zghalf.values # Should be 32 x 83886080
zh_hr = ds_zh_hr.h.values[0]  # Should be 91 x 83886080

HORIZ_FIELDS = zh_lr.shape[1]
VERT_LAYERS_LR = zh_lr.shape[0] - 1
VERT_LAYERS_HR = zh_hr.shape[0] - 1

# # Have to run this code only once!
# # Storing the weights in a file
weights = np.zeros((VERT_LAYERS_LR, VERT_LAYERS_HR, HORIZ_FIELDS), dtype=np.bool_)

for j in range(VERT_LAYERS_LR):
    z_u = zh_lr[j, :]
    z_l = zh_lr[j+1, :]
    weights_layer = np.maximum(np.minimum(z_u, zh_hr[:-1]) - np.maximum(zh_hr[1:], z_l), 0)

    weights[j, weights_layer > 0] = True
    
np.save(os.path.join(grid_path, 'weights_DYAMOND_R02B10_cloud_area_fraction.npy'), np.int8(weights))