# Vertically interpolate cloud cover into cloud area fraction
# For more documentation see cloud_area_fraction.ipynb

import os
import xarray as xr
import numpy as np
from joblib import Parallel, delayed

SOURCE = 'DYAMOND'

# Define all paths
path = '/home/b/b309170/bd1179_work/DYAMOND'
grid_path = '/home/b/b309170/bd1179_work/DYAMOND/grid_extpar'
output_path = os.path.join(path, 'vcg_data')

# Load files (ds_zh_lr = ds_zhalf_lowres)
ds_zh_lr = xr.open_dataset(os.path.join(grid_path, 'zghalf_icon-a_capped_upsampled_R02B05_DYAMOND.nc')) # 83886080 x (48-16)
# ds_zh_hr = xr.open_dataset(os.path.join(grid_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid.nc')) # 83886080 x 91
# Extract values
zh_lr = ds_zh_lr.zghalf.values # Should be 32 x 83886080
# zh_hr = ds_zh_hr.h.values  # Should be 91 x 83886080

HORIZ_FIELDS = zh_lr.shape[1]
VERT_LAYERS_LR = zh_lr.shape[0] - 1
# VERT_LAYERS_HR = zh_hr.shape[0] - 1

weights = np.load(os.path.join(grid_path, 'weights_DYAMOND_R02B10_cloud_area_fraction.npy'), mmap_mode='r')[:, -60:] # Only 60 layers in clc
# weights = np.load(os.path.join(grid_path, 'weights_DYAMOND_R02B10_cloud_area_fraction.npy'))[:, -60:] # Only 60 layers in clc

files = os.listdir(os.path.join(path, 'clc_data'))
files_nc = [files[i] for i in range(len(files)) if '.nc' in files[i]]

def convert(input_file):    
    # Skip if the file is already in output_path. Careful: g2 != nc!!
    if 'int_var_' + input_file[:-2] + 'nc' in os.listdir(output_path):
        return
    
    output_file = 'int_var_' + input_file
    with open(os.path.join(output_path, output_file), 'w') as file:
        file.write('Dummy file created.')

    print(input_file)

    DS = xr.open_dataset(os.path.join(path, 'clc_data', input_file))
    clc = DS.clc.values
    TIME_STEPS = len(DS.time)

    # Modify the ndarray. Desired output shape: (8, 31, 83886080). (clc_out = clc, vertically interpolated)
    clc_out = np.full((TIME_STEPS, VERT_LAYERS_LR, HORIZ_FIELDS), np.nan)

    for t in range(TIME_STEPS):
        for j in range(VERT_LAYERS_LR):    
            clc_out[t][j] = np.max(weights[j]*clc[t], axis=0) #Element-wise product

    clc_new_da = xr.DataArray(clc_out, coords={'time':DS.time, 'height':DS.height[:VERT_LAYERS_LR]}, 
                              dims=['time', 'height', 'cells'], name='cl_area') 

    # Save it in a new file
    clc_new_da.to_netcdf(os.path.join(output_path, output_file))
    
# Requires 700GB. If we don't have that amount of memory, then set n_jobs = 6, and mmap_mode = 'r' above. 
Parallel(n_jobs=2, verbose=0, backend="threading")(map(delayed(convert), np.array(files_nc)))