# For more documentation see vert_int_method_variability.ipynb

import os
import sys
import xarray as xr
import numpy as np

## Set by the user ##
GRID_RES = 'R02B05'      # Can be 'R02B05', 'R02B04'. Relevant for both the input and the output grid.
SOURCE = 'ERA5'          # Can be 'NARVAL', 'QUBICC', 'HDCP2', 'DYAMOND', 'ERA5'. To set paths, input grids and variable names.
                         # For NARVAL additionally check var_path and output_path in lines 70-80s
# VAR_TYPES = 'state_vars' # Can be 'state_vars', 'cloud_vars'. To focus on specific variables.
var_name = sys.argv[1]
#####################

# Setting the paths, grids and variables
# -> ds_zh_lr: Low resolution vertical half levels
# -> ds_zh_hr: High resolution vertical half levels
if SOURCE == 'NARVAL':
    base_path = '~/my_work/NARVAL'
    # Variable name of half levels
    height_var = 'z_ifc'
    # Setting var_names (which variables to vertically interpolate)
    if VAR_TYPES == 'state_vars':
        var_names = ['qv', 'pres', 'rho', 'temp', 'u', 'v']
    elif VAR_TYPES == 'cloud_vars':
        var_names = ['clc', 'qi', 'qc']
    if GRID_RES == 'R02B05':
        ds_zh_lr = xr.open_dataset(os.path.join(base_path, 'grid_extpar/zghalf_icon-a_capped_R02B05.nc'))
        ds_zh_hr = xr.open_dataset(os.path.join(base_path, 'grid_extpar/z_ifc_R02B05_NARVAL_fg_DOM01_ML_capped.nc'))
    elif GRID_RES == 'R02B04':
        zg_lowres_path = os.path.join(narval_path, 'data_var_vertinterp/zg')
        zg_highres_path = os.path.join(narval_path, 'data/z_ifc')
        ds_zh_lr = xr.open_dataset(os.path.join(zg_lowres_path, 'zghalf_icon-a_capped.nc'))
        ds_zh_hr = xr.open_dataset(os.path.join(zg_highres_path, 'z_ifc_R02B04_NARVALI_fg_DOM01.nc')) 
elif SOURCE == 'QUBICC': 
    base_path = '/scratch/snx3000/agrundne' #Scratch
    # Variable name of half levels
    height_var = 'zghalf'
    # Setting var_names (which variables to vertically interpolate)
    if VAR_TYPES == 'state_vars':
        var_names = ['hus', 'pfull', 'rho', 'ta', 'ua', 'va']
    elif VAR_TYPES == 'cloud_vars':
        var_names = ['cl', 'cli', 'clw']
    # Setting ds_zh_lr and ds_zh_hr
    if GRID_RES == 'R02B05':
        ds_zh_lr = xr.open_dataset(os.path.join(base_path, 'grids/zghalf_icon-a_capped_R02B05.nc'))
        if VAR_TYPES == 'state_vars':
            ds_zh_hr = xr.open_dataset(os.path.join(base_path, 'grids/qubicc_l89_zghalf_ml_0019_R02B05_G.nc'))
        elif VAR_TYPES == 'cloud_vars':
            ds_zh_hr = xr.open_dataset(os.path.join(base_path, 'grids/qubicc_l91_zghalf_ml_0019_R02B05_G.nc'))
    elif GRID_RES == 'R02B04':
        ds_zh_lr = xr.open_dataset(os.path.join(base_path, 'grids/zghalf_icon-a_capped.nc'))
        ds_zh_hr = xr.open_dataset(os.path.join(base_path, 'grids/qubicc_l91_zghalf_ml_0015_R02B04_G.nc'))
elif SOURCE == 'HDCP2': # Actually I'm not working with HDCP2 data at the moment
    base_path = '~/bd1179_work/hdcp2' 
    # Variable name of half levels
    height_var = 'z_ifc'
    # Setting var_names (which variables to vertically interpolate)
    if VAR_TYPES == 'state_vars':
        var_names = ['hus', 'ninact', 'pres', 'qg', 'qh', 'qnc', 'qng', 'qnh', 'qni', 'qnr', 'qns', 'qr', 'qs', 'ta', 'ua', 'va', 'zg']
    elif VAR_TYPES == 'cloud_vars':
        var_names = ['clc', 'cli', 'clw']
    # Setting ds_zh_lr and ds_zh_hr
    assert GRID_RES == 'R02B04' # Not implemented for R02B05
    zghalf_highres_path = os.path.join(base_path, 'grids') # Need 151 vertical layers here, not 76 like in NARVAL.
    zghalf_lowres_path = os.path.join('~/my_work/NARVAL', 'data_var_vertinterp/zg') 
    ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, 'z_ifc_vert_remapcon_3d_coarse_ll_DOM03_ML.nc'))
    ds_zh_lr = xr.open_dataset(os.path.join(zghalf_lowres_path, 'zghalf_icon-a_capped.nc'))
elif SOURCE == 'DYAMOND':
    base_path = '~/bd1179_work/DYAMOND'
    # Variable name of half levels
    height_var = 'h'
    # Setting var_names (which variables to vertically interpolate)
    if VAR_TYPES == 'state_vars':
        var_names = ['pa', 'hus', 'ta', 'clw', 'cli', 'ua', 'va']
    elif VAR_TYPES == 'cloud_vars':
        var_names = ['clc']
    # Setting ds_zh_lr and ds_zh_hr
    # Both ds_zh_hr and ds_zh_lr have to be defined on exactly the same horizontal grid!
    assert GRID_RES == 'R02B05'
    zghalf_highres_path = os.path.join(base_path, 'grid_extpar')
    zghalf_lowres_path = '~/my_work/QUBICC/grids/'
    if VAR_TYPES == 'state_vars':
        ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid_R02B05_l78.nc'))
    elif VAR_TYPES == 'cloud_vars':
        ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid_R02B05_l61.nc'))
    ds_zh_lr = xr.open_dataset(os.path.join(zghalf_lowres_path, 'zghalf_icon-a_capped_R02B05.nc'))
elif SOURCE == 'ERA5':
    base_path = '~/bd1179_work/ERA5'
    # Variable name of half levels
    height_var = 'z'
    # Setting ds_zh_lr and ds_zh_hr
    # Both ds_zh_hr and ds_zh_lr have to be defined on exactly the same horizontal grid!
    assert GRID_RES == 'R02B05'
    zghalf_highres_path = os.path.join(base_path, 'hcg_data')
    zghalf_lowres_path = '~/my_work/QUBICC/grids/'
    # # In ERA5, the height of full and half levels depend on the time step!
    # if VAR_TYPES == 'state_vars':
    #     ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid_R02B05_l78.nc'))
    # elif VAR_TYPES == 'cloud_vars':
    #     ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, 'zghalf_dyamond_R2B10_lkm1007_vgrid_R02B05_l61.nc'))
    ds_zh_lr = xr.open_dataset(os.path.join(zghalf_lowres_path, 'zghalf_icon-a_capped_R02B05.nc'))
    

# Actual vertical interpolation method
print('Currently processing %s'%var_name)

# Input and output folders
if SOURCE == 'NARVAL':
    var_path = os.path.join('~/bd1179_work/narval/hcg_files', var_name)
    output_path = os.path.join('~/bd1179_work/narval/hvcg_files', var_name)
elif SOURCE in ['QUBICC', 'DYAMOND', 'ERA5']:
    var_path = os.path.join(base_path, 'hcg_data', var_name)
    output_path = os.path.join(base_path, 'hvcg_data', var_name)
elif SOURCE == 'HDCP2':
    var_path = os.path.join(base_path, 'hor_cg_files_temp', var_name)
    output_path = os.path.join(base_path, 'data', var_name)

# Can only happen in QUBICC
if SOURCE == 'QUBICC' and var_name == 'clw':
    var_name = 'qclw_phy'

ls = os.listdir(var_path)
    
def process_files(file_inds):
    for i in file_inds:
        # Which file to load
        input_file = os.listdir(var_path)[i]        
        print(input_file)

        # Skip if the file is already in output_path. Otherwise create it quickly!
        if 'int_var_' + input_file in os.listdir(output_path):
            continue
            
        # Load files (ds_zh_lr = ds_zhalf_lowres)
        ds = xr.open_dataset(os.path.join(var_path, input_file))

        # In ERA5, the height of full and half levels depend on the time step and sometimes they are missing the last time step!
        if SOURCE == 'ERA5':
            # Read half-level file
            height_file = input_file.split('_')[0] + '_' + input_file.split('_')[1] + '_' + input_file.split('_')[2] + '_zh_' + input_file.split('_')[4] 
            ds_zh_hr = xr.open_dataset(os.path.join(zghalf_highres_path, height_file))
            time_steps = len(ds_zh_hr.time)
        else:
            time_steps = len(ds.time)

        # Extract values
        var = getattr(ds, var_name).values
        zh_lr = ds_zh_lr.zghalf.values
        zh_hr = (getattr(ds_zh_hr, height_var).values).squeeze()

        HORIZ_FIELDS = zh_hr.shape[-1]

        # Extract not-nan entries (var_n = var_notnan)
        not_nan = ~np.isnan(var[0,-1,:])        
        var_n = var[:,:,not_nan]        
        zh_lr_n = zh_lr[:,not_nan]
        if SOURCE == 'ERA5':
            zh_hr_n = zh_hr[:,:,not_nan]
        else:
            zh_hr_n = zh_hr[:,not_nan]

        # Modify the ndarray. Have 31 vertical full levels in the output. (var_out = var, vertically interpolated)
        var_out = np.full((time_steps, 31, var_n.shape[2]), np.nan) # var_n.shape[2] = Number of not_nans

        # Pretty fast implementation:
        for t in range(time_steps):
            for j in range(31):
                z_u = zh_lr_n[j, :]
                z_l = zh_lr_n[j+1, :]
                # weights.shape = var_n[0].shape = high-res_layers x len(not_nan)
                # len(z_u) = len(z_l) = len(not_nan)
                if SOURCE == 'ERA5':
                    weights = np.maximum(np.minimum(z_u, zh_hr_n[t,:-1]) - np.maximum(zh_hr_n[t,1:], z_l), 0)
                else:
                    weights = np.maximum(np.minimum(z_u, zh_hr_n[:-1]) - np.maximum(zh_hr_n[1:], z_l), 0)
                var_out[t,j,:] = np.einsum('ij,ji->i', weights.T, var_n[t])/(z_u - z_l)


                # If the low-dim grid extends farther than the high-dim grid, we reinsert nans:
                should_be_nan = np.where(np.abs((z_u - z_l) - np.sum(weights, axis = 0)) >= 0.5)
                var_out[t,j,should_be_nan] = np.full(len(should_be_nan), np.nan)

        # Put it back in. Have 20480/81920 horizontal fields in the output.
        var_new = np.full((time_steps, 31, HORIZ_FIELDS), np.nan)

        var_new[:,:,not_nan] = var_out

        var_new_da = xr.DataArray(var_new, coords={'time':ds.time[:time_steps], 'lon':ds.clon, 'lat':ds.clat, 'height':ds_zh_lr.height[-32:-1]}, dims=['time', 'height', 'cell'], name=var_name)

        # Save it in a new file
        output_file = 'int_var_' + input_file
        var_new_da.to_netcdf(os.path.join(output_path, output_file))

    
from contextlib import contextmanager
import multiprocessing as mlp
import gc

@contextmanager
def poolcontext(*args, **kwargs):
    pool = mlp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()  

# Parallelize w.r.t. files
procs = 10
# procs = 1
with poolcontext(processes=procs) as pool:
    # Every process received a part of data_dict
    pool.map(process_files, [np.arange(k*np.ceil(len(ls)/procs), np.minimum((k+1)*np.ceil(len(ls)/procs), len(ls)), dtype=int) for k in range(procs)])
