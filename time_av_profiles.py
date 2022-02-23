import numpy as np
import xarray as xr
import os as os

def time_av_prof(vars, dx_list, time_in, indir, data='bomex'):
    for i, dx in enumerate(dx_list):
        if data=='bomex':
            file_in = f'{indir}{dx}/diagnostic_files/BOMEX_m{dx}_all_{time_in}.nc'
        elif data=='dry_cbl':
            time_in='13200'
            if dx=='5':
                print('find file path for 5m profile data - subgrid directory?')
                #file_in = f'/storage/silver/scenario/si818415/phd/{dx}mLES/cbl_{time_in}.nc'
            else:
                file_in = f'/storage/silver/scenario/si818415/phd/{dx}mLES/cbl_{time_in}.nc'
        else:
            print("data type not yet configured for a file path")

        #### create dir:
        path1 = './files/{data}/'
        path2 = './plots/{data}/'
        isExist1 = os.path.exists(path1)
        if not isExist1:
            os.makedirs(path1)
        isExist2 = os.path.exists(path2)
        if not isExist2:
            os.makedirs(path2)

        ds_in = xr.open_dataset(file_in)
        if data=='bomex':
            time_data = ds_in['time_series_600_600']
        elif data=='dry_cbl':
            time_data = ds_in['time_series_25_300 ']
        else:
            print('find time veriable for data')
        times = time_data.data
        nt = len(times)
        z_in = ds_in['z']
        z = z_in.data
        np.save(f'files/{data}/{dx}_z', z)

        for j, var_in in enumerate(vars):
            var_data = np.zeros(len(z))
            av_var_data = np.zeros(len(z))
            for k, time_stamp in enumerate(times):
                my_var = ds_in[f'{var_in}'][k,...]
                var_data += my_var.data
            av_var_data = var_data/nt
            np.save(f'files/{data}/{dx}_{var_in}_time_av', av_var_data)
    return

## Dataset.mean(time).plot()