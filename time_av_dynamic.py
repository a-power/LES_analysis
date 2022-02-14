import numpy as np
import xarray as xr
import dynamic as dyn

def time_av_dyn(dx_in, time_in, indir, grid):

    file_in = f'{indir}{dx_in}/diagnostic_files/BOMEX_m{dx_in}_all_{time_in}.nc'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time_series_600_600']
    times = time_data.data
    nt = len(times)
    z_in = ds_in['z']
    z = z_in.data
    np.save(f'files/{dx}_z', z)




    return