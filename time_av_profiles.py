import numpy as np
import xarray as xr

def time_av_prof(vars, dx_list, time_in, indir):
    for i, dx in enumerate(dx_list):
        file_in = f'{indir}{dx}/diagnostic_files/BOMEX_m{dx}_all_{time_in}.nc'
        ds_in = xr.open_dataset(file_in)
        time_in = ds_in['time_series_600_600']
        times = time_in.data[0]
        nt = len(times)
        z_in = ds_in['z']
        z = z_in.data[0]

        for j, var_in in enumerate(vars):
            var_data = np.zeros(z)
            av_var_data = np.zeros(z)
            for k, time_stamp in enumerate(times):
                my_var = ds_in[f'{var_in}'][k,...]
                var_data += my_var.data
            av_var_data = var_data/nt
            np.save(f'files/{dx}_{var_in}_time_av', av_var_data)
    return z