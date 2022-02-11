import numpy as np
import xarray as xr

def time_av_prof(vars, dx_list, time_in, indir):
    for i, dx in enumerate(dx_list):
        file_in = f'{indir}{dx}/diagnostic_files/BOMEX_m{dx}_all_{time_in}.nc'
        print('retrieved file')
        ds_in = xr.open_dataset(file_in)
        print('file opened')
        time_data = ds_in['time_series_600_600']
        print('time retrieved')
        times = time_data.data
        nt = len(times)
        print("nt = ", nt)
        z_in = ds_in['z']
        z = z_in.data
        print("length of z = ", len(z))

        for j, var_in in enumerate(vars):
            var_data = np.zeros(len(z))
            av_var_data = np.zeros(len(z))
            for k, time_stamp in enumerate(times):
                my_var = ds_in[f'{var_in}'][k,...]
                var_data += my_var.data
                print("shape var = ", np.shape(var_data))
            av_var_data = var_data/nt
            print("av var calculated", np.shape(av_var_data))
            np.save(f'files/{dx}_{var_in}_time_av', av_var_data)
            print("saved")
    return z