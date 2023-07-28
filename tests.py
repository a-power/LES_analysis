import xarray as xr
import numpy as np

def calc_z_ML_and_CL(file_path, time_stamp=-1):

    prof_data = xr.open_dataset(file_path)

    wth_prof = prof_data['wtheta_cn_mean'].data[time_stamp, ...]
    z_ML = np.where(wth_prof = np.min(wth_prof))

    cloud = prof_data['total_cloud_fraction'].data[time_stamp, ...]
    z_cloud = np.where(wth_prof != 0)
    z_CL = [ np.min(z_cloud), np.max(z_cloud) ]

    return z_ML, z_CL

file_path = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_ts_18000.nc'

z_ML, z_CL = calc_z_ML_and_CL(file_path, time_stamp=-1)

print('z_ML = ', z_ML, 'z_CL = ', z_CL)