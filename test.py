import numpy as np
import xarray as xr

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_ga00.nc"
indir = path20f+file20

file_in = f'{indir}'
ds_in = xr.open_dataset(file_in)
time_data = ds_in['time']
times = time_data.data
print('time array is', times)