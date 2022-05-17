import numpy as np
import xarray as xr
from subfilter.io.dataout import save_field
import dynamic as dyn

import subfilter


av_type = 'all'
mygrid = 'w'
ingrid = mygrid
t_in = 0

path20f = '/storage/silver/MONC_data/Alanna/bomex/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

#os.makedirs(outdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
dataset_name2 = path20f+file20+'Pr_profiles_2D.nc'

file_in = data_2D
ds_in = xr.open_dataset(file_in)
time_data = ds_in['time']
times = time_data.data
nt = len(times)
print('lenght of the time array in Cs function is', nt)

x_data = ds_in['x_p']
x_s = x_data.data

y_data = ds_in['y_p']
y_s = y_data.data

z_data = ds_in['z']
z_s = z_data.data

j_data = ds_in['j']
j_s = j_data.data

ds_in.close()



ds_in = xr.open_dataset(file_in)
u_th = ds_in[f's(u,th)_on_{ingrid}'].data[t_in, ...]
v_th = ds_in[f's(v,th)_on_{ingrid}'].data[t_in, ...]
w_th = ds_in[f's(w,th)_on_{ingrid}'].data[t_in, ...]

Hj = dyn.H_j(u_th, v_th, w_th)

u_th = None # Save storage
v_th = None # Save storage
w_th = None # Save storage

hat_abs_S = ds_in['f(abs_S)_r'].data[t_in, ...]
dth_dx_hat = ds_in['f(dth_dx)_r'].data[:,t_in, ...]
HAT_abs_S_dth_dx = ds_in['f(abs_S_dth_dx)_r'].data[t_in, ...]

Rj = dyn.R_j(20, 40, hat_abs_S, dth_dx_hat, HAT_abs_S_dth_dx, beta=1)

Hj = xr.DataArray(Hj[np.newaxis, ...], coords={'time': [times[t_in]], 'j': j_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                        dims=["time", "j", "x_p", "y_p", "z"], name='Hj')

Rj = xr.DataArray(Rj[np.newaxis, ...], coords={'time': [times[t_in]], 'j': j_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                        dims=["time", "j", "x_p", "y_p", "z"], name='Rj')


ds_2 = xr.Dataset()

ds_2.to_netcdf(dataset_name2, mode='w')

ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, Hj)
save_field(ds_in2, Rj)


ds_2.close()







