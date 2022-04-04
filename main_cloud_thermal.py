import numpy as np
import os
import dynamic as dy
import xarray as xr

set_time = '14400'
set_time_step = 0
in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
dx = '0020_g0800'
outdir_og = '/work/scratch-pw/apower/'
outdir = outdir_og + '20m_cloud_thermal' +'/'
plotdir = outdir_og+'plots/dyn/'


file_in = f'{in_dir}{dx}/diagnostic_files/BOMEX_m{dx}_all_{set_time}.nc'


ds_in = xr.open_dataset(file_in)
w_in = ds_in['w']
w_field = w_in.data

q_in = ds_in['q_cloud_liquid_mass']
q_cloud = q_in.data


thermals_field, w_95th = dy.w_therm_field(w_field, t_in=set_time_step, return_all=True)
clouds_field = dy.cloud_field_ind(q_cloud, t_in=set_time_step, cloud_liquid_threshold = 10**(-5))

np.save(outdir+'thermals'+set_time_step, thermals_field)
np.save(outdir+'w_95th'+set_time_step, w_95th)
np.save(outdir+'clouds'+set_time_step, clouds_field)
