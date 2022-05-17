import numpy as np
import xarray as xr
from subfilter.io.dataout import save_field
import time_av_dynamic as t_dy

import subfilter

av_type = 'all'
mygrid = 'w'
t_in = 0

path20f = '/storage/silver/MONC_data/Alanna/bomex/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

#os.makedirs(outdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
dataset_name2 = path20f+file20+'Pr_profiles_2D.nc'


C_th_sq_prof, C_th_prof, HR_prof, RR_prof, C_th_sq_field, HR_field, RR_field, Hj, Rj = \
    t_dy.C_th(data_2D, dx=20, dx_hat=40, ingrid=mygrid)



ds_2 = xr.Dataset()

ds_2.to_netcdf(dataset_name2, mode='w')

ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, C_th_sq_prof)
save_field(ds_in2, C_th_prof)
save_field(ds_in2, HR_prof)
save_field(ds_in2, RR_prof)
save_field(ds_in2, C_th_sq_field)
save_field(ds_in2, HR_field)
save_field(ds_in2, RR_field)
save_field(ds_in2, Hj)
save_field(ds_in2, Rj)


ds_2.close()







