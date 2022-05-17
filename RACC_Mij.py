import time_av_dynamic as t_dy
from subfilter.io.dataout import save_field
import os
import numpy as np
import xarray as xr


import dynamic as dyn
import dask
import subfilter

av_type = 'all'
mygrid = 'w'

path20f = '/storage/silver/MONC_data/Alanna/bomex/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

#os.makedirs(outdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
dataset_name2 = path20f+file20+'NEW_profiles_2D.nc'


Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d, Cs_sq_field_2d, LM_field_2d, MM_field_2d, Lij_2D, Mij_2D = \
    t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=0, save_all=3)


ds_2 = xr.Dataset()
ds_2.to_netcdf(dataset_name2, mode='w')

ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, Cs_prof_sq_2d)
save_field(ds_in2, Cs_prof_2d)
save_field(ds_in2, LM_prof_2d)
save_field(ds_in2, MM_prof_2d)
save_field(ds_in2, Cs_sq_field_2d)
save_field(ds_in2, LM_field_2d)
save_field(ds_in2, MM_field_2d)
save_field(ds_in2, Lij_2D)
save_field(ds_in2, Mij_2D)

Cs_prof_sq_2d = None        #free memory
Cs_prof_2d = None           #free memory
LM_prof_2d = None           #free memory
MM_prof_2d = None           #free memory
Cs_sq_field_2d = None       #free memory
LM_field_2d = None          #free memory
MM_field_2d = None          #free memory

ds_2.close()
print('finished')