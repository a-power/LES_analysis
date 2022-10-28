import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn_6hrs/'
file20 = "BOMEX_m0020_g0800_all_21600_gaussian_filter_"

# outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
# outdir = outdir_og + '20m_update_subfilt' + '/'
# plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'
# os.makedirs(outdir, exist_ok = True)
# os.makedirs(plotdir, exist_ok = True)


data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')
data_8D = path20f+file20+str('ga02.nc')

dataset_name2 = path20f+file20+'Cs_2D.nc'
dataset_name4 = path20f+file20+'Cs_4D.nc'
dataset_name8 = path20f+file20+'Cs_8D.nc'


Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d = \
    dy_s.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, save_all=1)



ds_2 = xr.Dataset()
ds_2.to_netcdf(dataset_name2, mode='w')
ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, Cs_prof_sq_2d)
save_field(ds_in2, Cs_prof_2d)
save_field(ds_in2, LM_prof_2d)
save_field(ds_in2, MM_prof_2d)

Cs_prof_sq_2d = None        #free memory
Cs_prof_2d = None           #free memory
LM_prof_2d = None           #free memory
MM_prof_2d = None           #free memory

ds_2.close()




Cs_prof_sq_4d, Cs_prof_4d, LM_prof_4d, MM_prof_4d = \
    dy_s.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, save_all=1)

ds_4 = xr.Dataset()
ds_4.to_netcdf(dataset_name4, mode='w')
ds_in4 = {'file':dataset_name4, 'ds': ds_4}

save_field(ds_in4, Cs_prof_sq_4d)
save_field(ds_in4, Cs_prof_4d)
save_field(ds_in4, LM_prof_4d)
save_field(ds_in4, MM_prof_4d)

Cs_prof_sq_4d = None        #free memory
Cs_prof_4d = None           #free memory
LM_prof_4d = None           #free memory
MM_prof_4d = None           #free memory

ds_4.close()





Cs_prof_sq_8d, Cs_prof_8d, LM_prof_8d, MM_prof_8d = \
    dy_s.Cs(data_8D, dx=20, dx_hat=160, ingrid = mygrid, save_all=1)


ds_8 = xr.Dataset()
ds_8.to_netcdf(dataset_name8, mode='w')
ds_in8 = {'file':dataset_name8, 'ds': ds_8}

save_field(ds_in8, Cs_prof_sq_8d)
save_field(ds_in8, Cs_prof_8d)
save_field(ds_in8, LM_prof_8d)
save_field(ds_in8, MM_prof_8d)

Cs_prof_sq_8d = None        #free memory
Cs_prof_8d = None           #free memory
LM_prof_8d = None           #free memory
MM_prof_8d = None           #free memory

ds_8.close()


