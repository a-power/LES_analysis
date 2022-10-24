import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

# outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
# outdir = outdir_og + '20m_update_subfilt' + '/'
# plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'
# os.makedirs(outdir, exist_ok = True)
# os.makedirs(plotdir, exist_ok = True)


data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')
data_8D = path20f+file20+str('ga02.nc')
data_16D = path20f+file20+str('ga03.nc')

dataset_name2 = path20f+file20+'Cs_2D.nc'
dataset_name4 = path20f+file20+'Cs_4D.nc'
dataset_name8 = path20f+file20+'Cs_8D.nc'
dataset_name16 = path20f+file20+'Cs_16D.nc'



Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d, Cs_sq_field_2d, LM_field_2d, MM_field_2d, L_ij_2, M_ij_2 = \
    dy_s.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, save_all=1)



ds_2 = xr.Dataset()
ds_2.to_netcdf(dataset_name2, mode='w')
ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, Cs_prof_sq_2d)
save_field(ds_in2, Cs_prof_2d)
save_field(ds_in2, LM_prof_2d)
save_field(ds_in2, MM_prof_2d)
save_field(ds_in2, LM_field_2d)
save_field(ds_in2, MM_field_2d)
save_field(ds_in2, L_ij_2)
save_field(ds_in2, M_ij_2)
save_field(ds_in2, Cs_sq_field_2d)

Cs_prof_sq_2d = None        #free memory
Cs_prof_2d = None           #free memory
LM_prof_2d = None           #free memory
MM_prof_2d = None           #free memory
Cs_sq_field_2d = None       #free memory
LM_field_2d = None          #free memory
MM_field_2d = None          #free memory
L_ij_2 = None
M_ij_2 = None

ds_2.close()


Cs_prof_sq_4d, Cs_prof_4d, LM_prof_4d, MM_prof_4d, Cs_sq_field_4d, LM_field_4d, MM_field_4d, L_ij_4, M_ij_4  = \
    dy_s.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, save_all=1)

ds_4 = xr.Dataset()
ds_4.to_netcdf(dataset_name4, mode='w')
ds_in4 = {'file':dataset_name4, 'ds': ds_4}

save_field(ds_in4, Cs_prof_sq_4d)
save_field(ds_in4, Cs_prof_4d)
save_field(ds_in4, LM_prof_4d)
save_field(ds_in4, MM_prof_4d)
save_field(ds_in4, LM_field_4d)
save_field(ds_in4, MM_field_4d)
save_field(ds_in4, L_ij_4)
save_field(ds_in4, M_ij_4)
save_field(ds_in4, Cs_sq_field_4d)

Cs_prof_sq_4d = None        #free memory
Cs_prof_4d = None           #free memory
LM_prof_4d = None           #free memory
MM_prof_4d = None           #free memory
Cs_sq_field_4d = None       #free memory
LM_field_4d = None          #free memory
MM_field_4d = None          #free memory
L_ij_4 = None
M_ij_4 = None

ds_4.close()



Cs_prof_sq_8d, Cs_prof_8d, LM_prof_8d, MM_prof_8d, Cs_sq_field_8d, LM_field_8d, MM_field_8d, L_ij_8, M_ij_8 = \
    dy_s.Cs(data_8D, dx=20, dx_hat=160, ingrid = mygrid, save_all=1)


ds_8 = xr.Dataset()
ds_8.to_netcdf(dataset_name8, mode='w')
ds_in8 = {'file':dataset_name8, 'ds': ds_8}

save_field(ds_in8, Cs_prof_sq_8d)
save_field(ds_in8, Cs_prof_8d)
save_field(ds_in8, LM_prof_8d)
save_field(ds_in8, MM_prof_8d)
save_field(ds_in8, LM_field_8d)
save_field(ds_in8, MM_field_8d)
save_field(ds_in8, L_ij_8)
save_field(ds_in8, M_ij_8)
save_field(ds_in8, Cs_sq_field_8d)

Cs_prof_sq_8d = None        #free memory
Cs_prof_8d = None           #free memory
LM_prof_8d = None           #free memory
MM_prof_8d = None           #free memory
Cs_sq_field_8d = None       #free memory
LM_field_8d = None          #free memory
MM_field_8d = None          #free memory
L_ij_8 = None
M_ij_8 = None

ds_8.close()




###########################################################################################

#
#
#
# Cs_prof_sq_16d, Cs_prof_16d, LM_prof_16d, MM_prof_16d, Cs_sq_field_16d, LM_field_16d, MM_field_16d, L_ij_16, M_ij_16 = \
#     dy_s.Cs(data_16D, dx=20, dx_hat=320, ingrid = mygrid, save_all=1)
#
#
#
# ds_16 = xr.Dataset()
# ds_16.to_netcdf(dataset_name16, mode='w')
# ds_in16 = {'file':dataset_name16, 'ds': ds_16}
#
# save_field(ds_in16, Cs_prof_sq_16d)
# save_field(ds_in16, Cs_prof_16d)
# save_field(ds_in16, LM_prof_16d)
# save_field(ds_in16, MM_prof_16d)
# save_field(ds_in16, LM_field_16d)
# save_field(ds_in16, MM_field_16d)
# save_field(ds_in16, L_ij_16)
# save_field(ds_in16, M_ij_16)
# save_field(ds_in16, Cs_sq_field_16d)
#
# Cs_prof_sq_18d = None        #free memory
# Cs_prof_18d = None           #free memory
# LM_prof_18d = None           #free memory
# MM_prof_18d = None           #free memory
# Cs_sq_field_18d = None       #free memory
# LM_field_18d = None          #free memory
# MM_field_18d = None          #free memory
# L_ij_18 = None
# M_ij_18 = None
#
# ds_16.close()