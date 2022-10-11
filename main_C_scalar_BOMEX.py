import time_av_dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

scalar = ['th', 'q_total']


av_type = 'all'
mygrid = 'w'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

# outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
# outdir = outdir_og + '20m_update_subfilt' + '/'
#
# os.makedirs(outdir, exist_ok = True)
# os.makedirs(plotdir, exist_ok = True)


data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')
data_8D = path20f+file20+str('ga02.nc')
data_16D = path20f+file20+str('ga03.nc')

dataset_name2 = path20f+file20+'profiles_2D.nc'
dataset_name4 = path20f+file20+'profiles_4D.nc'
dataset_name8 = path20f+file20+'profiles_8D.nc'
dataset_name16 = path20f+file20+'profiles_16D.nc'


########################################################################

C_q_sq_prof_2D, C_q_prof_2D, HR_prof_2D, RR_prof_2D, C_q_sq_field_2D, HR_field_2D, RR_field_2D, Hj_2D, Rj_2D = \
    dy_s.C_scalar('q_total', data_2D, dx=20, dx_hat=40, ingrid=mygrid)

ds_2 = xr.Dataset()
ds_2.to_netcdf(dataset_name2, mode='w')
ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, C_q_sq_prof_2D)
save_field(ds_in2, C_q_prof_2D)
save_field(ds_in2, HR_prof_2D)
save_field(ds_in2, RR_prof_2D)
save_field(ds_in2, C_q_sq_field_2D)
save_field(ds_in2, HR_field_2D)
save_field(ds_in2, RR_field_2D)
save_field(ds_in2, Hj_2D)
save_field(ds_in2, Rj_2D)

ds_2.close()

C_q_sq_prof_2D = None
C_q_prof_2D = None
HR_prof_2D = None
RR_prof_2D = None
C_q_sq_field_2D = None
HR_field_2D = None
RR_field_2D = None
Hj_2D = None
Rj_2D = None

##########################################

C_q_sq_prof_4D, C_q_prof_4D, HR_prof_4D, RR_prof_4D, C_q_sq_field_4D, HR_field_4D, RR_field_4D, Hj_4D, Rj_4D = \
    dy_s.C_scalar('q_total', data_4D, dx=20, dx_hat=80, ingrid=mygrid)

ds_4 = xr.Dataset()
ds_4.to_netcdf(dataset_name4, mode='w')
ds_in4 = {'file':dataset_name4, 'ds': ds_4}

save_field(ds_in4, C_q_sq_prof_4D)
save_field(ds_in4, C_q_prof_4D)
save_field(ds_in4, HR_prof_4D)
save_field(ds_in4, RR_prof_4D)
save_field(ds_in4, C_q_sq_field_4D)
save_field(ds_in4, HR_field_4D)
save_field(ds_in4, RR_field_4D)
save_field(ds_in4, Hj_4D)
save_field(ds_in4, Rj_4D)

ds_4.close()

C_q_sq_prof_4D = None
C_q_prof_4D = None
HR_prof_4D = None
RR_prof_4D = None
C_q_sq_field_4D = None
HR_field_4D = None
RR_field_4D = None
Hj_4D = None
Rj_4D = None


#########################################################################


C_q_sq_prof_8D, C_q_prof_8D, HR_prof_8D, RR_prof_8D, C_q_sq_field_8D, HR_field_8D, RR_field_8D, Hj_8D, Rj_8D = \
    dy_s.C_scalar('q_total', data_8D, dx=20, dx_hat=160, ingrid=mygrid)

ds_8 = xr.Dataset()
ds_8.to_netcdf(dataset_name8, mode='w')
ds_in8 = {'file':dataset_name8, 'ds': ds_8}

save_field(ds_in8, C_q_sq_prof_8D)
save_field(ds_in8, C_q_prof_8D)
save_field(ds_in8, HR_prof_8D)
save_field(ds_in8, RR_prof_8D)
save_field(ds_in8, C_q_sq_field_8D)
save_field(ds_in8, HR_field_8D)
save_field(ds_in8, RR_field_8D)
save_field(ds_in8, Hj_8D)
save_field(ds_in8, Rj_8D)

ds_8.close()

C_q_sq_prof_8D = None
C_q_prof_8D = None
HR_prof_8D = None
RR_prof_8D = None
C_q_sq_field_8D = None
HR_field_8D = None
RR_field_8D = None
Hj_8D = None
Rj_8D = None

##########################################

C_q_sq_prof_16D, C_q_prof_16D, HR_prof_16D, RR_prof_16D, C_q_sq_field_16D, HR_field_16D, RR_field_16D, Hj_16D, Rj_16D = \
    dy_s.C_scalar('q_total', data_16D, dx=20, dx_hat=3200, ingrid=mygrid)

ds_16 = xr.Dataset()
ds_16.to_netcdf(dataset_name16, mode='w')
ds_in16 = {'file':dataset_name16, 'ds': ds_16}

save_field(ds_in16, C_q_sq_prof_16D)
save_field(ds_in16, C_q_prof_16D)
save_field(ds_in16, HR_prof_16D)
save_field(ds_in16, RR_prof_16D)
save_field(ds_in16, C_q_sq_field_16D)
save_field(ds_in16, HR_field_16D)
save_field(ds_in16, RR_field_16D)
save_field(ds_in16, Hj_16D)
save_field(ds_in16, Rj_16D)

ds_16.close()

C_q_sq_prof_16D = None
C_q_prof_16D = None
HR_prof_16D = None
RR_prof_16D = None
C_q_sq_field_16D = None
HR_field_16D = None
RR_field_16D = None
Hj_16D = None
Rj_16D = None
