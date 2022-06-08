import time_av_dynamic as t_dy
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'
path20f = '/work/scratch-pw/apower/20m_gauss_dyn_update_subfilt/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
outdir = outdir_og + '20m_update_subfilt' + '/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')


dataset_name2 = path20f+file20+'Cq_2D_attempt.nc'
dataset_name4 = path20f+file20+'Cq_4D_attempt.nc'


########################################################################

# C_q_sq_prof_2D, C_q_prof_2D, HR_prof_2D, RR_prof_2D, C_q_sq_field_2D, HR_field_2D, RR_field_2D, Hj_2D, Rj_2D = \
#     t_dy.C_scalar('q_total', data_2D, dx=20, dx_hat=40, ingrid=mygrid)
#
# ds_2 = xr.Dataset()
# ds_2.to_netcdf(dataset_name2, mode='w')
# ds_in2 = {'file':dataset_name2, 'ds': ds_2}
#
# save_field(ds_in2, C_q_sq_prof_2D)
# save_field(ds_in2, C_q_prof_2D)
# save_field(ds_in2, HR_prof_2D)
# save_field(ds_in2, RR_prof_2D)
# save_field(ds_in2, C_q_sq_field_2D)
# save_field(ds_in2, HR_field_2D)
# save_field(ds_in2, RR_field_2D)
# save_field(ds_in2, Hj_2D)
# save_field(ds_in2, Rj_2D)
#
# ds_2.close()
#
# C_q_sq_prof_2D = None
# C_q_prof_2D = None
# HR_prof_2D = None
# RR_prof_2D = None
# C_q_sq_field_2D = None
# HR_field_2D = None
# RR_field_2D = None
# Hj_2D = None
# Rj_2D = None

##########################################

C_q_sq_prof_4D, C_q_prof_4D, HR_prof_4D, RR_prof_4D, C_q_sq_field_4D, HR_field_4D, RR_field_4D, Hj_4D, Rj_4D = \
    t_dy.C_scalar('q_total', data_2D, dx=20, dx_hat=80, ingrid=mygrid)

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
