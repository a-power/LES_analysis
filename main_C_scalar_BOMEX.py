import time_av_dynamic as t_dy
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'
path20f = '/work/scratch-pw/apower/20m_gauss_dyn_update_subfilt/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

# outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
# outdir = outdir_og + '20m_update_subfilt' + '/'
# os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')

dataset_name2 = path20f+file20+'C_scalar_2D.nc'
dataset_name4 = path20f+file20+'C_scalar_4D.nc'


########################################################################


ds_2 = xr.Dataset()
ds_2.to_netcdf(dataset_name2, mode='w')
ds_in2 = {'file':dataset_name2, 'ds': ds_2}

# Cs_sq_prof_2D, Cs_prof_2D, LM_prof_2D, MM_prof_2D, Cs_sq_field_2D, LM_field_2D, MM_field_2D, Lij_2D, Mij_2D = \
#     t_dy.Cs(data_2D, dx=20, dx_hat=20, ingrid=mygrid, t_in=0, save_all=3, reaxes=False)

# save_field(ds_in2, Cs_sq_prof_2D)
# save_field(ds_in2, Cs_prof_2D)
# save_field(ds_in2, LM_prof_2D)
# save_field(ds_in2, MM_prof_2D)
# save_field(ds_in2, Cs_sq_field_2D)
# save_field(ds_in2, LM_field_2D)
# save_field(ds_in2, MM_field_2D)
# save_field(ds_in2, Lij_2D)
# save_field(ds_in2, Mij_2D)
#
# Cs_sq_prof_2D = None
# Cs_prof_2D = None
# LM_prof_2D = None
# MM_prof_2D = None
# Cs_sq_field_2D = None
# LM_field_2D = None
# MM_field_2D = None
# Lij_2D = None
# Mij_2D = None

C_th_sq_prof_2D, C_th_prof_2D, HR_th_prof_2D, RR_th_prof_2D, C_th_sq_field_2D, HR_th_field_2D, RR_th_field_2D, Hj_th_2D, Rj_th_2D = \
    t_dy.C_scalar('th', data_2D, dx=20, dx_hat=40, ingrid=mygrid)

save_field(ds_in2, C_th_sq_prof_2D)
save_field(ds_in2, C_th_prof_2D)
save_field(ds_in2, HR_th_prof_2D)
save_field(ds_in2, RR_th_prof_2D)
save_field(ds_in2, C_th_sq_field_2D)
save_field(ds_in2, HR_th_field_2D)
save_field(ds_in2, RR_th_field_2D)
save_field(ds_in2, Hj_th_2D)
save_field(ds_in2, Rj_th_2D)

C_th_sq_prof_2D = None
C_th_prof_2D = None
HR_th_prof_2D = None
RR_th_prof_42 = None
C_th_sq_field_2D = None
HR_th_field_2D = None
RR_th_field_2D = None
Hj_th_2D = None
Rj_th_2D = None

# C_q_sq_prof_2D, C_q_prof_2D, HR_prof_2D, RR_prof_2D, C_q_sq_field_2D, HR_field_2D, RR_field_2D, Hj_2D, Rj_2D = \
#     t_dy.C_scalar('q_total', data_2D, dx=20, dx_hat=40, ingrid=mygrid)

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
# C_q_sq_prof_2D = None
# C_q_prof_2D = None
# HR_prof_2D = None
# RR_prof_2D = None
# C_q_sq_field_2D = None
# HR_field_2D = None
# RR_field_2D = None
# Hj_2D = None
# Rj_2D = None


C_qv_sq_prof_2D, C_qv_prof_2D, HR_v_prof_2D, RR_v_prof_2D, C_qv_sq_field_2D, HR_v_field_2D, RR_v_field_2D, Hj_v_2D, Rj_v_2D = \
    t_dy.C_scalar("q_vapour", data_2D, dx=20, dx_hat=40, ingrid=mygrid)

save_field(ds_in2, C_qv_sq_prof_2D)
save_field(ds_in2, C_qv_prof_2D)
save_field(ds_in2, HR_v_prof_2D)
save_field(ds_in2, RR_v_prof_2D)
save_field(ds_in2, C_qv_sq_field_2D)
save_field(ds_in2, HR_v_field_2D)
save_field(ds_in2, RR_v_field_2D)
save_field(ds_in2, Hj_v_2D)
save_field(ds_in2, Rj_v_2D)

C_qv_sq_prof_2D = None
C_qv_prof_2D = None
HR_v_prof_2D = None
RR_v_prof_2D = None
C_qv_sq_field_2D = None
HR_v_field_2D = None
RR_v_field_2D = None
Hj_v_2D = None
Rj_v_2D = None

C_ql_sq_prof_2D, C_ql_prof_2D, HR_l_prof_2D, RR_l_prof_2D, C_ql_sq_field_2D, HR_l_field_2D, RR_l_field_2D, Hj_l_2D, Rj_l_2D = \
    t_dy.C_scalar("q_cloud_liquid_mass", data_2D, dx=20, dx_hat=40, ingrid=mygrid)

save_field(ds_in2, C_ql_sq_prof_2D)
save_field(ds_in2, C_ql_prof_2D)
save_field(ds_in2, HR_l_prof_2D)
save_field(ds_in2, RR_l_prof_2D)
save_field(ds_in2, C_ql_sq_field_2D)
save_field(ds_in2, HR_l_field_2D)
save_field(ds_in2, RR_l_field_2D)
save_field(ds_in2, Hj_l_2D)
save_field(ds_in2, Rj_l_2D)

C_ql_sq_prof_2D = None
C_ql_prof_2D = None
HR_l_prof_2D = None
RR_l_prof_2D = None
C_ql_sq_field_2D = None
HR_l_field_2D = None
RR_l_field_2D = None
Hj_l_2D = None
Rj_l_2D = None


ds_2.close()

#########################################################################


##########################################

ds_4 = xr.Dataset()
ds_4.to_netcdf(dataset_name4, mode='w')
ds_in4 = {'file':dataset_name4, 'ds': ds_4}

# Cs_sq_prof_4D, Cs_prof_4D, LM_prof_4D, MM_prof_4D, Cs_sq_field_4D, LM_field_4D, MM_field_4D, Lij_4D, Mij_4D = \
#     t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid=mygrid, t_in=0, save_all=3, reaxes=False)

# save_field(ds_in4, Cs_sq_prof_4D)
# save_field(ds_in4, Cs_prof_4D)
# save_field(ds_in4, LM_prof_4D)
# save_field(ds_in4, MM_prof_4D)
# save_field(ds_in4, Cs_sq_field_4D)
# save_field(ds_in4, LM_field_4D)
# save_field(ds_in4, MM_field_4D)
# save_field(ds_in4, Lij_4D)
# save_field(ds_in4, Mij_4D)
#
# Cs_sq_prof_4D = None
# Cs_prof_4D = None
# LM_prof_4D = None
# MM_prof_4D = None
# Cs_sq_field_4D = None
# LM_field_4D = None
# MM_field_4D = None
# Lij_4D = None
# Mij_4D = None

C_th_sq_prof_4D, C_th_prof_4D, HR_th_prof_4D, RR_th_prof_4D, C_th_sq_field_4D, HR_th_field_4D, RR_th_field_4D, Hj_th_4D, Rj_th_4D = \
    t_dy.C_scalar('th', data_4D, dx=20, dx_hat=80, ingrid=mygrid)

save_field(ds_in4, C_th_sq_prof_4D)
save_field(ds_in4, C_th_prof_4D)
save_field(ds_in4, HR_th_prof_4D)
save_field(ds_in4, RR_th_prof_4D)
save_field(ds_in4, C_th_sq_field_4D)
save_field(ds_in4, HR_th_field_4D)
save_field(ds_in4, RR_th_field_4D)
save_field(ds_in4, Hj_th_4D)
save_field(ds_in4, Rj_th_4D)

C_th_sq_prof_4D = None
C_th_prof_4D = None
HR_th_prof_4D = None
RR_th_prof_4D = None
C_th_sq_field_4D = None
HR_th_field_4D = None
RR_th_field_4D = None
Hj_th_4D = None
Rj_th_4D = None

# C_q_sq_prof_4D, C_q_prof_4D, HR_prof_4D, RR_prof_4D, C_q_sq_field_4D, HR_field_4D, RR_field_4D, Hj_4D, Rj_4D = \
#     t_dy.C_scalar('q_total', data_4D, dx=20, dx_hat=80, ingrid=mygrid)

# save_field(ds_in4, C_q_sq_prof_4D)
# save_field(ds_in4, C_q_prof_4D)
# save_field(ds_in4, HR_prof_4D)
# save_field(ds_in4, RR_prof_4D)
# save_field(ds_in4, C_q_sq_field_4D)
# save_field(ds_in4, HR_field_4D)
# save_field(ds_in4, RR_field_4D)
# save_field(ds_in4, Hj_4D)
# save_field(ds_in4, Rj_4D)
#
# C_q_sq_prof_4D = None
# C_q_prof_4D = None
# HR_prof_4D = None
# RR_prof_4D = None
# C_q_sq_field_4D = None
# HR_field_4D = None
# RR_field_4D = None
# Hj_4D = None
# Rj_4D = None


C_qv_sq_prof_4D, C_qv_prof_4D, HR_v_prof_4D, RR_v_prof_4D, C_qv_sq_field_4D, HR_v_field_4D, RR_v_field_4D, Hj_v_4D, Rj_v_4D = \
    t_dy.C_scalar("q_vapour", data_4D, dx=20, dx_hat=80, ingrid=mygrid)

save_field(ds_in4, C_qv_sq_prof_4D)
save_field(ds_in4, C_qv_prof_4D)
save_field(ds_in4, HR_v_prof_4D)
save_field(ds_in4, RR_v_prof_4D)
save_field(ds_in4, C_qv_sq_field_4D)
save_field(ds_in4, HR_v_field_4D)
save_field(ds_in4, RR_v_field_4D)
save_field(ds_in4, Hj_v_4D)
save_field(ds_in4, Rj_v_4D)

C_qv_sq_prof_4D = None
C_qv_prof_4D = None
HR_v_prof_4D = None
RR_v_prof_4D = None
C_qv_sq_field_4D = None
HR_v_field_4D = None
RR_v_field_4D = None
Hj_v_4D = None
Rj_v_4D = None

C_ql_sq_prof_4D, C_ql_prof_4D, HR_l_prof_4D, RR_l_prof_4D, C_ql_sq_field_4D, HR_l_field_4D, RR_l_field_4D, Hj_l_4D, Rj_l_4D = \
    t_dy.C_scalar("q_cloud_liquid_mass", data_4D, dx=20, dx_hat=80, ingrid=mygrid)

save_field(ds_in4, C_ql_sq_prof_4D)
save_field(ds_in4, C_ql_prof_4D)
save_field(ds_in4, HR_l_prof_4D)
save_field(ds_in4, RR_l_prof_4D)
save_field(ds_in4, C_ql_sq_field_4D)
save_field(ds_in4, HR_l_field_4D)
save_field(ds_in4, RR_l_field_4D)
save_field(ds_in4, Hj_l_4D)
save_field(ds_in4, Rj_l_4D)

C_ql_sq_prof_4D = None
C_ql_prof_4D = None
HR_l_prof_4D = None
RR_l_prof_4D = None
C_ql_sq_field_4D = None
HR_l_field_4D = None
RR_l_field_4D = None
Hj_l_4D = None
Rj_l_4D = None


ds_4.close()

#########################################################################
