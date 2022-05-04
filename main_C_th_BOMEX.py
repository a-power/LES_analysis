import time_av_dynamic as t_dy
from subfilter.io.dataout import save_field
import os

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


Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d, Cs_sq_field_2d, LM_field_2d, MM_field_2d = \
    t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=0, save_all=2)

ds_in2 = xr.open_dataset(data_2D, mode='a')

save_field(ds_in2, Cs_prof_sq_2d)
save_field(ds_in2, Cs_prof_2d)
save_field(ds_in2, LM_prof_2d)
save_field(ds_in2, MM_prof_2d)
save_field(ds_in2, Cs_sq_field_2d)
save_field(ds_in2, LM_field_2d)
save_field(ds_in2, MM_field_2d)

Cs_prof_sq_2d = None        #free memory
Cs_prof_2d = None           #free memory
LM_prof_2d = None           #free memory
MM_prof_2d = None           #free memory
Cs_sq_field_2d = None       #free memory
LM_field_2d = None          #free memory
MM_field_2d = None          #free memory

ds_in2.close()


Cs_prof_sq_4d, Cs_prof_4d, LM_prof_4d, MM_prof_4d, Cs_sq_field_4d, LM_field_4d, MM_field_4d = \
    t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, t_in=0, save_all=2)

ds_in4 = xr.open_dataset(data_4D, mode='a')


save_field(ds_in4, Cs_prof_sq_4d)
save_field(ds_in4, Cs_prof_4d)
save_field(ds_in4, LM_prof_4d)
save_field(ds_in4, MM_prof_4d)
save_field(ds_in4, Cs_sq_field_4d)
save_field(ds_in4, LM_field_4d)
save_field(ds_in4, MM_field_4d)

ds_in4.close()