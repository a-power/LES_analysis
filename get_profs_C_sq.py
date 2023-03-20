import xarray as xr
import os
import analysis_plot_fns as apf
import numpy as np


homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_contour = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'p'

path20f = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
file20 = 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'

outdir = path20f+'C_profs/'
os.makedirs(outdir, exist_ok = True)

deltas=['2D', '4D', '8D', '16D', '32D', '64D']

dataset_name = [outdir+file20+'C_2D.nc', outdir+file20+'C_4D.nc', outdir+file20+'C_8D.nc',
                outdir+file20+'C_16D.nc', outdir+file20+'C_32D.nc', outdir+file20+'C_64D.nc']

# 'field': 'f(LM_field_on_w)_r'
# 'field': 'Cs_field'
# 'field': 'f(HR_th_field_on_w)_r'
# 'field': 'Cth_field'
# 'field': 'f(HR_q_total_field_on_w)_r'
# 'field': 'Cqt_field'

fields = ['Cs_sq_field', 'Cth_sq_field', 'Cqt_sq_field']
field_dir = ['Cs', 'C_th', 'C_qt']


cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'

gen_opts = {'contour_field_in': dir_contour,
            'deltas': None,
            'other_vars': [w_field, th_v_field],
            'cloud_thres': 10**(-5),
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'grid': mygrid,
               }

for j in range(len(dataset_name)):

    ds = xr.Dataset()
    ds.to_netcdf(dataset_name[j], mode='w')
    ds_in = {'file': dataset_name[j], 'ds': ds}

    for i, field_in in enumerate(fields):

        C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
            apf.get_conditional_profiles(field=field_in, **gen_opts, \
                    dataset_in = path20f+file20+str(f'{field_dir[i]}_{deltas[j]}_running_mean_filter_rm00.nc'))


        save_field(ds_in, C_sq_env_prof)
        save_field(ds_in, C_sq_cloud_prof)
        save_field(ds_in, C_sq_combo2_prof)
        save_field(ds_in, C_sq_combo3_prof)


    ds.close()