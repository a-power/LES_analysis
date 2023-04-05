import xarray as xr
import os
import analysis_plot_fns as apf
import numpy as np

beta=True

if beta==True:
    homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/smoothed_LM_HR_fields/'
    dir_contour = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
else:
    homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/smoothed_LM_HR_fields/'
    dir_contour = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

myfile = 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'

from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'p'

outdir = homedir+'C_profs/'
os.makedirs(outdir, exist_ok = True)

deltas=['2D', '4D', '8D', '16D', '32D', '64D']

if beta==True:
    dataset_name = [outdir+myfile+'C_2D_', outdir+myfile+'C_4D_', outdir+myfile+'C_8D_',
                outdir+myfile+'C_16D_', outdir+myfile+'C_32D_', outdir+myfile+'C_64D_']
    extra_filter = ['0']#, '1']
else:
    dataset_name = [outdir+myfile+'C_2D.nc', outdir+myfile+'C_4D.nc', outdir+myfile+'C_8D.nc',
                    outdir+myfile+'C_16D.nc', outdir+myfile+'C_32D.nc', outdir+myfile+'C_64D.nc']

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

gen_opts = {'deltas': None,
            'other_vars': [w_field, th_v_field],
            'cloud_thres': 1e-7,
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'grid': mygrid,
               }

for j, delta_in in enumerate(deltas):

    ds = xr.Dataset()
    ds.to_netcdf(dataset_name[j], mode='w')
    ds_in = {'file': dataset_name[j], 'ds': ds}

    ########### need to fix this, fo now only do one 2nd filt at a time

    for i, field_in in enumerate(fields):

        if beta == True:
            for k, name_2_gauss in enumerate(extra_filter):

                mydataset = homedir + myfile + \
                            str(f'{field_dir[i]}_{delta_in}_{name_2_gauss}_running_mean_filter_rm00.nc')
                mydir_contour = dir_contour + f'{j}_gaussian_filter_ga0{name_2_gauss}_running_mean_filter_rm00.nc'

                C_sq_prof, C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
                    apf.get_conditional_profiles(field=field_in, **gen_opts, dataset_in = mydataset,
                                                 contour_field_in = mydir_contour)

                save_field(ds_in, C_sq_prof)
                save_field(ds_in, C_sq_env_prof)
                save_field(ds_in, C_sq_cloud_prof)
                save_field(ds_in, C_sq_combo2_prof)
                save_field(ds_in, C_sq_combo3_prof)

        else:
            mydataset = homedir + myfile + str(f'{field_dir[i]}_{deltas[j]}_running_mean_filter_rm00.nc')
            mydir_contour = dir_contour + f'{j}_running_mean_filter_rm00.nc'

            C_sq_prof, C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
                apf.get_conditional_profiles(field=field_in, **gen_opts, dataset_in = mydataset,
                                             contour_field_in = mydir_contour + f'{i}_running_mean_filter_rm00.nc')

            save_field(ds_in, C_sq_prof)
            save_field(ds_in, C_sq_env_prof)
            save_field(ds_in, C_sq_cloud_prof)
            save_field(ds_in, C_sq_combo2_prof)
            save_field(ds_in, C_sq_combo3_prof)


    ds.close()