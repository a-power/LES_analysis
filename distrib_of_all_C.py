import xarray as xr
import os
import analysis_plot_fns as apf
import numpy as np

homedir_BOMEX = '/work/scratch-pw3/apower/BOMEX/second_filt/'
mydir_BOMEX = homedir + 'LM/update/BOMEX_m0020_g0800_all_14400_'
thdir_BOMEX = homedir + 'th/LM/update/BOMEX_m0020_g0800_all_14400_'
contour_data_BOMEX = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter'
deltas_in_BOMEX = ['80_160', '160_320', '320_640', '640_1280', '1280_2560']

dir_s_BOMEX = mydir_BOMEX + 'Cs_'
dir_th_BOMEX = thdir_BOMEX + 'C_th_'
dir_th_L_BOMEX = mydir_BOMEX + 'C_th_L_'
dir_qt_BOMEX = mydir_BOMEX + 'C_qt_'


homedir_ARM = '/work/scratch-pw3/apower/ARM/second_filt/'
mydir_ARM = homedir + f"LM/update/diagnostics_3d_ts_{set_time}_"
thdir_ARM = homedir + f"th/LM/diagnostics_3d_ts_{set_time}_"
contour_data_ARM = homedir + f"diagnostics_3d_ts_{set_time}_gaussian_filter"
deltas_in_ARM = ['100_200', '200_400', '400_800', '800_1600', '1600_3200']

dir_s_ARM = mydir_ARM + 'Cs_'
dir_th_ARM = thdir_ARM + 'C_th_'
dir_th_L_ARM = mydir_ARM + 'C_th_L_'
dir_qt_ARM = mydir_ARM + 'C_qt_'


plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/distribs/'

#'/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir_in, exist_ok = True)

cloud_field = f'f(q_cloud_liquid_mass_on_{mygrid})_r'
w_field = f'f(w_on_{mygrid})_r'
w2_field = f'f(w_on_{mygrid}.w_on_{mygrid})_r'
th_v_field = f'f(th_v_on_{mygrid})_r'



gen_options = {'plotdir': plotdir,
            'data_contour': dir_contour,
            'other_vars': None,
            'cloud_liquid_threshold_in': 10**(-5),
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'times': 0,
            'grid': 'p',
            'return_all_in': False,
            'set_bins':50,
            'deltas': ['4D', '16D', '64D']
            }


LijMij_options = {'field': f'f(LM_field_on_{mygrid})_r',
           'data_field_list': dir_s
           }
Cs_options = {'field': 'Cs_field',
           'data_field_list': dir_s
           }
Cs_sq_options = {'field': 'Cs_sq_field',
           'data_field_list': dir_s
           }


HjRj_th_options = {'field': f'f(HR_th_field_on_{mygrid})_r',
           'data_field_list': dir_th
           }
Cth_options = {'field': 'Cth_field',
           'data_field_list': dir_th
           }
Cth_sq_options = {'field': 'Cth_sq_field',
           'data_field_list': dir_th
           }


HjRj_th_L_options = {'field': f'f(HR_th_L_field_on_{mygrid})_r',
           'data_field_list': dir_th_L
           }
Cth_L_options = {'field': 'Cth_L_field',
           'data_field_list': dir_th_L
           }
Cth_L_sq_options = {'field': 'Cth_L_sq_field',
           'data_field_list': dir_th_L
           }



HjRj_qt_options = {'field': f'f(HR_q_total_field_on_{mygrid})_r',
           'data_field_list': dir_qt
           }
Cqt_options = {'field': 'Cqt_field',
           'data_field_list': dir_qt
           }
Cqt_sq_options = {'field': 'Cqt_sq_field',
           'data_field_list': dir_qt
           }


# apf.C_values_dist(**Cs_sq_options, **gen_options)
# print('Cs_sq fn done')
#
# apf.C_values_dist(**Cth_sq_options, **gen_options)
# print('Cth_sq fn done')

# apf.C_values_dist(**Cqt_sq_options, **gen_options)
# print('Cqt_sq fn done')

def C_values_dist_all_cases(plotdir, data_contour, set_bins, deltas=None, times='0', other_vars=None,
                  other_var_thres=None, less_greater_in=['less'], and_or_in = ['and'],
                  cloud_liquid_threshold_in=10**(-5),
                  res_counter_in=None, return_all_in = False, grid_in='p', **kwargs):

    data_dir = plotdir + f'data/'
    os.makedirs(data_dir, exist_ok=True)

    total_deltas = ['4D', '8D', '16D', '32D', '64D', '128D']

    if deltas==None:
        deltas = ['4D', '8D', '16D', '32D', '64D', '128D']


    for i in range(len(deltas)):
        for iter in range(len(total_deltas)):
            if deltas[i] == total_deltas[iter]:
                delta_BOMEX = deltas_in_BOMEX[iter]
                delta_ARM = deltas_in_ARM[iter]
                break






                

        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_contour +
                                                                f'{iter}_gaussian_filter_ga00.nc',
                                                                cloud_liquid_threshold=cloud_liquid_threshold_in)

        data_field = data_field_list+f'{iter}_0_running_mean_filter_rm00.nc'
        data_field_in = xr.open_dataset(data_field)

        if other_vars!=None:
            if return_all_in == False:
                if len(other_vars) == 1:
                    combo2_out_mask = clo.cloudy_and_or(data_contour +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask = clo.cloudy_and_or(data_contour  +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
            else:
                if len(other_vars) == 1:
                    combo2_out_mask, cloud_mask, var_mask = clo.cloudy_and_or(data_contour  +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask, cloud_mask, var_mask, extra_var_mask = \
                        clo.cloudy_and_or(data_contour + f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)

        if field == 'Cs_field' or field == 'Cs_sq_field':
            print('length of time array for LM is ', len(data_field_in[f'f(LM_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(LM_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(MM_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cs_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None

        elif field == 'Cth_field' or field == 'Cth_sq_field':
            print('length of time array for HR_th is ', len(data_field_in[f'f(HR_th_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(RR_th_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cth_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None

        elif field == 'Cqt_field' or field == 'Cqt_sq_field':
            print('length of time array for HR_qt is ',
                  len(data_field_in[f'f(HR_q_total_f_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(HR_q_total_f_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(RR_q_total_f_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cqt_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None



        else:
            data_field = data_field_in[f'{field}'].data[...]
            print(np.shape(data_field[...]))

        data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
        data_field_env = ma.masked_array(data_field, mask=env_only_mask)

        if len(other_vars) == 2:
            data_field_cloud_up = ma.masked_array(data_field, mask=combo2_out_mask)
            data_field_cloud_core = ma.masked_array(data_field, mask=combo3_out_mask)

        print(np.shape(data_field_env))


        if times != 'av':
            for time_set in times:
                plot_hist(plotdir, field, time_set, deltas[i],
                         data_field_env[time_set,...,0:24],
                          data_field_env[time_set,...,24:151],
                          data_field_cloud[time_set,...],
                          data_names=["ML", "CL: cloud-free", "CL: cloudy"],
                          bins_in=set_bins)


                if len(other_vars) == 2:
                    plot_hist(plotdir, field, time_set, deltas[i],
                              data_field_cloud[time_set,...],
                              data_field_cloud_up[time_set,...],
                              data_field_cloud_core[time_set,...],
                              data_names=["Cloud", "Cloud updraft", "Cloud core"],
                              bins_in=set_bins)


        else:

            plot_hist(plotdir, field, 'av', deltas[i],
                      data_field_env[..., 0:24],
                      data_field_env[..., 24:151],
                      data_field_cloud[...],
                      data_names=["ML", "CL: cloud-free", "CL: cloudy"],
                      bins_in=set_bins)

            if len(other_vars) == 2:
                plot_hist(plotdir, field, 'av', deltas[i],
                          data_field_cloud[...],
                          data_field_cloud_up[...],
                          data_field_cloud_core[...],
                          data_names = ["Cloud", "Cloud updraft", "Cloud core"],
                          bins_in=set_bins)

        np.save( data_dir + f'{deltas[i]}_{field}_flat_domain.npy', data_field.flatten() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_ML.npy', data_field_env[..., 0:24].compressed() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_clear_sky.npy', data_field_env[..., 24:151].compressed() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud.npy', data_field_cloud.compressed() )

        if len(other_vars) == 2:
            np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud_up', data_field_cloud_up.compressed() )
            np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud_core', data_field_cloud_core.compressed() )

    plt.close('all')

