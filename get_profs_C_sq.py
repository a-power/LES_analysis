import xarray as xr
import os
import analysis_plot_fns as apf
import numpy as np
from monc_utils.io.dataout import save_field
import os
import xarray as xr
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--times', type=str, default='18000')
parser.add_argument('--times', type=int, default=0)
parser.add_argument('--beta', type=int, default=0)
parser.add_argument('--case', type=str, default='ARM')
parser.add_argument('--th', type=str, default='th_v')
parser.add_argument('--q', type=str, default='qt')
parser.add_argument('--fold', type=str, default='th_v')

times_analysed = [ '18000', '25200', '32400', '39600' ]

args = parser.parse_args()
set_time = times_analysed[args.times]
case = args.case
beta = args.beta
th_set = args.th
q_set = args.q
sub_folder = args.fold

data_smoothed = False
av_type = 'all'
mygrid = 'p'

if case == 'BOMEX':
    dx = 20
    if beta==0 or beta==1:
        homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/{sub_folder}/LM/'
        dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
    else:
        homedir = f'/work/scratch-pw3/apower/BOMEX/first_filt/{sub_folder}/LM/'
        dir_contour = '/work/scratch-pw3/apower/BOMEX/first_filt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

    myfile = 'BOMEX_m0020_g0800_all_14400_'

    if beta == 0:
        dx_bar_in = 2 * np.array([20, 40, 80, 160, 320, 640])
        dx_hat_in = 2 * np.array([40, 80, 160, 320, 640, 1280])
    elif beta == 1:
        dx_bar_in = 2 * np.array([20, 40, 80, 160, 320, 640])
        dx_hat_in = 4 * np.array([40])  # , 80, 160, 320, 640, 1280])
    else:
        dx_bar_in = np.array([20, 20, 20, 20, 20, 20])
        dx_hat_in = np.array([40, 80, 160, 320, 640, 1280])

elif case == 'ARM':
    dx = 25
    if beta==0 or beta==1:
        homedir = f'/work/scratch-pw3/apower/ARM/second_filt/{sub_folder}/LM/'
        dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'
    else:
        homedir = f'/work/scratch-pw3/apower/ARM/first_filt/{sub_folder}/LM/'
        dir_contour = f'/work/scratch-pw3/apower/ARM/first_filt/diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'
    myfile = f"diagnostics_3d_ts_{set_time}_"

    if beta == 0:
        dx_bar_in = 2 * np.array([25, 50, 100, 200, 400, 800])
        dx_hat_in = 2 * np.array([50, 100, 200, 400, 800, 1600])
    elif beta == 1:
        dx_bar_in = 2 * np.array([25, 50, 100, 200, 400, 800])
        dx_hat_in = 4 * np.array([50])  # , 100, 200, 400, 800, 1600])
    else:
        dx_bar_in = np.array([25, 25, 25, 25, 25, 25])
        dx_hat_in = np.array([50, 100, 200, 400, 800, 1600])


outdir = homedir+'C_profs/'
os.makedirs(outdir, exist_ok = True)


dataset_name = outdir+myfile+f'C_cond_profs_'


# 'field': 'f(LM_field_on_w)_r'
# 'field': 'Cs_field'
# 'field': 'f(HR_th_field_on_w)_r'
# 'field': 'Cth_field'
# 'field': 'f(HR_q_total_field_on_w)_r'
# 'field': 'Cqt_field'

# fields = ['Cs_sq_field', 'Cth_sq_field', 'Cqt_sq_field']
# field_dir = ['Cs', 'C_th', 'C_qt']

if data_smoothed == True:
    fields = [f'f(LM_field_on_{mygrid})_r', f'f(HR_{th_set}_field_on_{mygrid})_r', f'f(HR_{q_set}_field_on_{mygrid})_r',
                  f'f(MM_field_on_{mygrid})_r', f'f(RR_{th_set}_field_on_{mygrid})_r', f'f(RR_{q_set}_field_on_{mygrid})_r']

    cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
    w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
    w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
    th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'
    buoy_field = f'f(f(buoyancy_on_{mygrid})_r_on_{mygrid})_r'
else:
     fields = [f'HR_{th_set}_field', f'RR_{th_set}_field']
         #['LM_field', f'HR_{th_set}_field', f'HR_{q_set}_field', 'MM_field', f'RR_{th_set}_field', f'RR_{q_set}_field']

     cloud_field = f'f(q_cloud_liquid_mass_on_{mygrid})_r'
     w_field = f'f(w_on_{mygrid})_r'
     w2_field = f'f(w_on_{mygrid}.w_on_{mygrid})_r'
     th_v_field = f'f(th_v_on_{mygrid})_r'
     buoy_field = f'f(buoyancy_on_{mygrid})_r'

field_dir = [f'C_{th_set}', f'C_{th_set}'] #['Cs', f'C_{th_set}', f'C_{q_set}', 'Cs', f'C_{th_set}', f'C_{q_set}']


# cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
# w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
# w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
# th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'



gen_opts = {'deltas': None,
            'other_vars': [w_field, th_v_field],
            'cloud_thres': 1e-5,
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'grid': mygrid
               }

if beta == 1:
    beta_num = 1
else:
    beta_num = 0



for j, delta_in in enumerate(dx_hat_in):


    ds = xr.Dataset()
    ds.to_netcdf(dataset_name + f'{dx_bar_in[j]}_{dx_hat_in[j]}.nc', mode='w')
    ds_in = {'file': dataset_name + f'{dx_bar_in[j]}_{dx_hat_in[j]}.nc', 'ds': ds}

    for i, field_in in enumerate(fields):

        if data_smoothed == True:
            mydataset = homedir + myfile + str(f'{field_dir[i]}_{dx*2**(j+1)}_{dx*2**(j+2+beta_num)}_running_mean_filter_rm00.nc')
            mydir_contour = dir_contour + f'{j}_gaussian_filter_ga0{beta_num}_running_mean_filter_rm00.nc'
        else:
            mydataset = homedir + myfile + str(f'{field_dir[i]}_{dx*2**(j+1)}_{dx*2**(j+2+beta_num)}.nc')
            mydir_contour = dir_contour + f'{j}_gaussian_filter_ga0{beta_num}.nc'

        C_sq_prof, C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
            apf.get_conditional_profiles(field=field_in, **gen_opts, dataset_in = mydataset,
                                         contour_field_in = mydir_contour, beta=False)

        save_field(ds_in, C_sq_prof)
        save_field(ds_in, C_sq_env_prof)
        save_field(ds_in, C_sq_cloud_prof)
        save_field(ds_in, C_sq_combo2_prof)
        save_field(ds_in, C_sq_combo3_prof)


    ds.close()