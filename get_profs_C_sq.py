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
parser.add_argument('--case', type=str, default='ARM')

times_analysed = [ '18000', '25200', '32400', '39600' ]

args = parser.parse_args()
set_time = times_analysed[args.times]

beta=False
data_smoothed = False
case = args.case

if case == 'BOMEX':

    dx = 20

    if beta==True:
        homedir = '/work/scratch-pw3/apower/BOMEX/second_filt/LM/'
        dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
    else:
        homedir = '/work/scratch-pw3/apower/BOMEX/second_filt/LM/'
        dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

    myfile = 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'

elif case == 'ARM':

    dx = 25

    if beta==True:
        homedir = '/work/scratch-pw3/apower/ARM/second_filt/LM/'
        dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'
    else:
        homedir = '/work/scratch-pw3/apower/ARM/second_filt/LM/'
    dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'
    myfile = f"diagnostics_3d_ts_{set_time}_gaussian_filter_"

av_type = 'all'
mygrid = 'p'

outdir = homedir+'C_profs/'
os.makedirs(outdir, exist_ok = True)

deltas=['4D', '8D', '16D', '32D', '64D', '128D']

if beta==True:
    dataset_name = [outdir+myfile+'C_2D_', outdir+myfile+'C_4D_', outdir+myfile+'C_8D_',
                outdir+myfile+'C_16D_', outdir+myfile+'C_32D_', outdir+myfile+'C_64D_']
    # dataset_name = [outdir + myfile + 'LM_2D_', outdir + myfile + 'LM_4D_', outdir + myfile + 'LM_8D_',
    #                             outdir+myfile+'LM_16D_', outdir+myfile+'LM_32D_', outdir+myfile+'LM_64D_']
    extra_filter = [0] #, 1]
else:
    dataset_name = [outdir+myfile+'C_4D', outdir+myfile+'C_8D', outdir+myfile+'C_16D',
                    outdir+myfile+'C_32D', outdir+myfile+'C_64D', outdir+myfile+'C_128D']

    # dataset_name = [outdir + myfile + 'LM_2D', outdir + myfile + 'LM_4D', outdir + myfile + 'LM_8D',
    #                 outdir + myfile + 'LM_16D', outdir + myfile + 'LM_32D', outdir + myfile + 'LM_64D']

# 'field': 'f(LM_field_on_w)_r'
# 'field': 'Cs_field'
# 'field': 'f(HR_th_field_on_w)_r'
# 'field': 'Cth_field'
# 'field': 'f(HR_q_total_field_on_w)_r'
# 'field': 'Cqt_field'

# fields = ['Cs_sq_field', 'Cth_sq_field', 'Cqt_sq_field']
# field_dir = ['Cs', 'C_th', 'C_qt']

if data_smoothed == True:
    fields = [f'f(LM_field_on_{mygrid})_r', f'f(HR_th_field_on_{mygrid})_r', f'f(HR_q_total_field_on_{mygrid})_r',
                  f'f(MM_field_on_{mygrid})_r', f'f(RR_th_field_on_{mygrid})_r', f'f(RR_q_total_field_on_{mygrid})_r']

    cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
    w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
    w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
    th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'
    buoy_field = f'f(f(buoyancy_on_{mygrid})_r_on_{mygrid})_r'
else:
     fields = ['LM_field', 'HR_th_field', 'HR_q_total_field', 'MM_field', 'RR_th_field', 'RR_q_total_field']

     cloud_field = f'f(q_cloud_liquid_mass_on_{mygrid})_r'
     w_field = f'f(w_on_{mygrid})_r'
     w2_field = f'f(w_on_{mygrid}.w_on_{mygrid})_r'
     th_v_field = f'f(th_v_on_{mygrid})_r'
     buoy_field = f'f(buoyancy_on_{mygrid})_r'

field_dir = ['Cs', 'C_th', 'C_qt', 'Cs', 'C_th', 'C_qt']


# cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
# w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
# w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
# th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'



gen_opts = {'deltas': None,
            'other_vars': [w_field, buoy_field],
            'cloud_thres': 1e-5,
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'grid': mygrid
               }

for j, delta_in in enumerate(deltas):

    if beta == True:
        for k, name_2_gauss in enumerate(extra_filter):

            ds = xr.Dataset()
            ds.to_netcdf(dataset_name[j]+f'{name_2_gauss}'+'.nc', mode='w')
            ds_in = {'file': dataset_name[j]+ f'{name_2_gauss}'+'.nc', 'ds': ds}

    ########### need to fix this, fo now only do one 2nd filt at a time

            for i, field_in in enumerate(fields):
                if data_smoothed == True:
                    mydataset = homedir + myfile + \
                                str(f'{field_dir[i]}_{j}_{name_2_gauss}_running_mean_filter_rm00.nc')
                    mydir_contour = dir_contour + f'{j}_gaussian_filter_ga0{name_2_gauss}_running_mean_filter_rm00.nc'
                else:
                    mydataset = homedir + myfile + \
                                str(f'{field_dir[i]}_{j}_{name_2_gauss}.nc')
                    mydir_contour = dir_contour + f'{j}_gaussian_filter_ga0{name_2_gauss}.nc'

                C_sq_prof, C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
                    apf.get_conditional_profiles(field=field_in, **gen_opts, dataset_in = mydataset,
                                                 contour_field_in = mydir_contour, beta=True)

                save_field(ds_in, C_sq_prof)
                save_field(ds_in, C_sq_env_prof)
                save_field(ds_in, C_sq_cloud_prof)
                save_field(ds_in, C_sq_combo2_prof)
                save_field(ds_in, C_sq_combo3_prof)

    else:
        ds = xr.Dataset()
        ds.to_netcdf(dataset_name[j] + '.nc', mode='w')
        ds_in = {'file': dataset_name[j] + '.nc', 'ds': ds}

        for i, field_in in enumerate(fields):

            if data_smoothed == True:
                mydataset = homedir + myfile + str(f'{field_dir[i]}_{dx*2**(j+1)}_{dx*2**(j+2)}_running_mean_filter_rm00.nc')
                mydir_contour = dir_contour + f'{j}_gaussian_filter_ga00_running_mean_filter_rm00.nc'
            else:
                mydataset = homedir + myfile + str(f'{field_dir[i]}_{dx*2**(j+1)}_{dx*2**(j+2)}.nc')
                mydir_contour = dir_contour + f'{j}_gaussian_filter_ga00.nc'

            C_sq_prof, C_sq_env_prof, C_sq_cloud_prof, C_sq_combo2_prof, C_sq_combo3_prof = \
                apf.get_conditional_profiles(field=field_in, **gen_opts, dataset_in = mydataset,
                                             contour_field_in = mydir_contour, beta=False)

            save_field(ds_in, C_sq_prof)
            save_field(ds_in, C_sq_env_prof)
            save_field(ds_in, C_sq_cloud_prof)
            save_field(ds_in, C_sq_combo2_prof)
            save_field(ds_in, C_sq_combo3_prof)


    ds.close()