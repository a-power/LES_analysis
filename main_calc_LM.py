import dynamic_script as dy_s
from monc_utils.io.dataout import save_field
import os
import xarray as xr
import argparse
import numpy as np
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=0)
parser.add_argument('--case', type=str, default='ARM')
parser.add_argument('--filting_filts', type=str, default='n')
args = parser.parse_args()
t_in = args.t
filtering_filters_yn = args.filting_filts

case_in = args.case

av_type = 'all'
mygrid = 'p'

if filtering_filters_yn == 'y' or filtering_filters_yn == 'yes':
    filtering_filters = True
else:
    filtering_filters = False

if case_in == 'BOMEX':
    path_f = '/work/scratch-pw3/apower/BOMEX/'
    folder_f = 'first_filt/'
    folder_ff = 'second_filt/'
    times_list = ['14400']
    time_in = times_list[0]
    file_f = f'BOMEX_m0020_g0800_all_{time_in}_'
    Delta = 20
    if filtering_filters == True:
        scalar = ['momentum', 'f(th_on_p)_r', 'q_total']
    else:
        scalar = ['momentum', 'th', 'q_total']
    dx_bar_in = np.array([20, 40, 80, 160, 320, 640])
    dx_hat_in = np.array([40, 80, 160, 320, 640, 1280])
    if filtering_filters == True:
        dx_bar_in = 2*dx_bar_in
        dx_hat_in = 2*dx_hat_in
    C_res = ['2D', '4D', '8D', '16D', '32D', '64D']

elif case_in == 'ARM':
    times_list = ['18000', '25200', '32400', '39600']
    time_in = times_list[t_in]
    path_f = '/work/scratch-pw3/apower/ARM/'
    folder_f = 'first_filt/'
    folder_ff = 'second_filt/'
    file_f = f'diagnostics_3d_ts_{time_in}_'
    Delta = 25
    # dx_bar_in = [56, 103, 202, 401, 800, 1600]
    # dx_hat_in = [75, 144, 284, 566, 1132, 2263]
    dx_bar_in = np.array([25, 50, 100, 200, 400, 800])
    dx_hat_in = np.array([50, 100, 200, 400, 800, 1600])
    if filtering_filters == True:
        dx_bar_in = 2*dx_bar_in
        dx_hat_in = 2*dx_hat_in
    C_res = ['2D', '4D', '8D', '16D', '32D', '64D']
    if filtering_filters == True:
        scalar = ['momentum', 'f(th_on_p)_r', 'q_total']
    else:
        scalar = ['momentum', 'th', 'q_total']

elif case_in=='dry':
    times_list = ['13800']
    time_in = times_list[t_in]
    path_f = f'/storage/silver/greybls/si818415/dry_CBL/'
    folder_f = 'first_filt/'
    folder_ff = 'second_filt/'
    file_f = f'cbl_{time_in}_'
    Delta=20
    if filtering_filters == True:
        scalar = ['momentum', 'f(th_on_p)_r']
    else:
        scalar = ['momentum', 'th']
    dx_bar_in = np.array([20, 40, 80, 160, 320, 640])
    dx_hat_in = np.array([40, 80, 160, 320, 640, 1280])
    if filtering_filters == True:
        dx_bar_in = 2*dx_bar_in
        dx_hat_in = 2*dx_hat_in
    C_res = ['2D', '4D', '8D', '16D', '32D', '64D']

else:
    print('case not recognised')

# if filtering_filters == True:
#     os.makedirs(path_f+folder_ff, exist_ok = True)


set_save_all = 2


for i, C_res_in in enumerate(C_res):

    if filtering_filters == True:
        file_in = file_f + f'gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'
        data_in = path_f + folder_ff + file_in
        print('reading files', data_in)

        os.makedirs(path_f + folder_ff + 'LM/', exist_ok = True)
        dataset_name = [path_f + folder_ff + 'LM/' + file_f + f'Cs_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_ff + 'LM/' + file_f + f'C_th_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_ff + 'LM/' + file_f + f'C_qt_{dx_bar_in[i]}_{dx_hat_in[i]}.nc']

    elif filtering_filters == False:
        file_in = file_f + f'gaussian_filter_ga0{i}.nc'
        data_in = path_f + folder_f + file_in
        print('reading files', data_in)
        os.makedirs(path_f + folder_f + 'LM/', exist_ok=True)
        dataset_name = [path_f + folder_f + 'LM/' + file_f + f'Cs_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_f + 'LM/' + file_f + f'C_th_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_f + 'LM/' + file_f + f'C_qt_{dx_bar_in[i]}_{dx_hat_in[i]}.nc']

    DX_in = {
        'indir': data_in,
        'dx_bar': dx_bar_in[i],
        'dx_hat': dx_hat_in[i]
    }

    for j, scalar_in in enumerate(scalar):

    ########################################################################
        #  = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \

        if scalar_in == 'momentum':
            scalar_index = 0
        elif scalar_in == 'th' or scalar_in == 'f(th_on_p)_r':
            scalar_index = 1
        elif scalar_in == 'q_total':
            scalar_index = 2
        else:
            print('scalar not set to momentum, th, or q_total')

        file_setup = dataset_name[scalar_index]
        # ds = xr.Dataset()
        # ds_in = {'file':dataset_name[scalar_index], 'ds': ds}
        # ds.to_netcdf(file_setup, mode='w')

        if scalar_in == 'momentum':
            #z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
            dy_s.Cs(file_save_to=file_setup, ingrid=mygrid, save_all=set_save_all, **DX_in)
            # = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        else:
            #z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
            dy_s.C_scalar(scalar=scalar_in, file_save_to=file_setup, ingrid=mygrid, save_all=set_save_all, **DX_in)

        gc.collect()

        # save_field(ds_in, HR_field)
        # save_field(ds_in, RR_field)
        # save_field(ds_in, z_save)
        # save_field(ds_in, zn_save)
        # save_field(ds_in, C_sq_prof)
        # save_field(ds_in, C_prof)
        # save_field(ds_in, HR_prof)
        # save_field(ds_in, RR_prof)
        # #save_field(ds_in2 = \ #, C_sq_field_2D)
        # # save_field(ds_in2, Hj_2D)
        # # save_field(ds_in2, Rj_2D)
        #
        # ds.close()

        # C_sq_prof = None
        # C_prof = None
        # HR_prof = None
        # RR_prof = None
        # HR_field = None
        # RR_field = None

