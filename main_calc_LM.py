import sys

import dynamic_script as dy_s
from monc_utils.io.dataout import save_field
import os
import xarray as xr
import argparse
import numpy as np
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=0)
parser.add_argument('--b', type=int, default=0)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--filting_filts', type=str, default='y')
parser.add_argument('--th_in', type=str, default='th')
parser.add_argument('--q_in', type=str, default='qv')

args = parser.parse_args()
t_in = args.t
beta = args.b
nfilt = args.start
filtering_filters_yn = args.filting_filts
th_type = args.th_in
q_type = args.q_in

case_in = args.case

av_type = 'all'
mygrid = 'p'

set_save_all = 2

print('about to start code')

if filtering_filters_yn == 'y' or filtering_filters_yn == 'yes':
    filtering_filters = True
    print('filtering_filters is set to True')
elif filtering_filters_yn == 'n' or filtering_filters_yn == 'no':
    filtering_filters = False
    print('filtering_filters is set to False')
else:
    print("filting_filts input must be 'y', 'yes', 'n', or 'no'.")
    sys.exit()

if case_in == 'BOMEX':
    print('using BOMEX')
    path_f = '/work/scratch-pw3/apower/BOMEX/'
    folder_f = 'first_filt/'
    folder_ff = f'second_filt/' #{th_type}_and_{q_type}/
    times_list = ['14400']
    time_in = times_list[0]
    file_f = f'BOMEX_m0020_g0800_all_{time_in}_'
    Delta = 20

    if filtering_filters == False:
        dx_bar_in = np.array([20, 20, 20, 20, 20, 20])
        dx_hat_in = np.array([40, 80, 160, 320, 640, 1280])
        C_res = ['2D', '4D', '8D', '16D', '32D', '64D']
        scalar = ['momentum', th_type, q_type]

    elif filtering_filters == True:
        dx_bar_in = 2*np.array([20, 40, 80, 160, 320, 640])
        if beta == 0:
            dx_hat_in = 2 * np.array([40, 80, 160, 320, 640, 1280])
        elif beta == 1:
            dx_hat_in = 4 * np.array([40])#, 80, 160, 320, 640, 1280])
        else:
            print('beta must be =0 or =1')
            sys.exit()

        C_res = ['4D', '8D', '16D', '32D', '64D', '128D']
        scalar = [th_type]#, q_type]
        # scalar = ['momentum' 'th_L', 'q_total'] #, 'f(th_on_p)_r'

elif case_in == 'ARM':
    print('using ARM')
    times_list = ['18000', '25200', '32400', '39600']
    time_in = times_list[t_in]
    path_f = '/work/scratch-pw3/apower/ARM/'
    folder_f = 'first_filt/'
    folder_ff = f'second_filt/'
    file_f = f'diagnostics_3d_ts_{time_in}_'
    Delta = 25

    if filtering_filters == False:
        dx_bar_in = np.array([25, 25, 25, 25, 25, 25])
        dx_hat_in = np.array([50, 100, 200, 400, 800, 1600])
        C_res = ['2D', '4D', '8D', '16D', '32D', '64D']
        scalar = ['momentum', th_type, q_type]

    elif filtering_filters == True:

        scalar = [th_type] #['momentum', 'th_L', 'q_total']

        dx_bar_in = 2*np.array([25, 50, 100, 200, 400, 800])
        if beta == 0:
            dx_hat_in = 2 * np.array([50, 100, 200, 400, 800, 1600])
        elif beta == 1:
            dx_hat_in = 4 * np.array([50])#, 100, 200, 400, 800, 1600])
        else:
            print('beta must be =0 or =1')
            sys.exit()
        C_res = ['4D', '8D', '16D', '32D', '64D', '128D']


elif case_in=='dry':
    print('using dry CBL case')
    times_list = ['13800']
    time_in = times_list[t_in]
    path_f = f'/storage/silver/greybls/si818415/dry_CBL/'
    folder_f = 'first_filt/'
    folder_ff = 'second_filt/'
    file_f = f'cbl_{time_in}_'
    Delta=20

    if filtering_filters == False:
        dx_bar_in = np.array([20, 20, 20, 20, 20, 20])
        dx_hat_in = np.array([40, 80, 160, 320, 640, 1280])
        C_res = ['2D', '4D', '8D', '16D', '32D', '64D']
        scalar = ['momentum', 'th']

    elif filtering_filters == True:
        dx_bar_in = 2*np.array([20, 40, 80, 160, 320, 640])
        if beta == 0:
            dx_hat_in = 2*np.array([40, 80, 160, 320, 640, 1280])
        elif beta == 1:
            dx_hat_in = 4 * np.array([40])#, 80, 160, 320, 640, 1280])
        else:
            print('beta must be =0 or =1')
            sys.exit()
        C_res = ['4D', '8D', '16D', '32D', '64D', '128D']
        scalar = ['momentum', 'f(th_on_p)_r']


else:
    print('case not recognised')

# if filtering_filters == True:
#     os.makedirs(path_f+folder_ff, exist_ok = True)




for it in range(len(dx_hat_in) - nfilt):
    i = int(it+nfilt)
    print(f'computing filter ga0{i}')

    if filtering_filters == True:
        print('using 2nd filt')
        file_in = file_f + f'gaussian_filter_ga0{i}_gaussian_filter_ga0{beta}.nc'
        data_in = path_f + folder_ff + file_in
        print('reading files', data_in)

        os.makedirs(path_f + folder_ff + '/LM/', exist_ok = True)
        dataset_name = [path_f + folder_ff + 'LM/' + file_f + f'Cs_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_ff + 'LM/' + file_f + f'C_{th_type}_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_ff + 'LM/' + file_f + f'C_qt_{dx_bar_in[i]}_{dx_hat_in[i]}.nc']

    elif filtering_filters == False:
        print('using 1st filt')
        file_in = file_f + f'gaussian_filter_ga0{i}.nc'
        data_in = path_f + folder_f + file_in
        print('reading files', data_in)
        os.makedirs(path_f + folder_f + 'LM/', exist_ok=True)
        dataset_name = [path_f + folder_f + 'LM/' + file_f + f'Cs_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                         path_f + folder_f + 'LM/' + file_f + f'C_{th_type}_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
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
        elif scalar_in == 'th' or scalar_in == 'th_tot' or scalar_in == 'f(th_on_p)_r' \
                or scalar_in == 'th_e' or scalar_in == 'th_L' or scalar_in == 'th_v':
            scalar_index = 1
        elif scalar_in == 'q_total' or scalar_in == 'q_vapour' or scalar_in == 'qv' or scalar_in == 'qt':
            scalar_index = 2
        else:
            print('scalar not set to momentum, th, or q_total')

        file_setup = dataset_name[scalar_index]
        # ds = xr.Dataset()
        # ds_in = {'file':dataset_name[scalar_index], 'ds': ds}
        # ds.to_netcdf(file_setup, mode='w')

        if scalar_in == 'momentum':
            print('about to start the Cs routine')
            #z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
            dy_s.Cs(file_save_to=file_setup, ingrid=mygrid, save_all=set_save_all, **DX_in)
            # = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        else:
            print(f'about to start the C_{scalar_in} routine')
            #z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
            dy_s.C_scalar(scalar=scalar_in, file_save_to=file_setup, ingrid=mygrid, save_all=set_save_all, **DX_in)
