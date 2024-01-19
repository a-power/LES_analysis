import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=2)
parser.add_argument('--case', type=str, default='ARM')
args = parser.parse_args()
t_in = args.t

case_in = args.case

av_type = 'all'
mygrid = 'p'

if case_in == 'BOMEX':
    path_f = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/'
    folder_ff = 'filtering_filtered/'
    file_f = 'BOMEX_m0020_g0800_all_'
    times_list = ['14400']
    Delta = 20
    scalar = ['momentum', 'th', 'q_total']
    dx_bar_in = [40, 80, 160, 320, 640, 1280]
    dx_hat_in = [80, 160, 320, 640, 1280, 2560]
    C_res = ['2D', '4D', '8D', '16D', '32D', '64D']

elif case_in == 'ARM':
    times_list = ['18000', '25200', '32400', '39600']
    time_in = times_list[t_in]
    path_f = '/work/scratch-pw3/apower/ARM/corrected_sigmas/'
    folder_ff = 'filtering_filtered/'
    file_f = f'diagnostics_3d_ts_{time_in}_'
    Delta = 25
    # dx_bar_in = [56, 103, 202, 401, 800, 1600]
    # dx_hat_in = [75, 144, 284, 566, 1132, 2263]
    dx_bar_in = [50]#, 100, 200, 400, 800, 1600]
    dx_hat_in = [100]#, 144, 284, 566, 1132, 2263]
    C_res = ['2D']#, '4D', '8D', '16D', '32D', '64D']
    scalar = ['th']#['momentum', 'th', 'q_total']

elif case_in=='dry':
    times_list = ['13800']
    time_in = times_list[t_in]
    path_f = f'/storage/silver/greybls/si818415/dry_CBL/'
    folder_ff = 'filtering_filtered/'
    file_f = f'cbl_{time_in}_'
    Delta=20
    scalar = ['momentum', 'th']
    dx_bar_in = [40, 80, 160, 320, 640, 1280]
    dx_hat_in = [80, 160, 320, 640, 1280, 2560]
    C_res = ['2D', '4D', '8D', '16D', '32D', '64D']

else:
    print('case not recognised')


os.makedirs(folder_ff, exist_ok = True)



set_save_all = 2


for i, C_res_in in enumerate(C_res):

    file_in = file_f + f'gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'
    data_in = path_f + folder_ff + file_in
    dataset_name = [path_f + folder_ff + file_f + f'Cs_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                     path_f + folder_ff + file_f + f'C_th_{dx_bar_in[i]}_{dx_hat_in[i]}.nc',
                     path_f + folder_ff + file_f + f'C_qt_{dx_bar_in[i]}_{dx_hat_in[i]}.nc']

    DX_in = {
        'indir': data_in,
        'dx_bar': dx_bar_in[i],
        'dx_hat': dx_hat_in[i]
    }

    for i, scalar_in in enumerate(scalar):

    ########################################################################
        #  = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        if scalar_in == 'momentum':
            z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
             dy_s.Cs(ingrid = mygrid, save_all = set_save_all, **DX_in)
            # = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        else:
            z_save, zn_save, C_sq_prof, C_prof, HR_prof, RR_prof, HR_field, RR_field = \
                dy_s.C_scalar(scalar=scalar_in, ingrid=mygrid, save_all = set_save_all, **DX_in)

        ds = xr.Dataset()
        if scalar_in == 'momentum':
            ds.to_netcdf(dataset_name[0], mode='w')
        elif scalar_in == 'th':
            ds.to_netcdf(dataset_name[0], mode='w')
        elif scalar_in == 'q_total':
            ds.to_netcdf(dataset_name[0], mode='w')
        else:
            print('scalar not set to momentum, th, or q_total')
        ds_in = {'file':dataset_name[i], 'ds': ds}

        save_field(ds_in, HR_field)
        save_field(ds_in, RR_field)
        save_field(ds_in, z_save)
        save_field(ds_in, zn_save)
        save_field(ds_in, C_sq_prof)
        save_field(ds_in, C_prof)
        save_field(ds_in, HR_prof)
        save_field(ds_in, RR_prof)
        #save_field(ds_in2 = \ #, C_sq_field_2D)
        # save_field(ds_in2, Hj_2D)
        # save_field(ds_in2, Rj_2D)

        ds.close()

        C_sq_prof = None
        C_prof = None
        HR_prof = None
        RR_prof = None
        HR_field = None
        RR_field = None
