import numpy as np
import os
import xarray as xr

import subfilter.filters as filt
import subfilter.subfilter as sf
import subfilter.utils as ut
import subfilter.utils.deformation as defm
from subfilter.utils.dask_utils import re_chunk
import argparse

import dynamic_functions as dyn
import dask
import subfilter

beta=True

parser = argparse.ArgumentParser()
# parser.add_argument('--times', type=str, default='25200')
parser.add_argument('--time_ind', type=int, default=0)
parser.add_argument('--start_filt', type=int, default=0)
#parser.add_argument('--n_filts', type=int, default=6)
parser.add_argument('--case_in', type=str, default='BOMEX')

args = parser.parse_args()
set_time_ind = args.time_ind
filters_start = args.start_filt
#how_many_filters = args.n_filts
how_many_filters = filters_start + 1
case = args.case_in


if case == 'ARM':
    times = ['18000', '25200', '32400', '39600']
else:
    times = ['14400']
set_time = times[set_time_ind]


opt_BOMEX = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,
          }

opt_ARM = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'override': True,
        'th_ref': 300.0,
        'dx': 25.0,
        'dy': 25.0,
        'domain' : 19.2,
          }


if case == 'BOMEX':
    opt = opt_BOMEX
    if beta==True:
        homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/'
        dir_cloud = homedir + 'contours/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
    else:
        homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/'
        dir_cloud = homedir + f'BOMEX_m0020_g0800_all_{set_time}_gaussian_filter_ga0'
    mydir = homedir + f"BOMEX_m0020_g0800_all_{set_time}_gaussian_filter_"

elif case == 'ARM':
    opt = opt_ARM
    if beta == True:
        homedir = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/'
        dir_cloud = homedir + f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'
    else:
        homedir = '/work/scratch-pw3/apower/ARM/corrected_sigmas/'
        dir_cloud = homedir + f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'


    mydir = homedir + f"diagnostics_3d_ts_{set_time}_gaussian_filter_"


dirs = [dir_cloud]#mydir,

if beta == True:
    res = ['0', '1', '2', '3', '4', '5']
    num_filts = ['0', '1']
else:
    #res = ['0', '1', '2', '3', '4', '5']
    res = ['2D', '4D', '8D', '16D', '32D', '64D']
    num_filts = ['0']
vars = ['Cs_', 'C_th_', 'C_qt_']

outdir = homedir +'smoothed_LM_HR_fields/'
os.makedirs(outdir, exist_ok = True)

filter_name = 'running_mean'
width_list = np.array([2])

#Note short serial queue on JASMIN times out after 3 filter scales
ingrid = 'p' #its on p but saved with the z coord saying z rather than zn woops



for i, indir in enumerate(dirs):
    for j in range(how_many_filters - filters_start): #, res[j] in enumerate(res):
        for l, filt_2nd in enumerate(num_filts):
            if indir == mydir:
                for k, var_in in enumerate(vars):
                    if beta == True:
                        file_in = f'{indir}{var_in}{res[j + filters_start]}_{filt_2nd}.nc'
                    else:
                        file_in = f'{indir}{var_in}{res[j + filters_start]}.nc'

                    if var_in == 'Cs_':
                        if case == 'BOMEX' and beta == True:
                            var_names = [f'f(LM_field_on_{ingrid})', f'f(MM_field_on_{ingrid})']
                        else:
                            var_names = ['LM_field', 'MM_field']

                    if var_in == 'C_th_':
                        if case == 'BOMEX' and beta == True:
                                var_names = [f'f(HR_th_field_on_{ingrid})', f'f(RR_th_field_on_{ingrid})']
                        else:
                            var_names = ['HR_th_field', 'RR_th_field']

                    if var_in == 'C_qt_':
                        if beta==True:
                            if case == 'BOMEX':
                                var_names = [f'f(HR_q_total_f_field_on_{ingrid})',
                                             f'f(RR_q_total_f_field_on_{ingrid})']
                            else:
                                var_names = ['HR_q_total_f_field', 'RR_q_total_f_field']
                        else:
                            var_names = ['HR_q_total_field', 'RR_q_total_field']

                    ds_in = xr.open_dataset(file_in)
                    time_data = ds_in['time']
                    times = time_data.data
                    nt = len(times)

                    dx_in = float(opt['dx'])
                    domain_in = float(opt['domain'])
                    N = int((domain_in * (1000)) / dx_in)

                    cutoff = 0.000001
                    sigma = 1

                    dask.config.set({"array.slicing.split_large_chunks": True})
                    [itime, iix, iiy, iiz] = ut.string_utils.get_string_index(ds_in.dims, ['time', 'x', 'y', 'zn'])
                    timevar = list(ds_in.dims)[itime]
                    xvar = list(ds_in.dims)[iix]
                    yvar = list(ds_in.dims)[iiy]
                    zvar = list(ds_in.dims)[iiz]
                    max_ch = 4*subfilter.global_config['chunk_size']

                    # This is a rough way to estimate chunck size
                    nch = np.min([int(ds_in.dims[xvar] / (2 ** int(np.log(ds_in.dims[xvar]
                                                                          * ds_in.dims[yvar]
                                                                          * ds_in.dims[zvar]
                                                                          / max_ch) / np.log(2) / 2))),
                                  ds_in.dims[xvar]])

                    ds_in.close()

                    dataset = xr.open_dataset(file_in, chunks={timevar: 1,
                                                               xvar: nch, yvar: nch,
                                                               'z': 'auto', 'zn': 'auto'})  # preprocess: check versions

                    ref_dataset = None
                    fname = filter_name

                    derived_data, exists = \
                        sf.setup_derived_data_file(file_in, outdir, fname,
                                                   opt, override=True)

                    filter_list = list([])

                    for ni, filt_set in enumerate(width_list):
                        print(filt_set)
                        if filter_name == 'gaussian':
                            filter_id = 'filter_ga{:02d}'.format(ni)
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name, npoints=N,
                                                      sigma=filt_set, width=-1,
                                                      delta_x=dx_in, cutoff=cutoff)
                        elif filter_name == 'wave_cutoff':
                            filter_id = 'filter_wc{:02d}'.format(ni)
                            twod_filter = filt.Filter(filter_id, filter_name,
                                                      wavenumber=filt_set,
                                                      width=-1, npoints=N,
                                                      delta_x=dx_in)
                        elif filter_name == 'running_mean':
                            filter_id = 'filter_rm{:02d}'.format(ni)
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name,
                                                      width=filt_set,
                                                      npoints=N,
                                                      delta_x=dx_in)
                        else:
                            print('Filter name not defined')
                            break

                        filter_list.append(twod_filter)

                    print(filter_list)

                    for nj, new_filter in enumerate(filter_list):
                        print("Processing using filter: ")
                        print(new_filter)

                        filtered_data, exists = \
                            sf.setup_filtered_data_file(file_in, outdir, fname,
                                                        opt, new_filter, override=True)
                        if exists:
                            print('Derived data file exists')

                        else:
                            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                                 derived_data, filtered_data,
                                                                 opt, new_filter,
                                                                 var_list=var_names,
                                                                 grid=ingrid)

                        filtered_data['ds'].close()
                    derived_data['ds'].close()
                    dataset.close()
            else:

                if beta == True:
                    file_in = f'{indir}{res[j+filters_start]}_gaussian_filter_ga0{filt_2nd}.nc'

                else:
                    file_in = f'{indir}{j+filters_start}.nc'


                if case == 'BOMEX' and beta == True:
                    var_names = [f'f(f(q_cloud_liquid_mass_on_{ingrid})_r_on_{ingrid})_r',
                                  f'f(f(th_v_on_{ingrid})_r_on_{ingrid})_r',
                                  f'f(f(w_on_{ingrid})_r_on_{ingrid})_r',
                                  f'f(f(u_on_{ingrid})_r_on_{ingrid})_r',
                                  f'f(f(v_on_{ingrid})_r_on_{ingrid})_r',
                                  f'f(f(w_on_{ingrid}.w_on_{ingrid})_r_on_{ingrid})_r']#,
                                  #f'f(f(w_on_{ingrid}.th_v_on_{ingrid})_r_on_{ingrid})_r' ] #,
                                  # f'f(f(w_on_{ingrid}.q_cloud_liquid_mass_on_{ingrid})_r_on_{ingrid})_r',
                                  # f'f(f(w_on_{ingrid}.th_on_{ingrid})_r_on_{ingrid})_r']

                else:
                    var_names = [f'f(q_cloud_liquid_mass_on_{ingrid})_r', f'f(th_v_on_{ingrid})_r',
                                 f'f(w_on_{ingrid})_r',
                             f'f(u_on_{ingrid})_r',
                             f'f(v_on_{ingrid})_r',
                             f'f(w_on_{ingrid}.w_on_{ingrid})_r']#,
                             # f'f(w_on_{ingrid}.th_v_on_{ingrid})_r'
                             # ]
                                 #
                                 # f'f(w_on_{ingrid}.q_cloud_liquid_mass_on_{ingrid})_r',
                                 # f'f(w_on_{ingrid}.th_on_{ingrid})_r',
                                 # f'f(w_on_{ingrid}.q_total_on_{ingrid})_r'


                ds_in = xr.open_dataset(file_in)
                time_data = ds_in['time']
                times = time_data.data
                nt = len(times)

                dx_in = float(opt['dx'])
                domain_in = float(opt['domain'])
                N = int((domain_in * (1000)) / dx_in)

                cutoff = 0.000001
                sigma = 1

                dask.config.set({"array.slicing.split_large_chunks": True})
                [itime, iix, iiy, iiz] = ut.string_utils.get_string_index(ds_in.dims, ['time', 'x', 'y', 'zn'])
                timevar = list(ds_in.dims)[itime]
                xvar = list(ds_in.dims)[iix]
                yvar = list(ds_in.dims)[iiy]
                zvar = list(ds_in.dims)[iiz]
                max_ch = 4*subfilter.global_config['chunk_size']

                # This is a rough way to estimate chunck size
                nch = np.min([int(ds_in.dims[xvar] / (2 ** int(np.log(ds_in.dims[xvar]
                                                                      * ds_in.dims[yvar]
                                                                      * ds_in.dims[zvar]
                                                                      / max_ch) / np.log(2) / 2))),
                                                                         ds_in.dims[xvar]])

                ds_in.close()

                dataset = xr.open_dataset(file_in, chunks={timevar: 1,
                                                           xvar: nch, yvar: nch,
                                                           'z': 'auto', 'zn': 'auto'})  # preprocess: check versions

                ref_dataset = None
                fname = filter_name

                derived_data, exists = \
                    sf.setup_derived_data_file(file_in, outdir, fname,
                                               opt, override=True)

                filter_list = list([])

                for ni, filt_set in enumerate(width_list):
                    print(filt_set)
                    if filter_name == 'gaussian':
                        filter_id = 'filter_ga{:02d}'.format(ni)
                        twod_filter = filt.Filter(filter_id,
                                                  filter_name, npoints=N,
                                                  sigma=filt_set, width=-1,
                                                  delta_x=dx_in, cutoff=cutoff)
                    elif filter_name == 'wave_cutoff':
                        filter_id = 'filter_wc{:02d}'.format(ni)
                        twod_filter = filt.Filter(filter_id, filter_name,
                                                  wavenumber=filt_set,
                                                  width=-1, npoints=N,
                                                  delta_x=dx_in)
                    elif filter_name == 'running_mean':
                        filter_id = 'filter_rm{:02d}'.format(ni)
                        twod_filter = filt.Filter(filter_id,
                                                  filter_name,
                                                  width=filt_set,
                                                  npoints=N,
                                                  delta_x=dx_in)
                    else:
                        print('Filter name not defined')
                        break

                    filter_list.append(twod_filter)

                print(filter_list)

                for nj, new_filter in enumerate(filter_list):
                    print("Processing using filter: ")
                    print(new_filter)

                    filtered_data, exists = \
                        sf.setup_filtered_data_file(file_in, outdir, fname,
                                                    opt, new_filter, override=True)
                    if exists:
                        print('Derived data file exists')

                    else:
                        field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                             derived_data, filtered_data,
                                                             opt, new_filter,
                                                             var_list=var_names,
                                                             grid=ingrid)

                    filtered_data['ds'].close()
            derived_data['ds'].close()
            dataset.close()