
import numpy as np
import os
import xarray as xr

import subfilter.filters as filt
import subfilter.subfilter as sf
import subfilter.utils as ut
import subfilter.utils.deformation as defm
from subfilter.utils.dask_utils import re_chunk

import dynamic_functions as dyn
import dask
import subfilter

homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/'
mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_cloud = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

dirs = [mydir, dir_cloud]
res = ['2D', '4D', '8D', '16D', '32D', '64D']
vars = ['Cs_', 'C_th_', 'C_qt_']

outdir = homedir +'filtered_LM_HR_fields/'
os.makedirs(outdir, exist_ok = True)

filter_name = 'running_mean'
width_list = np.array([2]) #dont forget CHANGE start time if youre short-serial filtering  #([20, 40, 80] ([160, 320, 640])

start_point=0
#Note short serial queue on JASMIN times out after 3 filter scales
ingrid = 'w' #its on p but saved with the z coord saying z rather than zn woops
cloud_grid = 'p'

opt = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,
          }

for i, indir in enumerate(dirs):
    for j, data_in_scale in enumerate(res):

        if indir == mydir:
            for k, var_in in enumerate(vars):
                file_in = f'{indir}{var_in}{data_in_scale}.nc'

                if var_in == 'Cs_':
                    var_names = ['LM_field', 'MM_field']

                if var_in == 'C_th_':
                    var_names = ['HR_th_field', 'RR_th_field']

                if var_in == 'C_qt_':
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
                [itime, iix, iiy, iiz] = ut.string_utils.get_string_index(ds_in.dims, ['time', 'x', 'y', 'z'])
                timevar = list(ds_in.dims)[itime]
                xvar = list(ds_in.dims)[iix]
                yvar = list(ds_in.dims)[iiy]
                zvar = list(ds_in.dims)[iiz]
                max_ch = subfilter.global_config['chunk_size']

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

                for i, filt_set in enumerate(width_list):
                    print(filt_set)
                    if filter_name == 'gaussian':
                        filter_id = 'filter_ga{:02d}'.format(i + start_point)
                        twod_filter = filt.Filter(filter_id,
                                                  filter_name, npoints=N,
                                                  sigma=filt_set, width=-1,
                                                  delta_x=dx_in, cutoff=cutoff)
                    elif filter_name == 'wave_cutoff':
                        filter_id = 'filter_wc{:02d}'.format(i + start_point)
                        twod_filter = filt.Filter(filter_id, filter_name,
                                                  wavenumber=filt_set,
                                                  width=-1, npoints=N,
                                                  delta_x=dx_in)
                    elif filter_name == 'running_mean':
                        filter_id = 'filter_rm{:02d}'.format(i + start_point)
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

                for j, new_filter in enumerate(filter_list):
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
            file_in = f'{indir}{j}.nc'
            var_names = [f'f(q_cloud_liquid_mass_on_{cloud_grid})_r']

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
            [itime, iix, iiy, iiz] = ut.string_utils.get_string_index(ds_in.dims, ['time', 'x', 'y', 'z'])
            timevar = list(ds_in.dims)[itime]
            xvar = list(ds_in.dims)[iix]
            yvar = list(ds_in.dims)[iiy]
            zvar = list(ds_in.dims)[iiz]
            max_ch = subfilter.global_config['chunk_size']

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

            for i, filt_set in enumerate(width_list):
                print(filt_set)
                if filter_name == 'gaussian':
                    filter_id = 'filter_ga{:02d}'.format(i + start_point)
                    twod_filter = filt.Filter(filter_id,
                                              filter_name, npoints=N,
                                              sigma=filt_set, width=-1,
                                              delta_x=dx_in, cutoff=cutoff)
                elif filter_name == 'wave_cutoff':
                    filter_id = 'filter_wc{:02d}'.format(i + start_point)
                    twod_filter = filt.Filter(filter_id, filter_name,
                                              wavenumber=filt_set,
                                              width=-1, npoints=N,
                                              delta_x=dx_in)
                elif filter_name == 'running_mean':
                    filter_id = 'filter_rm{:02d}'.format(i + start_point)
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

            for j, new_filter in enumerate(filter_list):
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