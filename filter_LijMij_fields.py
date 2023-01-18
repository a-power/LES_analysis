import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['14400'] # ,'12600', '16200', '18000'
homedir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/'
mydir = homedir + 'LijMij_HjRj/BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_cloud = homedir + 'q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

dirs = [mydir, dir_cloud]


outdir_og = '/work/scratch-pw2/apower/'
outdir = outdir_og + '20m_gauss_dyn' +'/filtered_LM_HR_fields/'

os.makedirs(outdir, exist_ok = True)

filter_name = 'running_mean'  # "wave_cutoff"
width_list = np.array([2, 3, 4]) #dont forget CHANGE start time if youre short-serial filtering  #([20, 40, 80] ([160, 320, 640])

start=0
#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'w'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,
          }

for j in range(len(set_time)):
        for i, indir in enumerate(dirs):

            file_in = f'{indir}{res_in}_all_{time_in}_gaussian_filter_{filtered_data}.nc'

            ds_in = xr.open_dataset(file_in)
            time_data = ds_in['time_series_600_600']
            times = time_data.data
            nt = len(times)

            dx_in = float(opt['dx'])
            domain_in = float(opt['domain'])
            N = int((domain_in * (1000)) / dx_in)

            filter_name = filt_in
            width = -1
            cutoff = 0.000001

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

            if ref_file is not None:
                ref_dataset = xr.open_dataset(indir + ref_file)
            else:
                ref_dataset = None

            fname = filter_name

            derived_data, exists = \
                sf.setup_derived_data_file(file_in, odir, fname,
                                           opt, override=True)

            filter_list = list([])

            for i, filt_set in enumerate(filt_scale):
                print(filt_set)
                if filter_name == 'gaussian':
                    filter_id = 'filter_ga{:02d}'.format(i + start_point)
                    twod_filter = filt.Filter(filter_id,
                                              filter_name, npoints=N,
                                              sigma=filt_set, width=width,
                                              delta_x=dx_in, cutoff=cutoff)
                elif filter_name == 'wave_cutoff':
                    filter_id = 'filter_wc{:02d}'.format(i + start_point)
                    twod_filter = filt.Filter(filter_id, filter_name,
                                              wavenumber=filt_set,
                                              width=width, npoints=N,
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
                    sf.setup_filtered_data_file(file_in, odir, fname,
                                                opt, new_filter, override=True)
                if exists:
                    print('Derived data file exists')
                else:

                    var_list = [
                        "u",
                        "v",
                        "w",
                        "th",
                        "th_v",
                        "th_L",
                        "q_total",
                        "q_vapour",
                        "q_cloud_liquid_mass"
                    ]

                    field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                         derived_data, filtered_data,
                                                         opt, new_filter,
                                                         var_list=var_list,
                                                         grid=ingrid)

                    var_list = [["u", "u"],
                                ["u", "v"],
                                ["u", "w"],
                                ["v", "v"],
                                ["v", "w"],
                                ["w", "w"],
                                ["u", "th"],
                                ["v", "th"],
                                ["w", "th"],
                                ["u", "q_total"],
                                ["v", "q_total"],
                                ["w", "q_total"],
                                ["w", "th_L"],
                                ["w", "q_vapour"],
                                ["w", "q_cloud_liquid_mass"],
                                ["th_L", "th_L"],
                                ["th_L", "q_total"],
                                ["q_total", "q_total"],
                                ["th_L", "q_vapour"],
                                ["th_L", "q_cloud_liquid_mass"]
                                ]

                    quad_field_list = sf.filter_variable_pair_list(dataset,
                                                                   ref_dataset,
                                                                   derived_data, filtered_data,
                                                                   opt, new_filter,
                                                                   var_list=var_list,
                                                                   grid=ingrid)
                    deform = defm.deformation(dataset,
                                              ref_dataset,
                                              derived_data,
                                              opt, ingrid)

                    dth_dx = dyn.ds_dxi('th', dataset, ref_dataset, opt, ingrid)
                    dth_dx.name = 'dth_dx'
                    dth_dx = re_chunk(dth_dx)

                    dq_dx = dyn.ds_dxi('q_total', dataset, ref_dataset, opt, ingrid)
                    dq_dx.name = 'dq_dx'
                    dq_dx = re_chunk(dq_dx)

                    S_ij_temp, abs_S_temp = defm.shear(deform, no_trace=False)

                    S_ij = 1 / 2 * S_ij_temp
                    S_ij.name = 'S_ij'
                    S_ij = re_chunk(S_ij)

                    abs_S = np.sqrt(abs_S_temp)
                    abs_S.name = "abs_S"
                    abs_S = re_chunk(abs_S)

                    S_ij_filt = sf.filter_field(S_ij, filtered_data,
                                                opt, new_filter)

                    abs_S_filt = sf.filter_field(abs_S, filtered_data,
                                                 opt, new_filter)

                    dth_dx_filt = sf.filter_field(dth_dx, filtered_data,
                                                  opt, new_filter)

                    dq_dx_filt = sf.filter_field(dq_dx, filtered_data,
                                                 opt, new_filter)

                    S_ij_abs_S = S_ij * abs_S
                    S_ij_abs_S.name = 'S_ij_abs_S'
                    S_ij_abs_S = re_chunk(S_ij_abs_S)

                    S_ij_abs_S_hat_filt = sf.filter_field(S_ij_abs_S, filtered_data,
                                                          opt, new_filter)

                    abs_S_dth_dx = dth_dx * abs_S
                    abs_S_dth_dx.name = 'abs_S_dth_dx'
                    abs_S_dth_dx = re_chunk(abs_S_dth_dx)

                    abs_S_dth_dx_filt = sf.filter_field(abs_S_dth_dx, filtered_data,
                                                        opt, new_filter)

                    abs_S_dq_dx = dq_dx * abs_S
                    abs_S_dq_dx.name = 'abs_S_dq_dx'
                    abs_S_dq_dx = re_chunk(abs_S_dq_dx)

                    abs_S_dq_dx_filt = sf.filter_field(abs_S_dq_dx, filtered_data,
                                                       opt, new_filter)

                filtered_data['ds'].close()
            derived_data['ds'].close()
            dataset.close()
            return

