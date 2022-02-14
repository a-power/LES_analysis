import numpy as np
import xarray as xr
import dynamic as dyn
import subfilter as sf
import filters as filt
import dask

def time_av_dyn(dx_in, time_in, filt, filt_scale, indir, odir, opt, grid, domain_in=16, ref_file = None):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """

    file_in = f'{indir}{dx_in}/diagnostic_files/BOMEX_m{dx_in}_all_{time_in}.nc'
    ds_in = xr.open_ds_in(file_in)
    time_data = ds_in['time_series_600_600']
    times = time_data.data
    nt = len(times)
    z_in = ds_in['z']
    z = z_in.data
    np.save(f'files/{dx_in}_z', z)

    N = domain_in*(1000)/dx_in
    filter_name = filt
    width = -1
    cutoff = 0.000001

    dask.config.set({"array.slicing.split_large_chunks": True})
    [itime, iix, iiy, iiz] = sf.find_var(ds_in.dims, ['time', 'x', 'y', 'z'])
    timevar = list(ds_in.dims)[itime]
    xvar = list(ds_in.dims)[iix]
    yvar = list(ds_in.dims)[iiy]
    zvar = list(ds_in.dims)[iiz]
    max_ch = sf.subfilter_setup['chunk_size']

    # This is a rough way to estimate chunck size
    nch = np.min([int(ds_in.dims[xvar] / (2 ** int(np.log(ds_in.dims[xvar]
                                                            * ds_in.dims[yvar]
                                                            * ds_in.dims[zvar]
                                                            / max_ch) / np.log(2) / 2))),
                  ds_in.dims[xvar]])

    ds_in.close()

    for t_in in(nt):

        dataset = xr.open_dataset(file_in, chunks={timevar: t_in,
                                                      xvar: nch, yvar: nch,
                                                      'z': 'auto', 'zn': 'auto'})

        if ref_file is not None:
            ref_dataset = xr.open_dataset(dir + ref_file)
        else:
            ref_dataset = None

        od = sf.options_database(dataset)
        if od is None:
            dx = opt['dx']
            dy = opt['dy']
        else:
            dx = float(od['dxx'])
            dy = float(od['dyy'])

        fname = filter_name

        derived_data, exists = \
            sf.setup_derived_data_file(indir, odir+str(timevar[t_in]), fname,
                                       opt, override=True)

        filter_list = list([])

        for i, filt_set in enumerate(filt_scale):
            print(filt_set)
            if filter_name == 'gaussian':
                filter_id = 'filter_ga{:02d}'.format(i)
                twod_filter = filt.Filter(filter_id,
                                          filter_name, npoints=N,
                                          sigma=filt_set, width=width,
                                          delta_x=dx, cutoff=cutoff)
            elif filter_name == 'wave_cutoff':
                filter_id = 'filter_wc{:02d}'.format(i)
                twod_filter = filt.Filter(filter_id, filter_name,
                                          wavenumber=filt_set,
                                          width=width, npoints=N,
                                          delta_x=dx)
            elif filter_name == 'running_mean':
                filter_id = 'filter_rm{:02d}'.format(i)
                twod_filter = filt.Filter(filter_id,
                                          filter_name,
                                          width=filt_set,
                                          npoints=N,
                                          delta_x=dx)

            filter_list.append(twod_filter)

        # Add whole domain filter
        filter_name = 'domain'
        filter_id = 'filter_do{:02d}'.format(len(filter_list))
        twod_filter = filt.Filter(filter_id, filter_name, delta_x=dx)
        filter_list.append(twod_filter)

        print(filter_list)

        for j, new_filter in enumerate(filter_list):
            print("Processing using filter: ")
            print(new_filter)

            filtered_data, exists = \
                sf.setup_filtered_data_file(file_in, odir+str(timevar[t_in]), fname,
                                            opt, new_filter, override=True)
            if exists:
                print('Derived data file exists')
            else:

                var_list = ["u",
                            "v",
                            "w",
                            "th"
                            ]

                field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                     derived_data, filtered_data,
                                                     opt, new_filter,
                                                     var_list=var_list,
                                                     grid=grid)

                var_list = [["w", "th"],
                            ["u", "u"],
                            ["u", "v"],
                            ["u", "w"],
                            ["v", "v"],
                            ["v", "w"],
                            ["w", "w"]
                            ]

                quad_field_list = sf.filter_variable_pair_list(dataset,
                                                               ref_dataset,
                                                               derived_data, filtered_data,
                                                               opt, new_filter,
                                                               var_list=var_list,
                                                               grid=grid)

                ##################### Lij filtered

                u_hat = filtered_data['u_on_p_r']
                v_hat = filtered_data['v_on_p_r']
                w_hat = filtered_data['w_on_p_r']
                uu_hat = filtered_data['u_on_p.u_on_p_r']
                uv_hat = filtered_data['u_on_p.v_on_p_r']
                uw_hat = filtered_data['u_on_p.w_on_p_r']
                vv_hat = filtered_data['v_on_p.v_on_p_r']
                vw_hat = filtered_data['v_on_p.w_on_p_r']
                ww_hat = filtered_data['w_on_p.w_on_p_r']

                L_ij = L_ij_sym(u_hat, v_hat, w_hat, uu_hat, uv_hat, uw_hat, vv_hat, vw_hat, ww_hat)
                L_ij.name = "L_ij"
                L_ij = save_field(filtered_data, L_ij)

                ### Lines 204 to 217 don't run - grid / indexing problem ###

                # deform_filt_r, deform_filt_s = sf.filtered_deformation(dataset,
                #                                      ref_dataset,
                #                                      derived_data, filtered_data,
                #                                      opt, new_filter)#,
                # grid = grid)

                #             S_ij_temp_hat, abs_S_temp_hat = sf.shear(deform_filt_r)

                #             S_ij_hat = 1/2*S_ij_temp_hat
                #             abs_S_hat = np.sqrt(abs_S_temp_hat)

                ###############################################

                deform = sf.deformation(dataset,
                                        ref_dataset,
                                        derived_data,
                                        opt)

                S_ij_temp, abs_S_temp = sf.shear(deform)

                S_ij = 1 / 2 * S_ij_temp
                S_ij.name = 'S_ij'
                abs_S = np.sqrt(abs_S_temp)
                abs_S.name = "abs_S"

                S_ij_filt = sf.filter_field(S_ij, filtered_data,
                                            opt, new_filter)

                abs_S_filt = sf.filter_field(abs_S, filtered_data,
                                             opt, new_filter)

                S_ij_abs_S = S_ij * abs_S
                S_ij_abs_S.name = 'S_ij_abs_S'

                S_ij_abs_S_hat_filt = sf.filter_field(S_ij_abs_S, filtered_data,
                                                      opt, new_filter)

                #             filt_scale_copy = filt_scale.copy()
                #             filt_scale_copy.append("domain")
                #             filt_scale = xr.DataArray(data=filt_scale_copy[j], name = "filt_scale")
                #             filt_scale = sf.save_field(filtered_data, filt_scale)

            filtered_data['ds'].close()
        derived_data['ds'].close()
        dataset.close()


    return