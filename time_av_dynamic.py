import numpy as np
import xarray as xr
import dynamic as dy
import subfilter as sf
import dask
from netCDF4 import Dataset

def time_av_dyn_timeloop(dx_in, time_in, filt, filt_scale, indir, odir, opt, grid, domain_in=16, ref_file = None):

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

        dataset = xr.open_dataset(file_in, chunks={timevar: 1,
                                                      xvar: nch, yvar: nch,
                                                      'z': 'auto', 'zn': 'auto'})[t_in,...]  #preprocess: check versions

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
            sf.setup_derived_data_file(indir, odir+"/"+str(timevar[t_in]), fname,
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

            filtered_data['ds'].close()
        derived_data['ds'].close()
        dataset.close()
    return



def time_av_dyn(dx_in, time_in, filt, filt_scale, indir, odir, opt, grid, domain_in=16, ref_file = None):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """

    file_in = f'{indir}{dx_in}/diagnostic_files/BOMEX_m{dx_in}_all_{time_in}.nc'
    ds_in = xr.open_dataset(file_in)
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

    dataset = xr.open_dataset(file_in, chunks={timevar: nt,
                                                      xvar: nch, yvar: nch,
                                                      'z': 'auto', 'zn': 'auto'}) #preprocess: check versions

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
        sf.setup_derived_data_file(indir, odir, fname,
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
            sf.setup_filtered_data_file(file_in, odir, fname,
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

        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()
    return


def time_av_Cs(indir, dx, dx_hat):

    """ function takes in:  """

    file_in = f'{indir}____.nc'
    ds_in = xr.open_ds_in(file_in)
    time_data = ds_in['time']
    times = time_data.data
    nt = len(times)
    ds_in.close()

    for t_in in(nt):

        data_in = Dataset(file_in, mode='r')

        Lij = dy.L_ij_sym(data_in)

        hat_Sij_abs_S = data_in.variables['S_ij_abs_S_r'][:, t_in, :, :, :]
        hat_Sij = data_in.variables['S_ij_r'][:, t_in, :, :, :]
        abs_S_hat_Sij_hat = dy.abs_S_hat_S_ij_hat(hat_Sij)
        Mij = dy.M_ij(dx, dx_hat, hat_Sij, hat_Sij_abs_S)