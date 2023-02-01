import numpy as np
import xarray as xr

import subfilter.filters as filt
import subfilter.subfilter as sf
import subfilter.utils as ut
import subfilter.utils.deformation as defm
from subfilter.utils.dask_utils import re_chunk

import dynamic_functions as dyn
import dask
import subfilter

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

#code now has theta_l and _v, q_l, q_v were all added to the filtering code in this iteration


def run_dyn(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid, start_point=0,
            ref_file = None, time_name = 'time_series_600_600'):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """


    file_in = f'{indir}{res_in}/diagnostic_files/BOMEX_m{res_in}_all_{time_in}.nc'

    ds_in = xr.open_dataset(file_in)
    time_data = ds_in[time_name]
    times = time_data.data
    nt = len(times)

    dx_in = float(opt['dx'])
    domain_in = float(opt['domain'])
    N = int((domain_in*(1000))/dx_in)

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
                                                      'z': 'auto', 'zn': 'auto'}) #preprocess: check versions

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
            filter_id = 'filter_ga{:02d}'.format(i+start_point)
            twod_filter = filt.Filter(filter_id,
                                      filter_name, npoints=N,
                                      sigma=filt_set, width=width,
                                      delta_x=dx_in, cutoff=cutoff)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i+start_point)
            twod_filter = filt.Filter(filter_id, filter_name,
                                      wavenumber=filt_set,
                                      width=width, npoints=N,
                                      delta_x=dx_in)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}'.format(i+start_point)
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


def run_dyn_on_filtered(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid, filtered_data, start_point=0,
            ref_file = None, time_name = 'time_series_600_600'):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """

    file_in = f'{indir}{res_in}_all_{time_in}_gaussian_filter_{filtered_data}.nc'

    ds_in = xr.open_dataset(file_in)
    time_data = ds_in[time_name]
    times = time_data.data
    nt = len(times)

    dx_in = float(opt['dx'])
    domain_in = float(opt['domain'])
    N = int((domain_in*(1000))/dx_in)

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
                                                      'z': 'auto', 'zn': 'auto'}) #preprocess: check versions

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
            filter_id = 'filter_ga{:02d}'.format(i+start_point)
            twod_filter = filt.Filter(filter_id,
                                      filter_name, npoints=N,
                                      sigma=filt_set, width=width,
                                      delta_x=dx_in, cutoff=cutoff)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i+start_point)
            twod_filter = filt.Filter(filter_id, filter_name,
                                      wavenumber=filt_set,
                                      width=width, npoints=N,
                                      delta_x=dx_in)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}'.format(i+start_point)
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
                        "q_total_f"
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
                        ["u", "q_total_f"],
                        ["v", "q_total_f"],
                        ["w", "q_total_f"],
                        ["q_total", "q_total_f"]
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

            dq_dx = dyn.ds_dxi('q_total_f', dataset, ref_dataset, opt, ingrid)
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




def Cs(indir, dx, dx_hat, ingrid, save_all=2, reaxes=False):

    """ function takes in:

    save_all: 1 is for profiles, 2 is for fields, 3 is for all fields PLUS Lij and Mij"""

    file_in = f'{indir}'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time']
    times = time_data.data
    nt = len(times)
    print('lenght of the time array in Cs function is', nt)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    ij_data = ds_in['i_j']
    ij_s = ij_data.data

    ds_in.close()

    ds_in = xr.open_dataset(file_in)
    uu = ds_in[f's(u,u)_on_{ingrid}'].data[...]
    uv = ds_in[f's(u,v)_on_{ingrid}'].data[...]
    uw = ds_in[f's(u,w)_on_{ingrid}'].data[...]
    vv = ds_in[f's(v,v)_on_{ingrid}'].data[...]
    vw = ds_in[f's(v,w)_on_{ingrid}'].data[...]
    ww = ds_in[f's(w,w)_on_{ingrid}'].data[...]

    Lij = dyn.L_ij_sym_xarray(uu, uv, uw, vv, vw, ww)

    uu = None # Save storage
    uv = None # Save storage
    uw = None # Save storage
    vv = None # Save storage
    vw = None # Save storage
    ww = None # Save storage

    hat_Sij = ds_in['f(S_ij)_r'].data[...]
    hat_abs_S = ds_in['f(abs_S)_r'].data[...]

    if reaxes == True:
        hat_Sij_abs_S_temp = ds_in['f(S_ij_abs_S)_r'].data[...] # (time, x, y, z, ij) --> (ij, time, x, y, z)
        hat_Sij_abs_S = np.transpose(hat_Sij_abs_S_temp, axes=[4, 0, 1, 2, 3])
        hat_Sij_abs_S_temp = None
    else:
        hat_Sij_abs_S = ds_in['f(S_ij_abs_S)_r'].data[...]

    Mij = dyn.M_ij(dx, dx_hat, hat_Sij, hat_abs_S, hat_Sij_abs_S)

    hat_Sij_abs_S = None
    hat_Sij = None
    hat_abs_S=None

    if save_all==1:
        Cs_sq_prof, Cs_prof, LM_prof, MM_prof = dyn.Cs_profiles(Lij, Mij, return_all=1)
        Lij = None
        Mij = None

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[np.newaxis, ...], coords={'time' : [nt], 'z': z_s},
                                  dims=['time', "z"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[np.newaxis, ...], coords={'time' : [nt], 'z': z_s},
                               dims=['time', "z"], name='Cs_prof')

        LM_prof = xr.DataArray(LM_prof[np.newaxis, ...], coords={'time' : [nt], 'z': z_s},
                               dims=['time', "z"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[np.newaxis, ...], coords={'time' : [nt], 'z': z_s},
                               dims=['time', "z"], name='MM_prof')

        return Cs_sq_prof, Cs_prof, LM_prof, MM_prof

    if save_all==2:
        # Cs_sq_field = dyn.C_s_sq(Lij, Mij)

        Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field = dyn.Cs_profiles(Lij, Mij, return_all=2)

        Lij = None
        Mij = None

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[np.newaxis, ...], coords={'time' : [nt],'z': z_s},
                                  dims=['time', "z"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[np.newaxis, ...], coords={'time' : [nt],'z': z_s},
                               dims=['time', "z"], name='Cs_prof')

        LM_prof = xr.DataArray(LM_prof[np.newaxis, ...], coords={'time' : [nt],'z': z_s},
                               dims=['time', "z"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[np.newaxis, ...], coords={'time' : [nt],'z': z_s},
                               dims=['time', "z"], name='MM_prof')


        LM_field = xr.DataArray(LM_field, coords={'time' : times, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                                  dims = ["time", "x_p", "y_p", "z"], name = 'LM_field')

        MM_field = xr.DataArray(MM_field, coords={'time' : times, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                                  dims = ["time", "x_p", "y_p", "z"], name = 'MM_field')


        # Cs_sq_field = xr.DataArray(Cs_sq_field[np.newaxis, ...], coords={'time' : [nt], 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
        #                           dims = ["time", "x_p", "y_p", "z"], name = 'Cs_sq_field')

        return Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field  #, Cs_sq_field

    if save_all==3:

        Cs_sq_field = dyn.C_s_sq(Lij, Mij)

        print('in dynamic script the shape of Cs_sq_field is', np.shape(Cs_sq_field))

        Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field = dyn.Cs_profiles(Lij, Mij, return_all=2)

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                  dims=['time', "z"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name='Cs_prof')

        Cs_sq_field = xr.DataArray(Cs_sq_field[np.newaxis, ...],
                                   coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                   dims=["time", "x_p", "y_p", "z"], name='Cs_sq_field')

        LM_prof = xr.DataArray(LM_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name='MM_prof')

        LM_field = xr.DataArray(LM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name='LM_field')

        MM_field = xr.DataArray(MM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name='MM_field')
        if len(Mij.shape) == 5:

            print("number of times = ", (Mij.shape)[1])

            Mij_av = np.mean(Mij, 1)
            Mij = None
            Lij_av = np.mean(Lij, 1)
            Lij = None

            Lij = xr.DataArray(Lij_av[np.newaxis, ...], coords={'time': [nt], 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                                    dims=["time", "i_j", "x_p", "y_p", "z"], name='Lij')

            Mij = xr.DataArray(Mij_av[np.newaxis, ...], coords={'time': [nt], 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                                    dims=["time", "i_j", "x_p", "y_p", "z"], name='Mij')
        else:
            Lij = xr.DataArray(Lij[np.newaxis, ...],
                               coords={'time': [nt], 'i_j': ij_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                               dims=["time", "i_j", "x_p", "y_p", "z"], name='Lij')

            Mij = xr.DataArray(Mij[np.newaxis, ...],
                               coords={'time': [nt], 'i_j': ij_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                               dims=["time", "i_j", "x_p", "y_p", "z"], name='Mij')


        return Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field, Cs_sq_field, Lij, Mij

    else:
        Cs_sq_prof, Cs_prof = dyn.Cs_profiles(Lij, Mij, return_all=0)

        Cs_prof = xr.DataArray(Cs_sq_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                  dims=['time', "z"], name='Cs_sq_prof')

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[np.newaxis, ...], coords={'time' : [nt], 'z': z_s},
                                  dims=['time', "z"], name='Cs_sq_prof')

        Lij = None
        Mij = None

        return Cs_sq_prof, Cs_prof






def C_scalar(scalar, indir, dx, dx_hat, ingrid, save_all = 2, axisfix=False):
    """ function takes in:

    save_all: 1 is for profiles, 2 is for fields, 3 is for all fields PLUS Lij and Mij"""

    if scalar=='q_total':
        scalar_name='q'
    elif scalar == 'q_cloud_liquid_mass':
        scalar_name = 'q_l'
    elif scalar == 'q_vapour':
        scalar_name = 'q_v'
    elif scalar == 'th':
        scalar_name = 'th'
    else:
        print("scalar not recognised, only inputs available are 'th', 'q_cloud_liquid_mass', 'q_vapour', or 'q_total'.")
        return

    file_in = f'{indir}'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time']
    times = time_data.data
    nt = len(times)
    print('lenght of the time array in Cs function is', nt)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    j_data = ds_in['j']
    j_s = j_data.data

    ds_in.close()

    ds_in = xr.open_dataset(file_in)
    u_s = ds_in[f's(u,{scalar})_on_{ingrid}'].data[...]
    v_s = ds_in[f's(v,{scalar})_on_{ingrid}'].data[...]
    w_s = ds_in[f's(w,{scalar})_on_{ingrid}'].data[...]

    Hj = dyn.H_j(u_s, v_s, w_s)

    u_s = None  # Save storage
    v_s = None  # Save storage
    w_s = None  # Save storage

    hat_abs_S = ds_in['f(abs_S)_r'].data[...]
    ds_dx_hat = ds_in[f'f(d{scalar_name}_dx)_r'].data[...]

    ##########Rough axis fix###########

    if axisfix == True:
        HAT_abs_S_ds_dx_temp = ds_in[f'f(abs_S_d{scalar_name}_dx)_r'].data[...]
        HAT_abs_S_ds_dx = np.transpose(HAT_abs_S_ds_dx_temp, axes=[4, 0, 1, 2, 3])
        HAT_abs_S_ds_dx_temp = None
    else:
        HAT_abs_S_ds_dx = ds_in[f'f(abs_S_d{scalar_name}_dx)_r'].data[...]


    Rj = dyn.R_j(dx, dx_hat, hat_abs_S, ds_dx_hat, HAT_abs_S_ds_dx, beta=1)
    HAT_abs_S_ds_dx = None

    if save_all == 1:

        C_scalar_sq_prof, C_scalar_prof = dyn.C_scalar_profiles(Hj, Rj, return_all=0)

        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                        dims=['time', "z"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name=f'C_{scalar}_prof')

        return C_scalar_sq_prof, C_scalar_prof



    if save_all == 2:

        C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field = dyn.C_scalar_profiles(Hj, Rj, return_all=2)


        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                        dims=['time', "z"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name=f'C_{scalar}_prof')

        HR_prof = xr.DataArray(HR_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name=f'HR_{scalar}_prof')

        RR_prof = xr.DataArray(RR_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name=f'RR_{scalar}_prof')

        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                     dims=["time", "x_p", "y_p", "z"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                     dims=["time", "x_p", "y_p", "z"], name=f'RR_{scalar}_field')


        return C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field



    if save_all == 3:

        C_scalar_sq_field = dyn.C_scalar_sq(Hj, Rj)

        C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field = dyn.C_scalar_profiles(Hj, Rj, return_all=2)

        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                    dims=['time', "z"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                 dims=['time', "z"], name=f'C_{scalar}_prof')

        HR_prof = xr.DataArray(HR_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name=f'HR_{scalar}_prof')

        RR_prof = xr.DataArray(RR_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                               dims=['time', "z"], name=f'RR_{scalar}_prof')

        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                     dims=["time", "x_p", "y_p", "z"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                     dims=["time", "x_p", "y_p", "z"], name=f'RR_{scalar}_field')

        C_scalar_sq_field = xr.DataArray(C_scalar_sq_field[np.newaxis, ...],
                                     coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                     dims=["time", "x_p", "y_p", "z"], name=f'C_{scalar}_sq_field')

        if len(Hj.shape) == 5:

            print("number of times = ", (Hj.shape)[1])

            Hj_av = np.mean(Hj, 1)
            Hj = None
            Rj_av = np.mean(Rj, 1)
            Rj = None

            Rj = xr.DataArray(Rj_av[np.newaxis, ...],
                              coords={'time': [nt], 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                              dims=["time", "i_j", "x_p", "y_p", "z"], name='Hj')

            Hj = xr.DataArray(Hj_av[np.newaxis, ...],
                              coords={'time': [nt], 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                              dims=["time", "i_j", "x_p", "y_p", "z"], name='Rj')
        else:

            Rj = xr.DataArray(Rj[np.newaxis, ...],
                              coords={'time': [nt], 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                              dims=["time", "i_j", "x_p", "y_p", "z"], name='Hj')

            Hj = xr.DataArray(Hj[np.newaxis, ...],
                              coords={'time': [nt], 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                              dims=["time", "i_j", "x_p", "y_p", "z"], name='Rj')

        return C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field, C_scalar_sq_field, Hj, Rj


def LijMij_fields(scalar, indir, dx, dx_hat, ingrid):

    if scalar == 'q_total':
        scalar_name = 'q'
    elif scalar == 'q_cloud_liquid_mass':
        scalar_name = 'q_l'
    elif scalar == 'q_vapour':
        scalar_name = 'q_v'
    elif scalar == 'th':
        scalar_name = 'th'
    elif scalar == 'momentum':
        scalar_name = 's'
    else:
        print("scalar not recognised, only inputs available are 'th', 'q_cloud_liquid_mass', 'q_vapour', or 'q_total'.")
        return

    file_in = f'{indir}'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time']
    times = time_data.data
    print('time array is', times)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    ds_in.close()

    if scalar == 'momentum':

        ds_in = xr.open_dataset(file_in)
        uu = ds_in[f's(u,u)_on_{ingrid}'].data[...]
        uv = ds_in[f's(u,v)_on_{ingrid}'].data[...]
        uw = ds_in[f's(u,w)_on_{ingrid}'].data[...]
        vv = ds_in[f's(v,v)_on_{ingrid}'].data[...]
        vw = ds_in[f's(v,w)_on_{ingrid}'].data[...]
        ww = ds_in[f's(w,w)_on_{ingrid}'].data[...]

        Lij = dyn.L_ij_sym_xarray(uu, uv, uw, vv, vw, ww)

        uu = None  # Save storage
        uv = None  # Save storage
        uw = None  # Save storage
        vv = None  # Save storage
        vw = None  # Save storage
        ww = None  # Save storage

        hat_Sij = ds_in['f(S_ij)_r'].data[...]
        hat_abs_S = ds_in['f(abs_S)_r'].data[...]

        hat_Sij_abs_S = ds_in['f(S_ij_abs_S)_r'].data[...]

        Mij = dyn.M_ij(dx, dx_hat, hat_Sij, hat_abs_S, hat_Sij_abs_S)

        hat_Sij_abs_S = None
        hat_Sij = None
        hat_abs_S = None


        LM_field = np.zeros_like(Lij[0, ...])
        MM_field = np.zeros_like(Mij[0, ...])

        for it in range(0, 6):
            if it in [0, 3, 5]:

                LM_field += Lij[it, ...] * Mij[it, ...]
                MM_field += Mij[it, ...] * Mij[it, ...]

            else:
                LM_field += 2 * (Lij[it, ...] * Mij[it, ...])
                MM_field += 2 * (Mij[it, ...] * Mij[it, ...])

        LM_field = xr.DataArray(LM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name='LM_field')

        MM_field = xr.DataArray(MM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name='MM_field')

        return LM_field, MM_field


    else:
        ds_in = xr.open_dataset(file_in)
        u_s = ds_in[f's(u,{scalar})_on_{ingrid}'].data[...]
        v_s = ds_in[f's(v,{scalar})_on_{ingrid}'].data[...]
        w_s = ds_in[f's(w,{scalar})_on_{ingrid}'].data[...]

        Hj = dyn.H_j(u_s, v_s, w_s)

        u_s = None  # Save storage
        v_s = None  # Save storage
        w_s = None  # Save storage

        hat_abs_S = ds_in['f(abs_S)_r'].data[...]
        ds_dx_hat = ds_in[f'f(d{scalar_name}_dx)_r'].data[...]

        HAT_abs_S_ds_dx = ds_in[f'f(abs_S_d{scalar_name}_dx)_r'].data[...]

        Rj = dyn.R_j(dx, dx_hat, hat_abs_S, ds_dx_hat, HAT_abs_S_ds_dx, beta=1)
        HAT_abs_S_ds_dx = None

        HR_field = np.zeros_like(Hj[0, ...])
        RR_field = np.zeros_like(Rj[0, ...])

        for it in range(0, 3):
            HR_field += Hj[it, ...] * Rj[it, ...]
            RR_field += Rj[it, ...] * Rj[it, ...]


        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                dims=["time", "x_p", "y_p", "z"], name=f'RR_{scalar}_field')

        return HR_field, RR_field




