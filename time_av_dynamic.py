import numpy as np
import xarray as xr

import subfilter.filters as filt
import subfilter.subfilter as sf
import subfilter.utils as ut
import subfilter.utils.deformation as defm
from subfilter.utils.dask_utils import re_chunk

# import filters as filt
# import subfilter as sf

import dynamic as dyn
import dask
import subfilter


def run_dyn(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid, ref_file = None):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """



    file_in = f'{indir}{res_in}/diagnostic_files/BOMEX_m{res_in}_all_{time_in}.nc'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time_series_600_600']
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
            filter_id = 'filter_ga{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                      filter_name, npoints=N,
                                      sigma=filt_set, width=width,
                                      delta_x=dx_in, cutoff=cutoff)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id, filter_name,
                                      wavenumber=filt_set,
                                      width=width, npoints=N,
                                      delta_x=dx_in)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}'.format(i)
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

            var_list = ["u",
                        "v",
                        "w",
                        "th"
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
                        ["w", "th"]]

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

            dth_dx = dyn.d_th_d_x_i(dataset, ref_dataset, opt, ingrid)
            dth_dx.name = 'dth_dx'
            dth_dx = re_chunk(dth_dx, xch=nch, ych=nch, zch='all')

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

            S_ij_abs_S = S_ij * abs_S
            S_ij_abs_S.name = 'S_ij_abs_S'
            S_ij_abs_S = re_chunk(S_ij_abs_S)

            S_ij_abs_S_hat_filt = sf.filter_field(S_ij_abs_S, filtered_data,
                                                  opt, new_filter)

            abs_S_dth_dx = abs_S * dth_dx
            abs_S_dth_dx.name = 'abs_S_dth_dx'
            abs_S_dth_dx = re_chunk(abs_S_dth_dx)

            abs_S_dth_dx_filt = sf.filter_field(abs_S_dth_dx, filtered_data,
                                                  opt, new_filter)

        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()
    return



def Cs(indir, dx, dx_hat, ingrid, t_in=0, save_all=0, Cs_av_method = 'all'):

    """ function takes in:  """

    file_in = f'{indir}'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time']
    times = time_data.data
    nt = len(times)
    ds_in.close()

    ds_in = xr.open_dataset(file_in)
    uu = ds_in[f's(u,u)_on_{ingrid}'].data[t_in, ...]
    uv = ds_in[f's(u,v)_on_{ingrid}'].data[t_in, ...]
    uw = ds_in[f's(u,w)_on_{ingrid}'].data[t_in, ...]
    vv = ds_in[f's(v,v)_on_{ingrid}'].data[t_in, ...]
    vw = ds_in[f's(v,w)_on_{ingrid}'].data[t_in, ...]
    ww = ds_in[f's(w,w)_on_{ingrid}'].data[t_in, ...]

    Lij = dyn.L_ij_sym_xarray(uu, uv, uw, vv, vw, ww)

    hat_Sij_abs_S = ds_in['S_ij_abs_S_r'].data[:, t_in, :, :, :]
    hat_Sij = ds_in['S_ij_r'].data[:, t_in, :, :, :]

    Mij = dyn.M_ij(dx, dx_hat, hat_Sij, hat_Sij_abs_S)

    Cs_prof = dyn.Cs_av_levels(Lij, Mij, av_method=Cs_av_method)

    if save_all==3:
        Cs_sq_field = dyn.C_s_sq(Lij, Mij)
        Cs_field = dyn.get_Cs(Cs_sq_field)
        return Cs_prof, Cs_field, Lij, Mij, times

    if save_all==2:
        Cs_sq_field = dyn.C_s_sq(Lij, Mij)
        Cs_field = dyn.get_Cs(Cs_sq_field)
        return Cs_prof, Cs_field, times

    if save_all==1:
        return Cs_prof, times
    else:
        return Cs_prof





