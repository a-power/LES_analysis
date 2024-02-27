import numpy as np
import xarray as xr

import subfilter.filters as filt
import subfilter.subfilter as sf
import monc_utils.monc_utils as ut
import monc_utils.data_utils.deformation as defm
import monc_utils.data_utils.cloud_monc as cldm

#from monc_utils.utils.default_variables import (get_default_variable_list, get_default_variable_pair_list)
from monc_utils.data_utils.string_utils import get_string_index
from monc_utils.io.datain import get_data_on_grid
from monc_utils.io.dataout import save_field, setup_child_file
from monc_utils.data_utils.dask_utils import re_chunk

import dynamic_functions as dyn
import dask
import subfilter

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

#code now has theta_l and _v, q_l, q_v were all added to the filtering code in this iteration


def run_dyn(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid,
            start_point=0, time_name = 'time_series_600_600', vapour=True):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """
    print('start point in the run_dyn script is', start_point, ' of type ', type(start_point))

    print(time_name)

    if res_in != None:
        file_in = f'{indir}{res_in}/diagnostic_files/BOMEX_m{res_in}_all_{time_in}.nc'
        ref_file = f'{indir}{res_in}/diagnostic_files/BOMEX_m{res_in}_all_{time_in}.nc'
    elif time_name == 'time_series_300_300':
        file_in = f'{indir}/cbl_{time_in}.nc'
        ref_file = None
    else:
        file_in = f'{indir}/diagnostics_3d_ts_{time_in}.nc'
        ref_file = f'{indir}/diagnostics_ts_{time_in}.nc'


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
    [itime, iix, iiy, iiz] = get_string_index(ds_in.dims, ['time', 'x', 'y', 'z'])
    timevar = list(ds_in.dims)[itime]
    xvar = list(ds_in.dims)[iix]
    yvar = list(ds_in.dims)[iiy]
    zvar = list(ds_in.dims)[iiz]
    max_ch = 4* (2**dx_in) #4*subfilter.global_config['chunk_size']

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
        ref_dataset = xr.open_dataset(ref_file)
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
            if vapour == False:

                var_list = [
                        "u",
                        "v",
                        "w",
                        "th"]
            else:
                var_list = [
                            "u",
                            "v",
                            "w",
                            "th",
                            "q_total",
                            "q_vapour",
                            "q_cloud_liquid_mass",
                            "buoyancy"
                            ]
                # "th_v",
                # "th_L",

            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 opt, new_filter,
                                                 var_list=var_list,
                                                 grid=ingrid)
            if vapour == False:
                var_list = [["u", "u"],
                        ["u", "v"],
                        ["u", "w"],
                        ["v", "v"],
                        ["v", "w"],
                        ["w", "w"],
                        ["u", "th"],
                        ["v", "th"],
                        ["w", "th"],
                        ]
            else:
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
                            ["u", "q_vapour"],
                            ["v", "q_vapour"],
                            ["w", "q_vapour"]
                            ]
                            # ["u", "th_v"],
                            # ["v", "th_v"],
                            # ["w", "th_v"],
                            # ["w", "th_L"],
                            # ["th_v", "q_total"],
                            # ["w", "q_cloud_liquid_mass"],
                            # ["q_total", "q_total"],
                            # ["th_L", "q_cloud_liquid_mass"]


            quad_field_list = sf.filter_variable_pair_list(dataset,
                                                           ref_dataset,
                                                           derived_data, filtered_data,
                                                           opt, new_filter,
                                                           var_list=var_list,
                                                           grid=ingrid)

        dth_dx = dyn.ds_dxi('th', dataset, ref_dataset, max_ch, opt, ingrid)
        print('ran   dth_dx = dyn.ds_dxi which has a shape of', np.shape(dth_dx))
        dth_dx.name = 'dth_dx'
        dth_dx = re_chunk(dth_dx)
        print('ran  rechunk of dth_dx which has a shape of', np.shape(dth_dx))

        dth_dx_filt = sf.filter_field(dth_dx, filtered_data,
                                      opt, new_filter)
        print('ran sf.filter_field(dth_dx) which has a shape of', np.shape(dth_dx_filt))

        if vapour == True:
            dq_dx = dyn.ds_dxi('q_total', dataset, ref_dataset, max_ch, opt, ingrid)
            dq_dx.name = 'dq_dx'
            dq_dx = re_chunk(dq_dx)

            dqv_dx = dyn.ds_dxi('q_vapour', dataset, ref_dataset, max_ch, opt, ingrid)
            dqv_dx.name = 'dqv_dx'
            dqv_dx = re_chunk(dqv_dx)

            dq_dx_filt = sf.filter_field(dq_dx, filtered_data,
                                         opt, new_filter)

            dqv_dx_filt = sf.filter_field(dqv_dx, filtered_data,
                                          opt, new_filter)




        deform = defm.deformation(dataset,
                                  ref_dataset,
                                  derived_data,
                                  opt)

        print('ran  deform = defm.deformation')

        S_ij_temp, abs_S_temp = defm.shear(deform, no_trace=False)  #
        print('ran S_ij_temp, abs_S_temp = defm.shear')

        S_ij = 0.5 * S_ij_temp
        S_ij_temp = None
        S_ij.name = 'S_ij'
        S_ij = re_chunk(S_ij)

        print('ran rechunk of S_ij')

        abs_S = np.sqrt(abs_S_temp)  ##### do not need to mult by 4 here, see Smag notes pg 8 on Boox
        abs_S.name = "abs_S"
        abs_S = re_chunk(abs_S)

        print('ran rechunk of abs_S')







        # Ri = dyn.calc_Ri(abs_S_temp, derived_data, filtered_data, ref_dataset, opt, ingrid)
        # Ri.name = 'Ri'
        # Ri = re_chunk(Ri)
        #
        # Ri_filt = sf.filter_field(Ri, filtered_data, opt, new_filter)
        #
        # fm_Ri = dyn.stab_fn_mom(Ri)
        # fm_Ri.name = 'fm_Ri'
        # fm_Ri = re_chunk(fm_Ri)
        #
        # fh_Ri = dyn.stab_fn_scal(Ri)
        # fh_Ri.name = 'fh_Ri'
        # fh_Ri = re_chunk(fh_Ri)


        S_ij_filt = sf.filter_field(S_ij, filtered_data,
                                    opt, new_filter)
        print('ran sf.filter_field(S_ij')

        abs_S_filt = sf.filter_field(abs_S, filtered_data,
                                     opt, new_filter)
        print('ran sf.filter_field(abs_S')


        S_ij_abs_S = S_ij * abs_S
        S_ij_abs_S.name = 'S_ij_abs_S'
        S_ij_abs_S = re_chunk(S_ij_abs_S)
        print('ran S_ij_abs_S = re_chunk')

        S_ij_abs_S_hat_filt = sf.filter_field(S_ij_abs_S, filtered_data,
                                              opt, new_filter)
        print('ran S_ij_abs_S_hat_filt = sf.filter_field')


        # S_ij_abs_S_fm_Ri = S_ij * abs_S * fm_Ri
        # S_ij_abs_S_fm_Ri.name = 'S_ij_abs_S_fm_Ri'
        # S_ij_abs_S_fm_Ri = re_chunk(S_ij_abs_S_fm_Ri)
        #
        # S_ij_abs_S_fm_filt = sf.filter_field(S_ij_abs_S_fm_Ri, filtered_data, opt, new_filter)





        abs_S_dth_dx = dth_dx * abs_S
        abs_S_dth_dx.name = 'abs_S_dth_dx'
        abs_S_dth_dx = re_chunk(abs_S_dth_dx)
        print('ran abs_S_dth_dx = re_chunk(abs_S_dth_dx)')

        abs_S_dth_dx_filt = sf.filter_field(abs_S_dth_dx, filtered_data,
                                              opt, new_filter)
        print('ran abs_S_dth_dx_filt = sf.filter_field(abs_S_dth_dx')

        # abs_S_dth_dx_fh_Ri = dth_dx * abs_S * fh_Ri
        # abs_S_dth_dx_fh_Ri.name = 'abs_S_dth_dx_fh_Ri'
        # abs_S_dth_dx_fh_Ri = re_chunk(abs_S_dth_dx_fh_Ri)
        #
        # abs_S_dth_dx_fh_Ri_filt = sf.filter_field(abs_S_dth_dx_fh_Ri, filtered_data,
        #                                       opt, new_filter)




        # abs_S_dq_dx_fh_Ri = dq_dx * abs_S * fh_Ri
        # abs_S_dq_dx_fh_Ri.name = 'abs_S_dq_dx_fh_Ri'
        # abs_S_dq_dx_fh_Ri = re_chunk(abs_S_dq_dx_fh_Ri)
        #
        # abs_S_dq_dx_fh_Ri_filt = sf.filter_field(abs_S_dq_dx_fh_Ri, filtered_data,
        #                                     opt, new_filter)

        if vapour == True:
            abs_S_dq_dx = dq_dx * abs_S
            abs_S_dq_dx.name = 'abs_S_dq_dx'
            abs_S_dq_dx = re_chunk(abs_S_dq_dx)

            abs_S_dq_dx_filt = sf.filter_field(abs_S_dq_dx, filtered_data,
                                                opt, new_filter)

            abs_S_dqv_dx = dqv_dx * abs_S
            abs_S_dqv_dx.name = 'abs_S_dqv_dx'
            abs_S_dqv_dx = re_chunk(abs_S_dqv_dx)

            abs_S_dqv_dx_filt = sf.filter_field(abs_S_dqv_dx, filtered_data,
                                                opt, new_filter)

        # abs_S_dqv_dx_fh_Ri = dqv_dx * abs_S * fh_Ri
        # abs_S_dqv_dx_fh_Ri.name = 'abs_S_dqv_dx_fh_Ri'
        # abs_S_dqv_dx_fh_Ri = re_chunk(abs_S_dqv_dx_fh_Ri)
        #
        # abs_S_dqv_dx_fh_Ri_filt = sf.filter_field(abs_S_dqv_dx_fh_Ri, filtered_data,
        #                                     opt, new_filter)

        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()
    return


def run_dyn_on_filtered(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid, filtered_data, start_point=0,
            ref_file = None, time_name = 'time_series_600_600', case='ARM'):

    """ function takes in:
     dx: the grid spacing and number of grid points in the format:  """
    if case=='BOMEX':
        file_in = f'{indir}{res_in}_all_{time_in}_gaussian_filter_{filtered_data}.nc'
    elif case=='ARM':
        file_in = f'{indir}/diagnostics_3d_ts_{time_in}_gaussian_filter_{filtered_data}.nc'
    elif case == 'dry':
        file_in = f'{indir}/cbl_{time_in}_gaussian_filter_{filtered_data}.nc'

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
    [itime, iix, iiy, iiz] = get_string_index(ds_in.dims, ['time', 'x', 'y', 'z'])
    timevar = list(ds_in.dims)[itime]
    xvar = list(ds_in.dims)[iix]
    yvar = list(ds_in.dims)[iiy]
    zvar = list(ds_in.dims)[iiz]
    max_ch = 4* (2**dx_in) #4*subfilter.global_config['chunk_size']

    # This is a rough way to estimate chunck size
    nch = np.min([int(ds_in.dims[xvar] / (2 ** int(np.log(ds_in.dims[xvar]
                                                            * ds_in.dims[yvar]
                                                            * ds_in.dims[zvar]
                                                            / max_ch) / np.log(2) / 2))),
                                                            ds_in.dims[xvar]])

    ds_in.close()

    dataset = xr.open_dataset(file_in, chunks={timevar: 1, xvar: nch, yvar: nch,
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
            if case == 'dry':
                var_list = [
                        "u",
                        "v",
                        "w",
                        "th"]
            else:
                var_list = [
                            "u",
                            "v",
                            "w",
                            "th",
                            "q_total",
                            "q_vapour",
                            "q_cloud_liquid_mass",
                            "buoyancy"
                            ]
                # "th_v",
                # "th_L",

            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 opt, new_filter,
                                                 var_list=var_list,
                                                 grid=ingrid)

            if case == 'dry':
                var_list = [["u", "u"],
                        ["u", "v"],
                        ["u", "w"],
                        ["v", "v"],
                        ["v", "w"],
                        ["w", "w"],
                        ["u", "th"],
                        ["v", "th"],
                        ["w", "th"],
                        ]
            else:
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
                            ["u", "q_vapour"],
                            ["v", "q_vapour"],
                            ["w", "q_vapour"]
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
                                opt, ingrid,
                                uvw_names=[f'f(u_on_{ingrid})_r', f'f(v_on_{ingrid})_r', f'f(w_on_{ingrid})_r'])

        dth_dx = dyn.ds_dxi(f'f(th_on_{ingrid})_r', dataset, ref_dataset, max_ch, opt, ingrid)
        dth_dx.name = 'dth_dx'
        dth_dx = re_chunk(dth_dx)

        if case != 'dry':
            dq_dx = dyn.ds_dxi(f'f(q_total_on_{ingrid})_r', dataset, ref_dataset, max_ch, opt, ingrid)
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

        if case != 'dry':
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

        if case != 'dry':
            abs_S_dq_dx = dq_dx * abs_S
            abs_S_dq_dx.name = 'abs_S_dq_dx'
            abs_S_dq_dx = re_chunk(abs_S_dq_dx)

            abs_S_dq_dx_filt = sf.filter_field(abs_S_dq_dx, filtered_data,
                                                opt, new_filter)

        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()

    return






def run_dyn_on_filtered_for_beta_contour(res_in, time_in, filt_in, filt_scale, indir, odir, opt, ingrid, filtered_data, start_point=0,
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
    max_ch = 4* (2**dx_in) #4*subfilter.global_config['chunk_size']

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
                        "w",
                        "th",
                        "q_total_f",
                        "th_v",
                        "q_cloud_liquid_mass"
                        ]

            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 opt, new_filter,
                                                 var_list=var_list,
                                                 grid=ingrid)

            var_list = [["u", "u"],
                        ["v", "v"],
                        ["w", "w"],
                        ["w", "th_v"]
                        ]


            quad_field_list = sf.filter_variable_pair_list(dataset,
                                                           ref_dataset,
                                                           derived_data, filtered_data,
                                                           opt, new_filter,
                                                           var_list=var_list,
                                                           grid=ingrid)



        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()
    return





def Cs(indir, dx_bar, dx_hat, ingrid, save_all=2, reaxes=False):

    """ function takes in:

    save_all: 1 is for profiles, 2 is for fields, 3 is for all fields PLUS Lij and Mij"""

    file_in = f'{indir}'
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in['time']
    times = time_data.data
    nt = len(times)
    print('length of the time array in Cs function is', nt)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    zn_data = ds_in['zn']
    zn_s = zn_data.data

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

    Mij = dyn.M_ij(dx_bar, dx_hat, hat_Sij, hat_abs_S, hat_Sij_abs_S)

    hat_Sij_abs_S = None
    hat_Sij = None
    hat_abs_S=None

    zn_save = np.zeros((nt, len(zn_s)))
    zn_save[0,...] = zn_s
    zn_save = xr.DataArray(zn_save, coords={'time': times, 'zn': zn_s},
                              dims=['time', "zn"], name='zn_save')

    z_save = np.zeros((nt, len(z_s)))
    z_save[0, ...] = z_s
    z_save = xr.DataArray(z_save, coords={'time': times, 'z': z_s},
                           dims=['time', "z"], name='z_save')

    if save_all==1:
        Cs_sq_prof, Cs_prof, LM_prof, MM_prof = dyn.Cs_profiles(Lij, Mij, return_all=1)
        Lij = None
        Mij = None

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[...], coords={'time' : times, 'zn': zn_s},
                                  dims=['time', "zn"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[...], coords={'time' : times, 'zn': zn_s},
                               dims=['time', "zn"], name='Cs_prof')

        LM_prof = xr.DataArray(LM_prof[...], coords={'time' : times, 'zn': zn_s},
                               dims=['time', "zn"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[...], coords={'time' : times, 'zn': zn_s},
                               dims=['time', "zn"], name='MM_prof')

        return z_save, zn_save, Cs_sq_prof, Cs_prof, LM_prof, MM_prof

    if save_all==2:

        #Cs_sq_field = dyn.C_s_sq(Lij, Mij)

        Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field = dyn.Cs_profiles(Lij, Mij, return_all=2)

        Lij = None
        Mij = None

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[...], coords={'time' : times,'zn': zn_s},
                                  dims=['time', "zn"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[...], coords={'time' : times,'zn': zn_s},
                               dims=['time', "zn"], name='Cs_prof')

        LM_prof = xr.DataArray(LM_prof[...], coords={'time' : times,'zn': zn_s},
                               dims=['time', "zn"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[...], coords={'time' : times,'zn': zn_s},
                               dims=['time', "zn"], name='MM_prof')


        LM_field = xr.DataArray(LM_field[...], coords={'time' : times, 'x_p' : x_s, 'y_p' : y_s, 'zn': zn_s},
                                  dims = ["time", "x_p", "y_p", "zn"], name = 'LM_field')

        MM_field = xr.DataArray(MM_field[...], coords={'time' : times, 'x_p' : x_s, 'y_p' : y_s, 'zn': zn_s},
                                  dims = ["time", "x_p", "y_p", "zn"], name = 'MM_field')


        #Cs_sq_field = xr.DataArray(Cs_sq_field[...], coords={'time' : times, 'x_p' : x_s, 'y_p' : y_s, 'zn': zn_s},
        #                          dims = ["time", "x_p", "y_p", "z"], name = 'Cs_sq_field')

        return z_save, zn_save, Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field#, Cs_sq_field

    if save_all==3:

        Cs_sq_field = dyn.C_s_sq(Lij, Mij)

        print('in dynamic script the shape of Cs_sq_field is', np.shape(Cs_sq_field))

        Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field = dyn.Cs_profiles(Lij, Mij, return_all=2)

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[...], coords={'time': times, 'zn': zn_s},
                                  dims=['time', "zn"], name='Cs_sq_prof')

        Cs_prof = xr.DataArray(Cs_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name='Cs_prof')

        Cs_sq_field = xr.DataArray(Cs_sq_field[...],
                                   coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                   dims=["time", "x_p", "y_p", "zn"], name='Cs_sq_field')

        LM_prof = xr.DataArray(LM_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name='LM_prof')

        MM_prof = xr.DataArray(MM_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name='MM_prof')

        LM_field = xr.DataArray(LM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name='LM_field')

        MM_field = xr.DataArray(MM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name='MM_field')
        if len(Mij.shape) == 5:

            print("number of times = ", (Mij.shape)[1])

            Mij_av = np.mean(Mij, 1)
            Mij = None
            Lij_av = np.mean(Lij, 1)
            Lij = None

            Lij = xr.DataArray(Lij_av[...], coords={'time': times, 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'zn': zn_s},
                                    dims=["time", "i_j", "x_p", "y_p", "zn"], name='Lij')

            Mij = xr.DataArray(Mij_av[...], coords={'time': times, 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'zn': zn_s},
                                    dims=["time", "i_j", "x_p", "y_p", "zn"], name='Mij')
        else:
            Lij = xr.DataArray(Lij[...],
                               coords={'time': times, 'i_j': ij_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                               dims=["time", "i_j", "x_p", "y_p", "zn"], name='Lij')

            Mij = xr.DataArray(Mij[...],
                               coords={'time': times, 'i_j': ij_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                               dims=["time", "i_j", "x_p", "y_p", "zn"], name='Mij')


        return zn_save, Cs_sq_prof, Cs_prof, LM_prof, MM_prof, LM_field, MM_field, Cs_sq_field, Lij, Mij

    else:
        Cs_sq_prof, Cs_prof = dyn.Cs_profiles(Lij, Mij, return_all=0)

        Cs_prof = xr.DataArray(Cs_sq_prof[...], coords={'time': times, 'zn': zn_s},
                                  dims=['time', "zn"], name='Cs_sq_prof')

        Cs_sq_prof = xr.DataArray(Cs_sq_prof[...], coords={'time' : times, 'zn': zn_s},
                                  dims=['time', "zn"], name='Cs_sq_prof')

        Lij = None
        Mij = None

        return z_save, zn_save, Cs_sq_prof, Cs_prof






def C_scalar(scalar, indir, dx_bar, dx_hat, ingrid, save_all = 2, axisfix=False):
    """ function takes in:

    save_all: 1 is for profiles, 2 is for fields, 3 is for all fields PLUS Lij and Mij"""

    if scalar=='q_total':
        scalar_name='q'
    elif scalar == 'q_cloud_liquid_mass':
        scalar_name = 'q_l'
    elif scalar == 'q_vapour':
        scalar_name = 'q_v'
    elif scalar == 'q_total_f':
        scalar_name = 'q'
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
    print('length of the time array in Cs function is', nt)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    zn_data = ds_in['zn']
    zn_s = zn_data.data

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


    Rj = dyn.R_j(dx_bar, dx_hat, hat_abs_S, ds_dx_hat, HAT_abs_S_ds_dx, beta=1)
    HAT_abs_S_ds_dx = None

    zn_save = np.zeros((nt, len(zn_s)))
    zn_save[0,...] = zn_s
    zn_save = xr.DataArray(zn_save, coords={'time': times, 'zn': zn_s},
                              dims=['time', "zn"], name='zn_save')

    z_save = np.zeros((nt, len(z_s)))
    z_save[0, ...] = z_s
    z_save = xr.DataArray(z_save, coords={'time': times, 'z': z_s},
                           dims=['time', "z"], name='z_save')

    if save_all == 1:

        C_scalar_sq_prof, C_scalar_prof = dyn.C_scalar_profiles(Hj, Rj, return_all=0)

        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[...], coords={'time': times, 'zn': zn_s},
                                        dims=['time', "zn"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[...], coords={'time': times, 'zn': zn_s},
                                     dims=['time', "zn"], name=f'C_{scalar}_prof')

        return z_save, zn_save, C_scalar_sq_prof, C_scalar_prof



    if save_all == 2:

        #C_scalar_sq_field = dyn.C_scalar_sq(Hj, Rj)

        C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field = dyn.C_scalar_profiles(Hj, Rj, return_all=2)


        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[...], coords={'time': times, 'zn': zn_s},
                                        dims=['time', "zn"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[...], coords={'time': times, 'zn': zn_s},
                                     dims=['time', "zn"], name=f'C_{scalar}_prof')

        HR_prof = xr.DataArray(HR_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name=f'HR_{scalar}_prof')

        RR_prof = xr.DataArray(RR_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name=f'RR_{scalar}_prof')

        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                     dims=["time", "x_p", "y_p", "zn"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                     dims=["time", "x_p", "y_p", "zn"], name=f'RR_{scalar}_field')

        # C_scalar_sq_field = xr.DataArray(C_scalar_sq_field[...],
        #                                  coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
        #                                  dims=["time", "x_p", "y_p", "z"], name=f'C_{scalar}_sq_field')


        return z_save, zn_save, C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field#, C_scalar_sq_field



    if save_all == 3:

        C_scalar_sq_field = dyn.C_scalar_sq(Hj, Rj)

        C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field = dyn.C_scalar_profiles(Hj, Rj, return_all=2)

        C_scalar_sq_prof = xr.DataArray(C_scalar_sq_prof[...], coords={'time': times, 'zn': zn_s},
                                    dims=['time', "zn"], name=f'C_{scalar}_sq_prof')

        C_scalar_prof = xr.DataArray(C_scalar_prof[...], coords={'time': times, 'zn': zn_s},
                                 dims=['time', "zn"], name=f'C_{scalar}_prof')

        HR_prof = xr.DataArray(HR_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name=f'HR_{scalar}_prof')

        RR_prof = xr.DataArray(RR_prof[...], coords={'time': times, 'zn': zn_s},
                               dims=['time', "zn"], name=f'RR_{scalar}_prof')

        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                     dims=["time", "x_p", "y_p", "zn"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                     dims=["time", "x_p", "y_p", "zn"], name=f'RR_{scalar}_field')

        C_scalar_sq_field = xr.DataArray(C_scalar_sq_field[...],
                                     coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                     dims=["time", "x_p", "y_p", "zn"], name=f'C_{scalar}_sq_field')

        if len(Hj.shape) == 5:

            print("number of times = ", (Hj.shape)[1])

            Hj_av = np.mean(Hj, 1)
            Hj = None
            Rj_av = np.mean(Rj, 1)
            Rj = None

            Rj = xr.DataArray(Rj_av[...],
                              coords={'time': times, 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                              dims=["time", "i_j", "x_p", "y_p", "zn"], name='Hj')

            Hj = xr.DataArray(Hj_av[...],
                              coords={'time': times, 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                              dims=["time", "i_j", "x_p", "y_p", "zn"], name='Rj')
        else:

            Rj = xr.DataArray(Rj[...],
                              coords={'time': times, 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                              dims=["time", "i_j", "x_p", "y_p", "zn"], name='Hj')

            Hj = xr.DataArray(Hj[...],
                              coords={'time': times, 'i_j': j_s, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                              dims=["time", "i_j", "x_p", "y_p", "zn"], name='Rj')

        return z_save, zn_save, C_scalar_sq_prof, C_scalar_prof, HR_prof, RR_prof, HR_field, RR_field, C_scalar_sq_field, Hj, Rj


def LijMij_fields(scalar, indir, dx_bar, dx_hat, ingrid):

    if scalar == 'q_total':
        scalar_name = 'q'
    if scalar == 'q_total_f':
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
    nt = len(times)
    print('time array is', times)

    x_data = ds_in['x_p']
    x_s = x_data.data

    y_data = ds_in['y_p']
    y_s = y_data.data

    z_data = ds_in['z']
    z_s = z_data.data

    zn_data = ds_in['zn']
    zn_s = zn_data.data

    ds_in.close()

    zn_save = np.zeros((nt, len(zn_s)))
    zn_save[0,...] = zn_s
    zn_save = xr.DataArray(zn_save, coords={'time': times, 'zn': zn_s},
                              dims=['time', "zn"], name='zn_save')

    z_save = np.zeros((nt, len(z_s)))
    z_save[0, ...] = z_s
    z_save = xr.DataArray(z_save, coords={'time': times, 'z': z_s},
                           dims=['time', "z"], name='z_save')

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

        Mij = dyn.M_ij(dx_bar, dx_hat, hat_Sij, hat_abs_S, hat_Sij_abs_S)

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

        LM_field = xr.DataArray(LM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name='LM_field')

        MM_field = xr.DataArray(MM_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name='MM_field')

        return z_save, zn_save, LM_field, MM_field


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

        Rj = dyn.R_j(dx_bar, dx_hat, hat_abs_S, ds_dx_hat, HAT_abs_S_ds_dx, beta=1)
        HAT_abs_S_ds_dx = None

        HR_field = np.zeros_like(Hj[0, ...])
        RR_field = np.zeros_like(Rj[0, ...])

        for it in range(0, 3):
            HR_field += Hj[it, ...] * Rj[it, ...]
            RR_field += Rj[it, ...] * Rj[it, ...]


        HR_field = xr.DataArray(HR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name=f'HR_{scalar}_field')

        RR_field = xr.DataArray(RR_field, coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
                                dims=["time", "x_p", "y_p", "zn"], name=f'RR_{scalar}_field')

        return z_save, zn_save, HR_field, RR_field




