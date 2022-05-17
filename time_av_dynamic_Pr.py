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



def C_th(indir, dx, dx_hat, ingrid, t_in=0, save_all=1):

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
    u_th = ds_in[f's(u,th)_on_{ingrid}'].data[t_in, ...]
    v_th = ds_in[f's(v,th)_on_{ingrid}'].data[t_in, ...]
    w_th = ds_in[f's(w,th)_on_{ingrid}'].data[t_in, ...]

    Hj = dyn.H_j(u_th, v_th, w_th)

    u_th = None # Save storage
    v_th = None # Save storage
    w_th = None # Save storage

    hat_abs_S = ds_in['f(abs_S)_r'].data[t_in, ...]
    dth_dx_hat = ds_in['f(dth_dx)_r'].data[:,t_in, ...]
    HAT_abs_S_dth_dx = ds_in['f(abs_S_dth_dx)_r'].data[t_in, ...]

    Rj = dyn.R_j(dx, dx_hat, hat_abs_S, dth_dx_hat, HAT_abs_S_dth_dx, beta=1)


    C_th_sq_prof, C_th_prof, HR_prof, RR_prof, RH, RR = dyn.C_th_profiles(Hj, Rj, return_all=2)
    C_th_sq_field = dyn.C_th_sq(Hj, Rj)

    C_th_sq_prof = xr.DataArray(C_th_sq_prof[np.newaxis, ...], coords={'time': [times[t_in]], 'z': z_s},
                              dims=['time', "z"], name='C_th_sq_prof')

    C_th_prof = xr.DataArray(C_th_prof[np.newaxis, ...], coords={'time': [times[t_in]], 'z': z_s},
                           dims=['time', "z"], name='C_th_prof')

    HR_prof = xr.DataArray(HR_prof[np.newaxis, ...], coords={'time': [times[t_in]], 'z': z_s},
                           dims=['time', "z"], name='HR_prof')

    RR_prof = xr.DataArray(RR_prof[np.newaxis, ...], coords={'time': [times[t_in]], 'z': z_s},
                           dims=['time', "z"], name='RR_prof')

    C_th_sq_field = xr.DataArray(C_th_sq_field[np.newaxis, ...],
                               coords={'time': [times[t_in]], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                               dims=["time", "x_p", "y_p", "z"], name='C_th_sq_field')

    HR_field = xr.DataArray(HR_field[np.newaxis, ...], coords={'time': [times[t_in]], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                            dims=["time", "x_p", "y_p", "z"], name='HR_field')

    RR_field = xr.DataArray(RR_field[np.newaxis, ...], coords={'time': [times[t_in]], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                            dims=["time", "x_p", "y_p", "z"], name='RR_field')

    Hj = xr.DataArray(Hj[np.newaxis, ...], coords={'time': [times[t_in]], 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                            dims=["time", "i_j", "x_p", "y_p", "z"], name='Hj')

    Rj = xr.DataArray(Rj[np.newaxis, ...], coords={'time': [times[t_in]], 'i_j': ij_s, 'x_p' : x_s, 'y_p' : y_s, 'z': z_s},
                            dims=["time", "i_j", "x_p", "y_p", "z"], name='Rj')


    return C_th_sq_prof, C_th_prof, HR_prof, RR_prof, C_th_sq_field, HR_field, RR_field, Hj, Rj






