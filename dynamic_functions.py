import numpy as np
import xarray as xr

import subfilter.utils.difference_ops as do
from subfilter.utils.string_utils import get_string_index
from subfilter.utils.dask_utils import re_chunk
from subfilter.io.datain import get_data

import subfilter

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')


def k_cut_find(delta):
    return np.pi / (delta)


def sigma_find(delta):
    return delta / 2


def l_mix_MONC(Cs, Delta, z_in, k=0.4):
    l_mix = np.sqrt(1 / ((1 / (Cs * Cs * Delta * Delta)) + (1 / (k * k * z_in * z_in))))

    return l_mix


def my_L_ij_sym(u, v, w, uu, uv, uw, vv, vw, ww):
    L_ij = np.array([(u * u - uu), (u * v - uv), (u * w - uw),
                     (v * v - vv), (v * w - vw),
                     (w * w - ww)])

    return L_ij


def L_ij_sym_xarray(uu, uv, uw, vv, vw, ww):
    L_ij = np.array([-(uu), -(uv), -(uw),
                     -(vv), -(vw),
                     -(ww)])

    return L_ij


def H_j(u_s, v_s, w_s):
    H_j = np.array([-u_s, -v_s, -w_s])
    return H_j


def S_ij(u, v, w, dx):
    surf_BC_0s = np.zeros_like(u)

    dudx = np.diff(np.insert(u, 0, [u[-1, :, :]], axis=0), axis=0) / dx
    dudy = np.diff(np.insert(u, 0, [u[:, -1, :]], axis=1), axis=1) / dx
    dudz = np.diff(np.insert(u, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2) / dx

    dvdx = np.diff(np.insert(v, 0, [v[-1, :, :]], axis=0), axis=0) / dx
    dvdy = np.diff(np.insert(v, 0, [v[:, -1, :]], axis=1), axis=1) / dx
    dvdz = np.diff(np.insert(v, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2) / dx

    dwdx = np.diff(np.insert(w, 0, [w[-1, :, :]], axis=0), axis=0) / dx
    dwdy = np.diff(np.insert(w, 0, [w[:, -1, :]], axis=1), axis=1) / dx
    dwdz = np.diff(np.insert(w, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2) / dx

    S_ij = 0.5 * np.array([[(2 * dudx), (dudy + dvdx), (dudz + dwdx)],
                           [(dvdx + dudy), (2 * dvdy), (dvdz + dwdy)],
                           [(dwdx + dudz), (dwdy + dvdz), (2 * dwdz)]])

    return S_ij


def abs_S(s):
    abs_S = np.sqrt(2 * (s[0, 0] * s[0, 0] + s[0, 1] * s[0, 1] + s[0, 2] * s[0, 2] +
                         s[1, 0] * s[1, 0] + s[1, 1] * s[1, 1] + s[1, 2] * s[1, 2] +
                         s[2, 0] * s[2, 0] + s[2, 1] * s[2, 1] + s[2, 2] * s[2, 2]
                         ))
    return abs_S


def abs_S_hat(S_filt_in):
    temp_S_sum = np.zeros_like(S_filt_in[0, :, :, :])
    S_sum = np.zeros_like(S_filt_in)

    for i in range(6):
        if i in [1, 2, 4]:
            S_sum += 2 * S_filt_in[i, :, :, :] * S_filt_in[i, :, :, :]
        else:
            S_sum += S_filt_in[i, :, :, :] * S_filt_in[i, :, :, :]
    abs_S_hat_out = np.sqrt(2 * S_sum)

    return abs_S_hat_out


def abs_S_hat_S_ij_hat(S_filt):
    abs_S_filt = abs_S_hat(S_filt)
    abs_S_hat_S_ij_hat = S_filt * abs_S_filt

    return abs_S_hat_S_ij_hat


def M_ij(dx_filt1, dx_filt2, S_filt, abs_S_filt, HAT_abs_S_Sij, beta=1):
    alpha = dx_filt2 / dx_filt1
    power = alpha / 2

    M_ij = dx_filt2 * dx_filt2 * (beta ** power) * abs_S_filt * S_filt - dx_filt1 * dx_filt1 * HAT_abs_S_Sij

    return M_ij


def M_ij_stab_fns(dx_filt1, dx_filt2, S_filt, abs_S_filt, HAT_abs_S_Sij_fRi, fRi_hat, beta=1):
    alpha = dx_filt2 / dx_filt1
    power = alpha / 2

    M_ij = dx_filt * dx_filt * (beta ** power) * abs_S_filt * S_filt * fRi_hat - dx * dx * HAT_abs_S_Sij_fRi

    return M_ij


def ds_dxi(scalar, source_dataset, ref_dataset_in, options, in_grid, filting_filted=False):
    # scalar can be either 'th' or "q_total"

    if scalar == 'q':
        if filting_filted == False:
            scalar = 'q_total'
        else:
            scalar = 'q_total_f'

    s = get_data(source_dataset, ref_dataset_in, str(scalar), options)

    [iix, iiy, iiz] = get_string_index(s.dims, ['x', 'y', 'z'])
    sh = np.shape(s)
    max_ch = subfilter.global_config['chunk_size']
    nch = int(sh[iix] / (2 ** int(np.log(sh[iix] * sh[iiy] * sh[iiz] / max_ch) / np.log(2) / 2)))
    print(f'{scalar} nch={nch}')

    s = re_chunk(s, xch=nch, ych=nch, zch='all')

    z = source_dataset["z"]
    zn = source_dataset["zn"]

    sx = do.d_by_dx_field(s, z, zn, grid=in_grid)
    sy = do.d_by_dy_field(s, z, zn, grid=in_grid)
    sz = do.d_by_dz_field(s, z, zn, grid=in_grid)

    s = None  # Save some memory

    s_xi = xr.concat([sx, sy, sz], dim='j', coords='minimal',
                     compat='override')

    return s_xi


def R_j(dx_filt1, dx_filt2, abs_S_hat, ds_dxj_hat, HAT_abs_S_ds_dxj, beta=1):
    alpha = dx_filt2 / dx_filt1
    power = alpha / 2

    R_j = dx_filt2 * dx_filt2 * beta ** power * abs_S_hat * ds_dxj_hat - dx_filt1 * dx_filt1 * HAT_abs_S_ds_dxj

    return R_j


def R_j_stab_fns(dx_filt1, dx_filt2, abs_S_hat, ds_dxj_hat, HAT_abs_S_ds_dxj_fRi, fRi_hat, beta=1):
    alpha = dx_filt2 / dx_filt1
    power = alpha / 2

    R_j = dx_filt2 * dx_filt2 * beta ** power * abs_S_hat * ds_dxj_hat * fRi_hat - dx_filt1 * dx_filt1 * HAT_abs_S_ds_dxj_fRi

    return R_j



def calc_Ri(abs_S_sq, derived_dataset, filtered_dataset, ref_dataset_in, options, in_grid):
    # scalar can be either 'th' or "q_total"

    # derived_dataset = dataset_path + '.nc'
    # filtered_dataset = dataset_path + f'_filter_ga0{}.nc'

    buoy = get_data(derived_dataset, ref_dataset_in, 'bouyancy_on_p', options)

    [iix, iiy, iiz] = get_string_index(buoy.dims, ['x', 'y', 'z'])
    sh = np.shape(buoy)
    max_ch = subfilter.global_config['chunk_size']
    nch = int(sh[iix] / (2 ** int(np.log(sh[iix] * sh[iiy] * sh[iiz] / max_ch) / np.log(2) / 2)))


    buoy = re_chunk(buoy, xch=nch, ych=nch, zch='all')

    z = filtered_dataset["z"]
    zn = filtered_dataset["zn"]

    buoy_x = do.d_by_dx_field(buoy, z, zn, grid=in_grid)
    buoy_y = do.d_by_dy_field(buoy, z, zn, grid=in_grid)
    buoy_z = do.d_by_dz_field(buoy, z, zn, grid=in_grid)

    buoy = None  # Save some memory

    dB_dx = xr.concat([buoy_x, buoy_y, buoy_z], dim='j', coords='minimal',
                     compat='override')

    Ri = dB_dx / abs_S_sq

    dB_dx = None

    return Ri

def stab_fn_mom(Ri):

    Ri_c = 0.25
    Pr_n = 0.7
    a = 1/Pr_n
    b = 40
    c = 16
    g = 1.2
    h = 0.0
    r = 4

    fm_Ri = Ri.copy()
    fm_Ri[Ri < Ri_c] = ( (1 - (Ri/Ri_c))**r ) * (1 - h*Ri)
    fm_Ri[Ri < 0] = np.sqrt(1 - c * Ri)
    fm_Ri[Ri >= Ri_c] = 0

    return fm_Ri


def stab_fn_scal(Ri):
    Ri_c = 0.25
    Pr_n = 0.7
    a = 1 / Pr_n
    b = 40
    c = 16
    g = 1.2
    h = 0.0
    r = 4

    fh_Ri = Ri.copy()
    fh_Ri[Ri < Ri_c] = ((1 - (Ri / Ri_c)) ** r) * (1 - g * Ri)
    fh_Ri[Ri < 0] = a * np.sqrt(1 - b * Ri)
    fh_Ri[Ri >= Ri_c] = 0

    return fh_Ri



def index_sym(i,j, i_j_in):

    if j < i:
        i_new = j
        j_new = i

    else:
        i_new = i
        j_new = j

    index_str = '{}_{}'.format(i_new,j_new)

    index = np.where(i_j_in == index_str)[0][0]

    return index


def C_s_sq(L_ij, M_ij, return_all=2, time_av=False):

    print('shape of Lij is', np.shape(L_ij))

    C_s_sq = np.zeros_like(L_ij[0, ...]) #ij
    C_s_num = np.zeros_like(L_ij[0, ...])
    C_s_den = np.zeros_like(L_ij[0, ...])

    for it in range(0,6):
        if it in [0,3,5]:

            C_s_num += L_ij[it, ...] * M_ij[it, ...]
            C_s_den += M_ij[it, ...]**2

        else:
            C_s_num += 2*(L_ij[it, ...] * M_ij[it, ...])
            C_s_den += 2*M_ij[it, ...]**2

    if time_av==True:
        print("number of times = ", (C_s_num.shape)[0])

        C_s_num_av = np.mean(C_s_num, 0)
        C_s_num = 0

        C_s_den_av = np.mean(C_s_den, 0)
        C_s_den = 0

        C_s_sq = 0.5 * C_s_num_av / C_s_den_av

    else:
        C_s_sq = 0.5 * C_s_num / C_s_den #LM and MM

    print('shape of C_S_sq in function dyn.C_s_sq is', np.shape(C_s_sq))

    if return_all==1:
        return C_s_sq
    else:
        return C_s_sq, C_s_num, C_s_den


def C_scalar_sq(Hj, Rj, return_all=2, time_av=False):

    C_th_num = np.zeros_like(Hj[0, ...])
    C_th_den = np.zeros_like(Hj[0, ...])

    for it in range(0, 3):
        C_th_num += Hj[it, ...] * Rj[it, ...]
        C_th_den += Rj[it, ...] * Rj[it, ...]

    if time_av==True:
        print("number of times = ", (C_th_num.shape)[0])

        C_th_num_av = np.mean(C_th_num, 0)
        C_th_num = 0

        C_th_den_av = np.mean(C_th_den, 0)
        C_th_den = 0

        C_th_sq = 0.5 * C_th_num_av / C_th_den_av

    else:
        C_th_sq = 0.5 * C_th_num / C_th_den

    if return_all==1:
        return C_th_sq
    else:
        return C_th_sq, C_th_num, C_th_den


def get_Cs(Cs_sq):
    """ calculates C_s from C_s^2 by setting neg values to zero
    and sq rooting"""

    Cs_sq_copy = Cs_sq.copy()
    Cs_sq_copy[Cs_sq < 0] = 0
    Cs = np.sqrt(Cs_sq_copy)

    return Cs


def Cs_profiles(L_ij, M_ij, return_all=1):
    """ Calculates the horizontal average Cs value at each level
    using the Lij and Mij fields as input.

    return_all: 1 is for profiles, 2 is for fields

    """

    C_s_num = np.zeros_like(L_ij[0, ...])
    C_s_den = np.zeros_like(M_ij[0, ...])


    for it in range(0,6):
        if it in [0,3,5]:

            C_s_num += L_ij[it, ...] * M_ij[it, ...]
            C_s_den += M_ij[it, ...] * M_ij[it, ...]

        else:
            C_s_num += 2*(L_ij[it, ...] * M_ij[it, ...])
            C_s_den += 2*(M_ij[it, ...] * M_ij[it, ...])

    z_num = (C_s_num.shape)[-1]
    horiz_num_temp = (C_s_num.shape)[-2]
    horiz_num = horiz_num_temp * horiz_num_temp

    if len(L_ij.shape) == 5:
        num_times = (C_s_num.shape)[0]
        total_num = num_times*horiz_num
        LM_flat = C_s_num.reshape(total_num, z_num)
        MM_flat = C_s_den.reshape(total_num, z_num)
    else:
        LM_flat = C_s_num.reshape(horiz_num,z_num)
        MM_flat = C_s_den.reshape(horiz_num,z_num)
        total_num = horiz_num


    # if return_all == 2:
    #     if len(L_ij.shape) == 5:
    #         LM_field_av = np.mean(C_s_num, 0)
    #         MM_field_av = np.mean(C_s_den, 0)
    #     else:
    #         LM_field_av = C_s_num.copy()
    #         MM_field_av = C_s_den.copy()
    #
    # C_s_num = None
    # C_s_den = None

    LM_av = np.zeros((num_times, z_num))
    MM_av = np.zeros((num_times, z_num))

    for k in range(z_num):

        LM_av[0, k] = np.sum(LM_flat[:,k])/total_num
        MM_av[0, k] = np.sum(MM_flat[:,k])/total_num


    Cs_av_sq = (0.5*(LM_av / MM_av))

    Cs_av = get_Cs(Cs_av_sq)

    if return_all == 1:
        return Cs_av_sq, Cs_av, LM_av, MM_av

    if return_all == 2:
        return Cs_av_sq, Cs_av, LM_av, MM_av, C_s_num, C_s_den
    else:
        return Cs_av_sq, Cs_av


def C_scalar_profiles(H_ij, R_ij, return_all=2):
    """ Calculates the horizontal average Cs value at each level
    using the Lij and Mij fields as input.

    return_all: 1 is for profiles, 2 is for fields

    """

    C_th_num = np.zeros_like(H_ij[0, ...])
    C_th_den = np.zeros_like(R_ij[0, ...])

    for it in range(0, 3):

        C_th_num += H_ij[it, ...] * R_ij[it, ...]
        C_th_den += R_ij[it, ...] * R_ij[it, ...]

    z_num = (C_th_num.shape)[-1]
    horiz_num_temp = (C_th_num.shape)[-2]
    horiz_num = horiz_num_temp * horiz_num_temp

    if len(H_ij.shape) == 5:
        num_times = (C_th_num.shape)[0]
        total_num = num_times * horiz_num
        HR_flat = C_th_num.reshape(total_num, z_num)
        RR_flat = C_th_den.reshape(total_num, z_num)
    else:
        HR_flat = C_th_num.reshape(horiz_num, z_num)
        RR_flat = C_th_den.reshape(horiz_num, z_num)
        total_num = horiz_num


    # if return_all == 2:
        # if len(H_ij.shape) == 5:
        #     HR_field_av = np.mean(C_th_num, 0)
        #     RR_field_av = np.mean(C_th_den, 0)
        # else:
        #     HR_field_av = C_th_num.copy()
        #     RR_field_av = C_th_den.copy()


    # C_th_num = None
    # C_th_den = None

    HR_av = np.zeros((num_times, z_num))
    RR_av = np.zeros((num_times, z_num))

    for k in range(z_num):
        HR_av[0, k] = np.sum(HR_flat[:, k]) / total_num
        RR_av[0, k] = np.sum(RR_flat[:, k]) / total_num

    C_th_av_sq = (0.5 * (HR_av / RR_av))

    C_th_av = get_Cs(C_th_av_sq)

    if return_all == 1:
        return C_th_av_sq, C_th_av, HR_av, RR_av

    if return_all == 2:
        return C_th_av_sq, C_th_av, HR_av, RR_av, C_th_num, C_th_den
    else:
        return C_th_av_sq, C_th_av



def minimal_beta_calc(Cs_2D_sq_in, Cs_4D_sq_in):

    beta = Cs_4D_sq_in/Cs_2D_sq_in
    # beta[beta < 0.125] = 0.125

    return beta


def beta_calc(C_2D_sq_in, C_4D_sq_in):

    Cs_2D_sq_copy1 = C_2D_sq_in.copy()
    Cs_2D_sq_copy2 = C_2D_sq_in.copy()
    Cs_4D_sq_copy = C_4D_sq_in.copy()

    Cs_2D_sq_copy2[Cs_2D_sq_copy1==0.00000000000] = 1
    Cs_4D_sq_copy[Cs_2D_sq_copy1==0.00000000000] = 100000 #go to inf

    beta = Cs_4D_sq_copy/Cs_2D_sq_copy2
    beta[beta < 0.125] = 0.125

    return beta


def Cs_sq_beta_dep(C_s2_sq, beta):

    """ calculates Cs_sq using C_s = C_s2_sq/beta """

    return C_s2_sq/beta



def Pr(Cs_sq, C_scalar_sq):
     Pr = Cs_sq/C_scalar_sq
     return Pr


def w_therm_field(w, t_in, return_all=False):
    w_95th = np.zeros_like(w[t_in, 0, 0, :])
    w_therm = np.zeros_like(w[t_in, ...])

    for k in range(len(w[t_in, 0, 0, :])):
        w_95th[k] = np.percentile(w[t_in, :, :, k], 95)
        w_therm[:, :, k] = (w[t_in, :, :, k] >= w_95th[k])

    if return_all == True:
        return w_therm, w_95th
    else:
        return w_therm


def cloud_field_ind(cloud_field, t_in, cloud_liquid_threshold = 10**(-5)):

    clouds = np.zeros_like(cloud_field[t_in, ...])

    for k in range(len(cloud_field[t_in, 0, 0, :])):
        clouds[:, :, k] = (cloud_field[t_in, :, :, k] >= cloud_liquid_threshold)

    return clouds


def Cs_therm_cloud_field(Cs, t_in, i_ind, j_ind, k_ind):
    Cs_therm_cloud = np.zeros_like(Cs[t_in, ...])

    for i in range(len(i_ind)):
        Cs_therm_cloud[i_ind[i], j_ind[i], k_ind[i]] = Cs[t_in, i_ind[i], j_ind[i], k_ind[i]]

    return Cs_therm_cloud


#### check time indices for Cs (Lij and Mij)