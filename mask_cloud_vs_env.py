import numpy as np
import numpy.ma as ma
import xarray as xr
import dynamic_functions as dyn


def get_th_v_prime(th_v):
    nt = len(th_v[:,0,0,0])
    z_num = len(th_v[0, 0, 0, :])
    horiz_num_temp = len(th_v[0,:,0,0])
    horiz_num = horiz_num_temp * horiz_num_temp

    th_v_flat = th_v.reshape(nt, horiz_num, z_num)
    th_v_prime_temp = th_v.reshape(nt, horiz_num, z_num)
    th_v_prof = np.zeros((nt, z_num))

    for t in range(nt):
        for k in range(z_num):
            th_v_prof[t, k] = np.sum(th_v_flat[t, :, k]) / horiz_num
            th_v_prime_temp[t, :, k] = th_v_flat[t, :, k] - th_v_prof[t, k]

    th_v_flat = None
    th_v_prof = None
    print('th_v_prof = ', th_v_prof)
    print('shape of th_v_prime before reshape is:', np.shape(th_v_prime_temp))
    th_v_prime = th_v_prime_temp.reshape(nt, horiz_num_temp, horiz_num_temp, z_num)
    th_v_prime_temp = None

    return th_v_prime




def cloud_vs_env_masks(data_in, cloud_liquid_threshold, res_counter=None, grid='p'):

    if res_counter != None:
        data_in_new = data_in + f'ga0{res_counter}.nc'
    else:
        data_in_new = data_in
    ds_in = xr.open_dataset(data_in_new)



    if f'f(q_cloud_liquid_mass_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(q_cloud_liquid_mass_on_{grid})_r']
    elif f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r']
    elif 'q_cloud_liquid_mass' in ds_in:
        q_in = ds_in['q_cloud_liquid_mass']
    else:
        ds_in2 = xr.open_dataset(f'/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0{res_counter}.nc')
        q_in = ds_in2['f(q_cloud_liquid_mass_on_w)_r']

    q_cloud = q_in.data

    masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold) #masking lower values
    masked_q_env = ma.masked_greater_equal(q_cloud, cloud_liquid_threshold) #masking larger values

    cloud_only_mask = ma.getmaskarray(masked_q_cloud)
    env_only_mask = ma.getmaskarray(masked_q_env)

    masked_q_cloud = None
    masked_q_env = None

    return cloud_only_mask, env_only_mask



def cloudy_and_or(data_in, other_var, var_thres, less_greater, and_or, \
                  cloud_liquid_threshold, res_counter=None, return_all = False, grid='p'):

    if res_counter != None:

        ds_in2 = xr.open_dataset(f'/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0{res_counter}.nc')
        q_in = ds_in2[f'f(q_cloud_liquid_mass_on_{grid})_r']

    else:
        ds_in = xr.open_dataset(data_in)

        if 'f(q_cloud_liquid_mass_on_w)_r' in ds_in:
            q_in = ds_in[f'f(q_cloud_liquid_mass_on_{grid})_r']
        elif f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r' in ds_in:
            q_in = ds_in[f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r']
        else:
            q_in = ds_in['q_cloud_liquid_mass']

    q_cloud = q_in.data
    masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold)

    if f'f({other_var[0]}_on_{grid})_r' in ds_in:
        var_in = ds_in[f'f({other_var[0]}_on_{grid})_r']
    elif f'f(f({other_var[0]}_on_{grid})_r_on_{grid})_r' in ds_in:
        var_in = ds_in[f'f({other_var[0]}_on_{grid})_r']
    else:
        var_in = ds_in[f'{other_var[0]}']

    if other_var[0] == f'f(f(th_v_on_{grid})_r_on_{grid})_r':
        var_temp = var_in.data
        var_data = get_th_v_prime(var_temp)
    else:
        var_data = var_in.data
    print('loaded other var:', other_var[0])

    if less_greater[0]=='less':
        masked_var = ma.masked_less(var_data, var_thres[0])
    elif less_greater[0]=='greater':
        masked_var = ma.masked_greater_equal(var_data, var_thres[0])
    else:
        print("must pick 'less' (values lower than threshold are masked/EXCLUDED) or 'greater' (values higher than threshold are masked/EXCLUDED).")
    print('masked other var:', other_var[0])

    cloud_mask = ma.getmaskarray(masked_q_cloud)
    var_mask = ma.getmaskarray(masked_var)

    if and_or[0] == 'and':
        out_mask = ma.mask_or(cloud_mask, var_mask) #masks work opposite
    elif and_or[0] == 'or':
        temp_mask = (cloud_mask == False) & (var_mask == False)
    else:
        print('must pick and or or')

    if len(other_var) > 1:
        if f'f({other_var[1]}_on_{grid})_r' in ds_in:
            extra_var_in = ds_in[f'f({other_var[1]}_on_{grid})_r']
        elif f'f(f({other_var[1]}_on_{grid})_r_on_{grid})_r' in ds_in:
            extra_var_in = ds_in[f'f({other_var[1]}_on_{grid})_r']
        else:
            extra_var_in = ds_in[f'{other_var[1]}']

        if other_var[1] == f'f(f(th_v_on_{grid})_r_on_{grid})_r':
            extra_var_temp = extra_var_in.data
            extra_var_data = get_th_v_prime(extra_var_temp)
            print('got theta prime')
        else:
            extra_var_data = extra_var_in.data
        print('loaded other var:', other_var[1])

        if less_greater[1] == 'less':
            extra_masked_var = ma.masked_less(extra_var_data, var_thres[1])
        elif less_greater[1] == 'greater':
            extra_masked_var = ma.masked_greater_equal(extra_var_data, var_thres[1])
        extra_var_mask = ma.getmaskarray(extra_masked_var)
        print('masked other var:', other_var[1])

        if and_or[1] == 'and':
            new_out_mask = ma.mask_or(out_mask, extra_var_mask)  # masks work opposite
        elif and_or[1] == 'or':
            new_temp_mask = (out_mask == False) & (extra_var_mask == False)

    if return_all == False:
        if len(other_var) == 1:
            return out_mask
        else:
            return out_mask, new_out_mask
    else:
        if len(other_var) == 1:
            return out_mask, cloud_mask, var_mask
        else:
            return out_mask, new_out_mask, cloud_mask, var_mask, extra_var_mask
