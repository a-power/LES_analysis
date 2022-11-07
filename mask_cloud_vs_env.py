import numpy as np
import numpy.ma as ma
import xarray as xr

def cloud_vs_env_masks(data_in, cloud_liquid_threshold=10**(-5)):

    ds_in = xr.open_dataset(data_in)
    q_in = ds_in['q_cloud_liquid_mass']
    q_cloud = q_in.data

    masked_q_cloud = ma.greater_equal(q_cloud, cloud_liquid_threshold)
    masked_q_env = ma.less_equal(q_cloud, cloud_liquid_threshold)

    cloud_only_mask = ma.getmaskarray(masked_q_cloud)
    env_only_mask = ma.getmaskarray(masked_q_env)

    return cloud_only_mask, env_only_mask


def cloudy_and_or(data_in, other_var, var_thres, less_greater_threas='greater', and_or = 'and', cloud_liquid_threshold=10**(-5)):

    ds_in = xr.open_dataset(data_in)
    q_in = ds_in['q_cloud_liquid_mass']
    q_cloud = q_in.data

    var_in = ds_in[f'{other_var}']
    var_data = var_in.data

    masked_q_cloud = ma.greater_equal(q_cloud, cloud_liquid_threshold)

    if less_greater_threas=='less':
        masked_var = ma.less_equal(var_data, var_thres)
    elif less_greater_threas=='greater':
        masked_var = ma.greater_equal(var_data, var_thres)
    else:
        print("must pick 'less' or 'greater'.")


    cloud_mask = ma.getmaskarray(masked_q_cloud)
    var_mask = ma.getmaskarray(masked_var)

    if and_or == 'and':
        out_mask = ma.mask_and(cloud_mask, var_mask)
    elif and_or == 'or':
        out_mask = ma.mask_and(cloud_mask, var_mask)

    return out_mask


def get_masked_fields(unfilt_dataset_in, C_data_fields, delta, other_var_choice = False, \
                      other_var_thres=False, less_greater='greater', my_and_or = 'and', cloud_thres=10**(-5)):

    data_s = xr.open_dataset(C_data_fields + f's_{delta}.nc')
    data_th = xr.open_dataset(C_data_fields + f'_th_{delta}.nc')
    data_qtot = xr.open_dataset(C_data_fields + f'q_tot_{delta}.nc')

    Cs = data_s['Cs_prof'].data[0, ...]
    Cth = data_th['C_th_prof'].data[0, ...]
    Cq = data_qtot['C_q_total_prof'].data[0, ...]

    if other_var_choice == False:

        cloud_only_mask, env_only_mask = cloud_vs_env_masks(unfilt_dataset_in, cloud_liquid_threshold=cloud_thres)




        mask = cloudy_and_or(data_in = dataset_in, other_var = other_var_choice, var_thres, less_greater_threas='greater', and_or='and',
                  cloud_liquid_threshold=10 ** (-5))



