import numpy as np
import numpy.ma as ma
import xarray as xr

def cloud_vs_env_masks(data_in, cloud_liquid_threshold=10**(-5), res_counter=None):

    ds_in = xr.open_dataset(data_in)

    if res_counter != None:

        ds_in2 = xr.open_dataset(f'/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0{res_counter}.nc')
        q_in = ds_in2['f(q_cloud_liquid_mass_on_w)_r']

    else:
        if 'f(q_cloud_liquid_mass_on_w)_r' in ds_in:
            q_in = ds_in['f(q_cloud_liquid_mass_on_w)_r']
        else:
            q_in = ds_in['q_cloud_liquid_mass']

    q_cloud = q_in.data

    masked_q_cloud = ma.greater_equal(q_cloud, cloud_liquid_threshold)
    masked_q_env = ma.less_equal(q_cloud, cloud_liquid_threshold)

    cloud_only_mask = ma.getmaskarray(masked_q_cloud)
    env_only_mask = ma.getmaskarray(masked_q_env)

    masked_q_cloud = None
    masked_q_env = None

    return cloud_only_mask, env_only_mask



def cloudy_and_or(data_in, other_var, var_thres, less_greater_threas='greater', and_or = 'and', \
                  cloud_liquid_threshold=10**(-5), res_counter=None, return_all = False):

    ds_in = xr.open_dataset(data_in)

    if res_counter != None:

        ds_in2 = xr.open_dataset(f'/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0{res_counter}.nc')
        q_in = ds_in2['f(q_cloud_liquid_mass_on_w)_r']

    else:
        if 'f(q_cloud_liquid_mass_on_w)_r' in ds_in:
            q_in = ds_in['f(q_cloud_liquid_mass_on_w)_r']
        else:
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

    if return_all=False:
        return out_mask
    else:
        return out_mask, cloud_mask, var_mask



def get_masked_fields(dataset_in, C_data_fields, delta, res_count = None, other_var_choice = False, \
                      other_var_thres=False, less_greater='greater', my_and_or = 'and', return_all_masks=False, \
                      return_fields=True, cloud_thres = 10**(-5)):

    data_s = xr.open_dataset(C_data_fields + f's_{delta}.nc')
    data_th = xr.open_dataset(C_data_fields + f'_th_{delta}.nc')
    data_qtot = xr.open_dataset(C_data_fields + f'q_tot_{delta}.nc')

    Cs = data_s['Cs_prof'].data[0, ...]
    Cth = data_th['C_th_prof'].data[0, ...]
    Cq = data_qtot['C_q_total_prof'].data[0, ...]



    if other_var_choice == False:

        cloud_only_mask, env_only_mask = cloud_vs_env_masks(dataset_in, cloud_liquid_threshold=cloud_thres, \
                                                            res_counter=res_count)
        Cs_cloud = ma.masked_array(Cs, mask=cloud_only_mask)
        Cs_env = ma.masked_array(Cs, mask=env_only_mask)
        Cth_cloud = ma.masked_array(Cth, masqk=cloud_only_mask)
        Cth_env = ma.masked_array(Cth, mask=env_only_mask)
        Cqt_cloud = ma.masked_array(Cq, mask=cloud_only_mask)
        Cqt_env = ma.masked_array(Cq, mask=env_only_mask)

        if return_fields == True:
            return Cs_cloud, Cs_env, Cth_cloud, Cth_env, Cqt_cloud, Cqt_env

        else:
            Cs_cloud_prof = np.mean(Cs_cloud, axis=2)
            Cs_env_prof = np.mean(Cs_env, axis=2)
            Cth_cloud_prof = np.mean(Cth_cloud, axis=2)
            Cth_env_prof = np.mean(Cth_env, axis=2)
            Cqt_cloud_prof = np.mean(Cqt_cloud, axis=2)
            Cqt_env_prof = np.mean(Cqt_env, axis=2)

            return Cs_cloud_prof, Cs_env_prof, Cth_cloud_prof, Cth_env_prof, Cqt_cloud_prof, Cqt_env_prof

    #
    # else:
    #
    #     mask = cloudy_and_or(data_in = dataset_in, other_var = other_var_choice, var_thres=other_var_thres, \
    #                          less_greater_threas='greater', and_or='and', cloud_liquid_threshold=10 ** (-5), \
    #                          res_counter=res_count, return_all=return_all_masks)



