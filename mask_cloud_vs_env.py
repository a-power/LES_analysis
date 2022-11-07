import numpy as np
import numpy.ma as ma
import xarray as xr
import dynamic_functions as dyn

def cloud_vs_env_masks(data_in, cloud_liquid_threshold=10**(-5), res_counter=None):

    data_in_new = data_in + f'ga0{res_counter}.nc'

    ds_in = xr.open_dataset(data_in_new)

    if 'f(q_cloud_liquid_mass_on_w)_r' in ds_in:
        q_in = ds_in['f(q_cloud_liquid_mass_on_w)_r']

    elif 'q_cloud_liquid_mass' in ds_in:
        q_in = ds_in['q_cloud_liquid_mass']

    else:
        ds_in2 = xr.open_dataset(f'/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0{res_counter}.nc')
        q_in = ds_in2['f(q_cloud_liquid_mass_on_w)_r']

    q_cloud_temp = q_in.data

    if len(np.shape(q_cloud_temp)) == 4:
        q_cloud = np.mean(q_cloud_temp, axis=0)
    elif len(np.shape(q_cloud_temp)) == 3:
        q_cloud = q_cloud_temp
        q_cloud_temp = None
    else:
        print('q_l shape is not len 3 or len 4: missing/extra dimentions. q_l has shape:', np.shape(q_cloud_temp))

    masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold) #masking lower values
    masked_q_env = ma.masked_greater_equal(q_cloud, cloud_liquid_threshold) #masking larger values

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

    q_cloud_temp = q_in.data

    if len(np.shape(q_cloud_temp)) == 4:
        q_cloud = np.mean(q_cloud_temp, axis=0)
    elif len(np.shape(q_cloud_temp)) == 3:
        q_cloud = q_cloud_temp
        q_cloud_temp = None
    else:
        print('q_l shape is not len 3 or len 4: missing/extra dimentions. q_l has shape:', np.shape(q_cloud_temp))


    if f'f({other_var}_on_w)_r' in ds_in:
        var_in = ds_in[f'f({other_var}_on_w)_r']
    else:
        var_in = ds_in[f'{other_var}']

    var_data_temp = var_in.data

    if len(np.shape(var_data_temp)) == 4:
        var_data = np.mean(var_data_temp, axis=0)
    elif len(np.shape(var_data_temp)) == 3:
        var_data = var_data_temp
        var_data_temp = None
    else:
        print(f'var shape is not len 3 or len 4: missing/extra dimentions. {other_var} has shape:', np.shape(var_data_temp))

    masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold)

    if less_greater_threas=='less':
        masked_var = ma.masked_less(var_data, var_thres)
    elif less_greater_threas=='greater':
        masked_var = ma.masked_greater_equal(var_data, var_thres)
    else:
        print("must pick 'less' (values lower than threshold are masked/EXCLUDED) or 'greater' (values higher than threshold are masked/EXCLUDED).")


    cloud_mask = ma.getmaskarray(masked_q_cloud)
    var_mask = ma.getmaskarray(masked_var)

    if and_or == 'and':
        out_mask = ma.mask_and(cloud_mask, var_mask)
    elif and_or == 'or':
        out_mask = ma.mask_and(cloud_mask, var_mask)

    if return_all == False:
        return out_mask
    else:
        return out_mask, cloud_mask, var_mask



def get_masked_fields(dataset_in, delta, res_count = None, return_fields=True, cloud_thres = 10**(-5), other_var_choice = False, \
                      other_var_thres=False, less_greater='greater', my_and_or = 'and', return_all_masks=False):

    data_s = xr.open_dataset(dataset_in + f'Cs_{delta}.nc')
    data_th = xr.open_dataset(dataset_in + f'C_th_{delta}.nc')
    data_qtot = xr.open_dataset(dataset_in + f'C_qt_{delta}.nc')

    Cs_sq = data_s['Cs_sq_field'].data[...]
    Cth_sq = data_th['C_th_sq_field'].data[...]
    Cqt_sq = data_qtot['C_q_total_sq_field'].data[...]

    print('shape of Cs_sq is = ', np.shape(Cs_sq))

    Cs = dyn.get_Cs(Cs_sq)
    Cth = dyn.get_Cs(Cth_sq)
    Cqt = dyn.get_Cs(Cqt_sq)

    time_data = data_s['time']
    nt = time_data.data
    x_data = data_s['x_p']
    x_s = x_data.data
    y_data = data_s['y_p']
    y_s = y_data.data
    z_data = data_s['z']
    z_s = z_data.data



    if other_var_choice == False:

        cloud_only_mask, env_only_mask = cloud_vs_env_masks(dataset_in, cloud_liquid_threshold=cloud_thres, \
                                                            res_counter=res_count)
        Cs_cloud = ma.masked_array(Cs, mask=cloud_only_mask)
        Cs_env = ma.masked_array(Cs, mask=env_only_mask)
        Cth_cloud = ma.masked_array(Cth, mask=cloud_only_mask)
        Cth_env = ma.masked_array(Cth, mask=env_only_mask)
        Cqt_cloud = ma.masked_array(Cqt, mask=cloud_only_mask)
        Cqt_env = ma.masked_array(Cqt, mask=env_only_mask)


        Cs_cloud_prof = np.mean(Cs_cloud, axis=2)
        Cs_env_prof = np.mean(Cs_env, axis=2)
        Cth_cloud_prof = np.mean(Cth_cloud, axis=2)
        Cth_env_prof = np.mean(Cth_env, axis=2)
        Cqt_cloud_prof = np.mean(Cqt_cloud, axis=2)
        Cqt_env_prof = np.mean(Cqt_env, axis=2)


        Cs_cloud_prof_out = xr.DataArray(Cs_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')
        Cs_env_prof_out = xr.DataArray(Cs_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')
        Cth_cloud_prof_out = xr.DataArray(Cth_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')
        Cth_env_prof_out = xr.DataArray(Cth_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')
        Cqt_cloud_prof_out = xr.DataArray(Cqt_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')
        Cqt_env_prof_out = xr.DataArray(Cqt_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='')


        if return_fields == False:

            Cs_cloud = None
            Cs_env = None
            Cth_cloud = None
            Cth_env = None
            Cqt_cloud = None
            Cqt_env = None

            return Cs_cloud_prof_out, Cs_env_prof_out, Cth_cloud_prof_out, Cth_env_prof_out, Cqt_cloud_prof_out, Cqt_env_prof_out

        else:

            Cs_cloud_out =  xr.DataArray(Cs_cloud[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='Cs_cloud_field')
            Cs_env_out =  xr.DataArray(Cs_env[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='Cs_env_field')
            Cth_cloud_out = xr.DataArray(Cth_cloud[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='C_th_cloud_field')
            Cth_env_out =  xr.DataArray(Cth_env[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='C_th_env_field')
            Cqt_cloud_out =  xr.DataArray(Cqt_cloud[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='C_qt_cloud_field')
            Cqt_env_out = xr.DataArray(Cqt_env[np.newaxis, ...],
                                             coords={'time': [nt], 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='C_qt_env_field')

            Cs_cloud = None
            Cs_env = None
            Cth_cloud = None
            Cth_env = None
            Cqt_cloud = None
            Cqt_env = None

            return Cs_cloud_out, Cs_env_out, Cth_cloud_out, Cth_env_out, Cqt_cloud_out, Cqt_env_out, \
                   Cs_cloud_prof_out, Cs_env_prof_out, Cth_cloud_prof_out, Cth_env_prof_out, Cqt_cloud_prof_out, Cqt_env_prof_out


    #
    # else:
    #
    #     mask = cloudy_and_or(data_in = dataset_in, other_var = other_var_choice, var_thres=other_var_thres, \
    #                          less_greater_threas='greater', and_or='and', cloud_liquid_threshold=10 ** (-5), \
    #                          res_counter=res_count, return_all=return_all_masks)



