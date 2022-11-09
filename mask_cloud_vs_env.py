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

    q_cloud = q_in.data

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

    q_cloud = q_in.data

    if f'f({other_var}_on_w)_r' in ds_in:
        var_in = ds_in[f'f({other_var}_on_w)_r']
    else:
        var_in = ds_in[f'{other_var}']

    var_data = var_in.data

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
        out_mask = ma.mask_or(cloud_mask, var_mask) #masks work opposite

    elif and_or == 'or':
        temp_mask = (cloud_mask == False) & (var_mask == False)

    else:
        print('must pick and or or')

    if return_all == False:
        return out_mask
    else:
        return out_mask, cloud_mask, var_mask



def get_masked_fields(dataset_in, delta, res_count = None, return_fields=True, cloud_thres = 10**(-5), other_var_choice = False, \
                      other_var_thres=False, less_greater='greater', my_and_or = 'and', return_all_masks=False):

    data_s = xr.open_dataset(dataset_in + f'LijMij_{delta}.nc')
    LijMij = data_s['LM_field'].data[...]
    MijMij = data_s['MM_field'].data[...]
    data_s.close()

    data_th = xr.open_dataset(dataset_in + f'HjRj_th_{delta}.nc')
    HjRj_th = data_th['HR_th_field'].data[...]
    RjRj_th = data_th['RR_th_field'].data[...]
    data_th.close()

    data_qtot = xr.open_dataset(dataset_in + f'HjRj_qt_{delta}.nc')
    HjRj_qt = data_qtot['HR_q_total_field'].data[...]
    RjRj_qt = data_qtot['RR_q_total_field'].data[...]


    print('shape of LijMij is = ', np.shape(LijMij), 'and the shape of HjRj_th is = ', np.shape(HjRj_th))

    time_data = data_qtot['time']
    times = time_data.data
    nt = len(times)
    x_data = data_qtot['x_p']
    x_s = x_data.data
    y_data = data_qtot['y_p']
    y_s = y_data.data
    z_data = data_qtot['z']
    z_s = z_data.data

    data_qtot.close()


    if other_var_choice == False:

        cloud_only_mask, env_only_mask = cloud_vs_env_masks(dataset_in, cloud_liquid_threshold=cloud_thres, \
                                                            res_counter=res_count)
        LijMij_cloud = ma.masked_array(LijMij, mask=cloud_only_mask)
        LijMij_env = ma.masked_array(LijMij, mask=env_only_mask)
        MijMij_cloud = ma.masked_array(MijMij, mask=cloud_only_mask)
        MijMij_env = ma.masked_array(MijMij, mask=env_only_mask)

        LijMij = None
        MijMij = None

        HjRj_th_cloud = ma.masked_array(HjRj_th, mask=cloud_only_mask)
        HjRj_th_env = ma.masked_array(HjRj_th, mask=env_only_mask)
        RjRj_th_cloud = ma.masked_array(RjRj_th, mask=cloud_only_mask)
        RjRj_th_env = ma.masked_array(RjRj_th, mask=env_only_mask)

        HjRj_th = None
        RjRj_th = None

        HjRj_qt_cloud = ma.masked_array(HjRj_qt, mask=cloud_only_mask)
        HjRj_qt_env = ma.masked_array(HjRj_qt, mask=env_only_mask)
        RjRj_qt_cloud = ma.masked_array(RjRj_qt, mask=cloud_only_mask)
        RjRj_qt_env = ma.masked_array(RjRj_qt, mask=env_only_mask)

        HjRj_qt = None



        z_num = (RjRj_qt.shape)[-1]
        horiz_num_temp = (RjRj_qt.shape)[-2]
        horiz_num = horiz_num_temp * horiz_num_temp

        if len(RjRj_qt.shape) == 4:
            num_times = (RjRj_qt.shape)[0]
            total_num = num_times * horiz_num
            RjRj_qt = None

            LM_flat_cloud = LijMij_cloud.reshape(total_num, z_num)
            LM_flat_env = LijMij_env.reshape(total_num, z_num)
            MM_flat_cloud = MijMij_cloud.reshape(total_num, z_num)
            MM_flat_env = MijMij_env.reshape(total_num, z_num)

            HR_th_flat_cloud = HjRj_th_cloud.reshape(total_num, z_num)
            HR_th_flat_env = HjRj_th_env.reshape(total_num, z_num)
            RR_th_flat_cloud = RjRj_th_cloud.reshape(total_num, z_num)
            RR_th_flat_env = RjRj_th_env.reshape(total_num, z_num)

            HR_qt_flat_cloud = HjRj_qt_cloud.reshape(total_num, z_num)
            HR_qt_flat_env = HjRj_qt_env.reshape(total_num, z_num)
            RR_qt_flat_cloud = RjRj_qt_cloud.reshape(total_num, z_num)
            RR_qt_flat_env = RjRj_qt_env.reshape(total_num, z_num)

        else:
            RjRj_qt = None
            LM_flat_cloud = LijMij_cloud.reshape(horiz_num, z_num)
            LM_flat_env = LijMij_env.reshape(horiz_num, z_num)
            MM_flat_cloud = MijMij_cloud.reshape(horiz_num, z_num)
            MM_flat_env = MijMij_env.reshape(horiz_num, z_num)

            HR_th_flat_cloud = HjRj_th_cloud.reshape(horiz_num, z_num)
            HR_th_flat_env = HjRj_th_env.reshape(horiz_num, z_num)
            RR_th_flat_cloud = RjRj_th_cloud.reshape(horiz_num, z_num)
            RR_th_flat_env = RjRj_th_env.reshape(horiz_num, z_num)

            HR_qt_flat_cloud = HjRj_qt_cloud.reshape(horiz_num, z_num)
            HR_qt_flat_env = HjRj_qt_env.reshape(horiz_num, z_num)
            RR_qt_flat_cloud = RjRj_qt_cloud.reshape(horiz_num, z_num)
            RR_qt_flat_env = RjRj_qt_env.reshape(horiz_num, z_num)

            total_num = horiz_num

        LM_cloud_av = np.zeros(z_num)
        LM_env_av = np.zeros(z_num)
        MM_cloud_av = np.zeros(z_num)
        MM_env_av = np.zeros(z_num)

        HR_th_cloud_av = np.zeros(z_num)
        HR_th_env_av = np.zeros(z_num)
        RR_th_cloud_av = np.zeros(z_num)
        RR_th_env_av = np.zeros(z_num)

        HR_qt_cloud_av = np.zeros(z_num)
        HR_qt_env_av = np.zeros(z_num)
        RR_qt_cloud_av = np.zeros(z_num)
        RR_qt_env_av = np.zeros(z_num)

        for k in range(z_num):

            LM_cloud_av[k] = np.sum(LM_flat_cloud[:, k]) / total_num
            LM_env_av[k] = np.sum(LM_flat_env[:, k]) / total_num
            MM_cloud_av[k] = np.sum(MM_flat_cloud[:, k]) / total_num
            MM_env_av[k] = np.sum(MM_flat_env[:, k]) / total_num

            HR_th_cloud_av[k] = np.sum( HR_th_flat_cloud[:, k] ) / total_num
            HR_th_env_av[k] = np.sum( HR_th_flat_env[:, k] ) / total_num
            RR_th_cloud_av[k] = np.sum( RR_th_flat_cloud[:, k] ) / total_num
            RR_th_env_av[k] = np.sum( RR_th_flat_env[:, k] ) / total_num

            HR_qt_cloud_av[k] = np.sum( HR_qt_flat_cloud[:, k] ) / total_num
            HR_qt_env_av[k] = np.sum( HR_qt_flat_env[:, k] ) / total_num
            RR_qt_cloud_av[k] = np.sum( RR_qt_flat_cloud[:, k] ) / total_num
            RR_qt_env_av[k] = np.sum( RR_qt_flat_env[:, k] ) / total_num

        LM_flat_cloud = None
        LM_flat_env = None
        MM_flat_cloud = None
        MM_flat_env = None

        HR_th_flat_cloud = None
        HR_th_flat_env = None
        RR_th_flat_cloud = None
        RR_th_flat_env = None

        HR_qt_flat_cloud = None
        HR_qt_flat_env = None
        RR_qt_flat_cloud = None
        RR_qt_flat_env = None


        Cs_cloud_av_sq = (0.5 * (LM_cloud_av / MM_cloud_av))
        Cs_env_av_sq = (0.5 * (LM_env_av / MM_env_av))

        C_th_cloud_av_sq = (0.5 * (HR_th_cloud_av / RR_th_cloud_av))
        C_th_env_av_sq = (0.5 * (HR_th_env_av / RR_th_env_av))

        C_qt_cloud_av_sq = (0.5 * (HR_qt_cloud_av / RR_qt_cloud_av))
        C_qt_env_av_sq = (0.5 * (HR_qt_env_av / RR_qt_env_av))


        Cs_cloud_prof = dyn.get_Cs(Cs_cloud_av_sq)
        Cs_env_prof = dyn.get_Cs(Cs_env_av_sq)
        Cth_cloud_prof = dyn.get_Cs(C_th_cloud_av_sq)
        Cth_env_prof = dyn.get_Cs(C_th_env_av_sq)
        Cqt_cloud_prof = dyn.get_Cs(C_qt_cloud_av_sq)
        Cqt_env_prof = dyn.get_Cs(C_qt_env_av_sq)


        Cs_cloud_prof_out = xr.DataArray(Cs_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cs_cloud_prof')
        Cs_env_prof_out = xr.DataArray(Cs_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cs_env_prof')
        Cth_cloud_prof_out = xr.DataArray(Cth_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cth_cloud_prof')
        Cth_env_prof_out = xr.DataArray(Cth_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cth_env_prof')
        Cqt_cloud_prof_out = xr.DataArray(Cqt_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cqt_cloud_prof')
        Cqt_env_prof_out = xr.DataArray(Cqt_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cqt_env_prof')



        LM_cloud_av = xr.DataArray(LM_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='LijMij_cloud_prof')
        LM_env_av = xr.DataArray(LM_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='LijMij_env_prof')
        MM_cloud_av = xr.DataArray(MM_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='MijMij_cloud_prof')
        MM_env_av = xr.DataArray(MM_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='MijMij_env_prof')

        HR_th_cloud_av = xr.DataArray(HR_th_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_th_cloud_prof')
        HR_th_env_av = xr.DataArray(HR_th_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_th_env_prof')
        RR_th_cloud_av = xr.DataArray(RR_th_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_th_cloud_prof')
        RR_th_env_av = xr.DataArray(RR_th_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_th_env_prof')

        HR_qt_cloud_av = xr.DataArray(HR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_qt_cloud_prof')
        HR_qt_env_av = xr.DataArray(HR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_qt_env_prof')
        RR_qt_cloud_av = xr.DataArray(RR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_qt_cloud_prof')
        RR_qt_env_av = xr.DataArray(RR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_qt_env_prof')


        if return_fields == False:

            return Cs_cloud_prof_out, Cs_env_prof_out, Cth_cloud_prof_out, Cth_env_prof_out, Cqt_cloud_prof_out, Cqt_env_prof_out, \
                   LM_cloud_av, LM_env_av, MM_cloud_av, MM_env_av, \
                   HR_th_cloud_av, HR_th_env_av, RR_th_cloud_av, RR_th_env_av, \
                   HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av

        else:

            LijMij_cloud = xr.DataArray(LijMij_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='LijMij_cloud_field')
            LijMij_env = xr.DataArray(LijMij_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='LijMij_env_field')
            MijMij_cloud = xr.DataArray(MijMij_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='MijMij_cloud_field')
            MijMij_env = xr.DataArray(MijMij_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='MijMij_env_field')

            HjRj_th_cloud = xr.DataArray(HjRj_th_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_th_cloud_field')
            HjRj_th_env = xr.DataArray(HjRj_th_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_th_env_field')
            RjRj_th_cloud = xr.DataArray(RjRj_th_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_th_cloud_field')
            RjRj_th_env = xr.DataArray(RjRj_th_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_th_env_field')

            HjRj_qt_cloud = xr.DataArray(HjRj_qt_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_qt_cloud_field')
            HjRj_qt_env = xr.DataArray(HjRj_qt_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_qt_env_field')
            RjRj_qt_cloud = xr.DataArray(RjRj_qt_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_qt_cloud_field')
            RjRj_qt_env = xr.DataArray(RjRj_qt_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_qt_env_field')

            return Cs_cloud_prof_out, Cs_env_prof_out, Cth_cloud_prof_out, Cth_env_prof_out, Cqt_cloud_prof_out, Cqt_env_prof_out, \
                   LM_cloud_av, LM_env_av, MM_cloud_av, MM_env_av, \
                   HR_th_cloud_av, HR_th_env_av, RR_th_cloud_av, RR_th_env_av, \
                   HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av, \
                    LijMij_cloud, LijMij_env, MijMij_cloud, MijMij_env, \
                          HjRj_th_cloud, HjRj_th_env, RjRj_th_cloud, RjRj_th_env, \
                          HjRj_qt_cloud, HjRj_qt_env, RjRj_qt_cloud, RjRj_qt_env


    #
    # else:
    #
    #     mask = cloudy_and_or(data_in = dataset_in, other_var = other_var_choice, var_thres=other_var_thres, \
    #                          less_greater_threas='greater', and_or='and', cloud_liquid_threshold=10 ** (-5), \
    #                          res_counter=res_count, return_all=return_all_masks)







def get_masked_fields_HR(dataset_in, delta, res_count = None, return_fields=True, cloud_thres = 10**(-5), other_var_choice = False, \
                      other_var_thres=False, less_greater='greater', my_and_or = 'and', return_all_masks=False):


    data_qtot = xr.open_dataset(dataset_in + f'LijMij_HjRj_{delta}.nc')
    HjRj_qt = data_qtot['HR_q_total_field'].data[...]
    RjRj_qt = data_qtot['RR_q_total_field'].data[...]

    print('the shape of HjRj_qt is = ', np.shape(HjRj_qt))

    time_data = data_qtot['time']
    times = time_data.data
    nt = len(times)
    x_data = data_qtot['x_p']
    x_s = x_data.data
    y_data = data_qtot['y_p']
    y_s = y_data.data
    z_data = data_qtot['z']
    z_s = z_data.data

    data_qtot.close()


    if other_var_choice == False:

        cloud_only_mask, env_only_mask = cloud_vs_env_masks(dataset_in, cloud_liquid_threshold=cloud_thres, \
                                                            res_counter=res_count)

        HjRj_qt_cloud = ma.masked_array(HjRj_qt, mask=cloud_only_mask)
        HjRj_qt_env = ma.masked_array(HjRj_qt, mask=env_only_mask)
        RjRj_qt_cloud = ma.masked_array(RjRj_qt, mask=cloud_only_mask)
        RjRj_qt_env = ma.masked_array(RjRj_qt, mask=env_only_mask)

        HjRj_qt = None

        z_num = (RjRj_qt.shape)[-1]
        horiz_num_temp = (RjRj_qt.shape)[-2]
        horiz_num = horiz_num_temp * horiz_num_temp

        if len(RjRj_qt.shape) == 4:
            num_times = (RjRj_qt.shape)[0]
            total_num = num_times * horiz_num
            RjRj_qt = None

            HR_qt_flat_cloud = HjRj_qt_cloud.reshape(total_num, z_num)
            HR_qt_flat_env = HjRj_qt_env.reshape(total_num, z_num)
            RR_qt_flat_cloud = RjRj_qt_cloud.reshape(total_num, z_num)
            RR_qt_flat_env = RjRj_qt_env.reshape(total_num, z_num)

        else:
            RjRj_qt = None

            HR_qt_flat_cloud = HjRj_qt_cloud.reshape(horiz_num, z_num)
            HR_qt_flat_env = HjRj_qt_env.reshape(horiz_num, z_num)
            RR_qt_flat_cloud = RjRj_qt_cloud.reshape(horiz_num, z_num)
            RR_qt_flat_env = RjRj_qt_env.reshape(horiz_num, z_num)

            total_num = horiz_num


        HR_qt_cloud_av = np.zeros(z_num)
        HR_qt_env_av = np.zeros(z_num)
        RR_qt_cloud_av = np.zeros(z_num)
        RR_qt_env_av = np.zeros(z_num)

        for k in range(z_num):

            HR_qt_cloud_av[k] = np.sum( HR_qt_flat_cloud[:, k] ) / total_num
            HR_qt_env_av[k] = np.sum( HR_qt_flat_env[:, k] ) / total_num
            RR_qt_cloud_av[k] = np.sum( RR_qt_flat_cloud[:, k] ) / total_num
            RR_qt_env_av[k] = np.sum( RR_qt_flat_env[:, k] ) / total_num


        HR_qt_flat_cloud = None
        HR_qt_flat_env = None
        RR_qt_flat_cloud = None
        RR_qt_flat_env = None


        C_qt_cloud_av_sq = (0.5 * (HR_qt_cloud_av / RR_qt_cloud_av))
        C_qt_env_av_sq = (0.5 * (HR_qt_env_av / RR_qt_env_av))


        Cqt_cloud_prof = dyn.get_Cs(C_qt_cloud_av_sq)
        Cqt_env_prof = dyn.get_Cs(C_qt_env_av_sq)

        Cqt_cloud_prof_out = xr.DataArray(Cqt_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cqt_cloud_prof')
        Cqt_env_prof_out = xr.DataArray(Cqt_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='Cqt_env_prof')




        HR_qt_cloud_av = xr.DataArray(HR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_qt_cloud_prof')
        HR_qt_env_av = xr.DataArray(HR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='HjRj_qt_env_prof')
        RR_qt_cloud_av = xr.DataArray(RR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_qt_cloud_prof')
        RR_qt_env_av = xr.DataArray(RR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
                                     dims=['time', "z"], name='RjRj_qt_env_prof')


        if return_fields == False:

            return Cqt_cloud_prof_out, Cqt_env_prof_out, \
                   HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av

        else:


            HjRj_qt_cloud = xr.DataArray(HjRj_qt_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_qt_cloud_field')
            HjRj_qt_env = xr.DataArray(HjRj_qt_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='HjRj_qt_env_field')
            RjRj_qt_cloud = xr.DataArray(RjRj_qt_cloud[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_qt_cloud_field')
            RjRj_qt_env = xr.DataArray(RjRj_qt_env[np.newaxis, ...],
                                             coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'z': z_s},
                                             dims=["time", "x_p", "y_p", "z"], name='RjRj_qt_env_field')

            return Cqt_cloud_prof_out, Cqt_env_prof_out, \
                   HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av, \
                          HjRj_qt_cloud, HjRj_qt_env, RjRj_qt_cloud, RjRj_qt_env
