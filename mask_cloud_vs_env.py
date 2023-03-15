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
    print('shape of th_v_prime before reshape is:', np.shape(th_v_prime_temp))
    th_v_prime = th_v_prime_temp.reshape(nt, horiz_num_temp, horiz_num_temp, z_num)
    th_v_prime_temp = None

    return th_v_prime




def cloud_vs_env_masks(data_in, cloud_liquid_threshold=10**(-5), res_counter=None, grid='p'):

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



def cloudy_and_or(data_in, other_var, var_thres, less_greater=['less'], and_or = ['and'], \
                  cloud_liquid_threshold=10**(-5), res_counter=None, return_all = False, grid='p'):

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
        if f'f({other_var[1]}_on_{grid})_r' in ds_in:
            extra_var_in = ds_in[f'f({other_var[1]}_on_{grid})_r']
        else:
            extra_var_in = ds_in[f'{other_var[1]}']

        if other_var[1] == f'f(f(th_v_on_{grid})_r_on_{grid})_r':
            extra_var_temp = extra_var_in.data
            extra_var_data = get_th_v_prime(extra_var_temp)
        else:
            extra_var_data = var_in.data
        print('loaded other var:', other_var[1])

        if less_greater[1] == 'less':
            extra_masked_var = ma.masked_less(extra_var_data, var_thres[1])
        elif less_greater[1] == 'greater':
            extra_masked_var = ma.masked_greater_equal(extra_var_data, var_thres[1])
        extra_var_mask = ma.getmaskarray(extra_masked_var)
        print('masked other var:', other_var[0])

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

#

# def get_masked_fields(dataset_in, contour_field_in, field, res_count = None, return_fields=True,
#                       cloud_thres = 10**(-5), other_var_choice = False, horiz_av=False, \
#                       other_var_thres=False, less_greater='less', my_and_or = 'and', return_all_masks=False, grid='p'):
#
#     if field == 'Cs_field':
#         field_name = '$C_s$'
#         field_name_sq = '$C_s^2$'
#     if field == 'Cth_field':
#         field_name = '$C_{\\theta}$'
#         field_name_sq = '$C_{\\theta}^2$'
#     if field == 'Cqt_field':
#         field_name = '$C_{qt}$'
#         field_name_sq = '$C_{qt}^2$'
#
#     if field == 'LM_field':
#         field_name = '$LM$'
#     if field == 'HR_th_field':
#         field_name = '$HR_{\\theta}$'
#     if field == 'HR_qt_field':
#         field_name = '$HR_{qt}$'
#     if field == 'MM_field':
#         field_name = '$MM$'
#     if field == 'RR_th_field':
#         field_name = '$RR_{\\theta}$'
#     if field == 'RR_qt_field':
#         field_name = '$RR_{qt}$'
#
#
#     data_set = xr.open_dataset(dataset_in)
#
#     time_data = data_set['time']
#     times = time_data.data
#     nt = len(times)
#     x_data = data_set['x_p']
#     x_s = x_data.data
#     y_data = data_set['y_p']
#     y_s = y_data.data
#     z_data = data_set['z']
#     z_s = z_data.data
#     zn_data = data_set['zn']
#     zn_s = zn_data.data
#
#     if field == 'Cs_field':
#         print('length of time array for LM is ', len(data_set[f'f(LM_field_on_{grid})_r'].data[:, 0, 0, 0]))
#         num_field = data_set[f'f(LM_field_on_{grid})_r'].data[...]
#         den_field = data_set[f'f(MM_field_on_{grid})_r'].data[...]
#
#         data_field_sq = 0.5 * num_field / den_field
#         data_field = dyn.get_Cs(data_field_sq)
#
#     elif field == 'Cth_field':
#         print('length of time array for HR_th is ', len(data_set[f'f(HR_th_field_on_{grid})_r'].data[:, 0, 0, 0]))
#         num_field = data_set[f'f(HR_th_field_on_{grid})_r'].data[...]
#         den_field = data_set[f'f(RR_th_field_on_{grid})_r'].data[...]
#
#         data_field_sq = 0.5 * num_field / den_field
#         data_field = dyn.get_Cs(data_field_sq)
#
#     elif field == 'Cqt_field':
#         print('length of time array for HR_qt is ', len(data_set[f'f(HR_q_total_field_on_{grid})_r'].data[:, 0, 0, 0]))
#         num_field = data_set[f'f(HR_q_total_field_on_{grid})_r'].data[...]
#         den_field = data_set[f'f(RR_q_total_field_on_{grid})_r'].data[...]
#
#         data_field_sq = 0.5 * num_field / den_field
#         data_field = dyn.get_Cs(data_field_sq)
#
#     else:
#         print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_{grid})_r'].data[:, 0, 0, 0]))
#         data_field = data_set[f'f({field}_on_{grid})_r'].data[...]
#
#     data_set.close()
#
#
#     if other_var_choice == False:
#
#         cloud_only_mask, env_only_mask = cloud_vs_env_masks(contour_field_in, cloud_liquid_threshold=cloud_thres)
#
#         num_field_cloud = ma.masked_array(num_field, mask=cloud_only_mask)
#         num_field_env = ma.masked_array(num_field, mask=env_only_mask)
#         den_field_cloud = ma.masked_array(den_field, mask=cloud_only_mask)
#         den_field_env = ma.masked_array(den_field, mask=env_only_mask)
#
#         data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
#         data_field_env = ma.masked_array(data_field, mask=env_only_mask)
#         data_field_sq_cloud = ma.masked_array(data_field_sq, mask=cloud_only_mask)
#         data_field_sq_env = ma.masked_array(data_field_sq, mask=env_only_mask)
#
#         if horiz_av == True:
#
#             z_num = len(z_s)
#             horiz_num_temp = len(y_s)
#             horiz_num = horiz_num_temp * horiz_num_temp
#
#             if nt == 4:
#                 total_num = nt * horiz_num
#
#                 LM_flat_cloud = LijMij_cloud.reshape(total_num, z_num)
#                 LM_flat_env = LijMij_env.reshape(total_num, z_num)
#                 MM_flat_cloud = MijMij_cloud.reshape(total_num, z_num)
#                 MM_flat_env = MijMij_env.reshape(total_num, z_num)
#
#                 HR_th_flat_cloud = HjRj_th_cloud.reshape(total_num, z_num)
#                 HR_th_flat_env = HjRj_th_env.reshape(total_num, z_num)
#                 RR_th_flat_cloud = RjRj_th_cloud.reshape(total_num, z_num)
#                 RR_th_flat_env = RjRj_th_env.reshape(total_num, z_num)
#
#                 HR_qt_flat_cloud = HjRj_qt_cloud.reshape(total_num, z_num)
#                 HR_qt_flat_env = HjRj_qt_env.reshape(total_num, z_num)
#                 RR_qt_flat_cloud = RjRj_qt_cloud.reshape(total_num, z_num)
#                 RR_qt_flat_env = RjRj_qt_env.reshape(total_num, z_num)
#
#             else:
#                 RjRj_qt = None
#                 LM_flat_cloud = LijMij_cloud.reshape(horiz_num, z_num)
#                 LM_flat_env = LijMij_env.reshape(horiz_num, z_num)
#                 MM_flat_cloud = MijMij_cloud.reshape(horiz_num, z_num)
#                 MM_flat_env = MijMij_env.reshape(horiz_num, z_num)
#
#                 HR_th_flat_cloud = HjRj_th_cloud.reshape(horiz_num, z_num)
#                 HR_th_flat_env = HjRj_th_env.reshape(horiz_num, z_num)
#                 RR_th_flat_cloud = RjRj_th_cloud.reshape(horiz_num, z_num)
#                 RR_th_flat_env = RjRj_th_env.reshape(horiz_num, z_num)
#
#                 HR_qt_flat_cloud = HjRj_qt_cloud.reshape(horiz_num, z_num)
#                 HR_qt_flat_env = HjRj_qt_env.reshape(horiz_num, z_num)
#                 RR_qt_flat_cloud = RjRj_qt_cloud.reshape(horiz_num, z_num)
#                 RR_qt_flat_env = RjRj_qt_env.reshape(horiz_num, z_num)
#
#                 total_num = horiz_num
#
#             LM_cloud_av = np.zeros(z_num)
#             LM_env_av = np.zeros(z_num)
#             MM_cloud_av = np.zeros(z_num)
#             MM_env_av = np.zeros(z_num)
#
#             HR_th_cloud_av = np.zeros(z_num)
#             HR_th_env_av = np.zeros(z_num)
#             RR_th_cloud_av = np.zeros(z_num)
#             RR_th_env_av = np.zeros(z_num)
#
#             HR_qt_cloud_av = np.zeros(z_num)
#             HR_qt_env_av = np.zeros(z_num)
#             RR_qt_cloud_av = np.zeros(z_num)
#             RR_qt_env_av = np.zeros(z_num)
#
#             for k in range(z_num):
#
#                 LM_cloud_av[k] = np.sum(LM_flat_cloud[:, k]) / total_num
#                 LM_env_av[k] = np.sum(LM_flat_env[:, k]) / total_num
#                 MM_cloud_av[k] = np.sum(MM_flat_cloud[:, k]) / total_num
#                 MM_env_av[k] = np.sum(MM_flat_env[:, k]) / total_num
#
#                 HR_th_cloud_av[k] = np.sum( HR_th_flat_cloud[:, k] ) / total_num
#                 HR_th_env_av[k] = np.sum( HR_th_flat_env[:, k] ) / total_num
#                 RR_th_cloud_av[k] = np.sum( RR_th_flat_cloud[:, k] ) / total_num
#                 RR_th_env_av[k] = np.sum( RR_th_flat_env[:, k] ) / total_num
#
#                 HR_qt_cloud_av[k] = np.sum( HR_qt_flat_cloud[:, k] ) / total_num
#                 HR_qt_env_av[k] = np.sum( HR_qt_flat_env[:, k] ) / total_num
#                 RR_qt_cloud_av[k] = np.sum( RR_qt_flat_cloud[:, k] ) / total_num
#                 RR_qt_env_av[k] = np.sum( RR_qt_flat_env[:, k] ) / total_num
#
#             LM_flat_cloud = None
#             LM_flat_env = None
#             MM_flat_cloud = None
#             MM_flat_env = None
#
#             HR_th_flat_cloud = None
#             HR_th_flat_env = None
#             RR_th_flat_cloud = None
#             RR_th_flat_env = None
#
#             HR_qt_flat_cloud = None
#             HR_qt_flat_env = None
#             RR_qt_flat_cloud = None
#             RR_qt_flat_env = None
#
#
#             Cs_cloud_av_sq = (0.5 * (LM_cloud_av / MM_cloud_av))
#             Cs_env_av_sq = (0.5 * (LM_env_av / MM_env_av))
#
#             C_th_cloud_av_sq = (0.5 * (HR_th_cloud_av / RR_th_cloud_av))
#             C_th_env_av_sq = (0.5 * (HR_th_env_av / RR_th_env_av))
#
#             C_qt_cloud_av_sq = (0.5 * (HR_qt_cloud_av / RR_qt_cloud_av))
#             C_qt_env_av_sq = (0.5 * (HR_qt_env_av / RR_qt_env_av))
#
#
#             Cs_cloud_prof = dyn.get_Cs(Cs_cloud_av_sq)
#             Cs_env_prof = dyn.get_Cs(Cs_env_av_sq)
#             Cth_cloud_prof = dyn.get_Cs(C_th_cloud_av_sq)
#             Cth_env_prof = dyn.get_Cs(C_th_env_av_sq)
#             Cqt_cloud_prof = dyn.get_Cs(C_qt_cloud_av_sq)
#             Cqt_env_prof = dyn.get_Cs(C_qt_env_av_sq)
#
#
#             Cs_cloud_prof_out = xr.DataArray(Cs_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cs_cloud_prof')
#             Cs_env_prof_out = xr.DataArray(Cs_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cs_env_prof')
#             Cth_cloud_prof_out = xr.DataArray(Cth_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cth_cloud_prof')
#             Cth_env_prof_out = xr.DataArray(Cth_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cth_env_prof')
#             Cqt_cloud_prof_out = xr.DataArray(Cqt_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cqt_cloud_prof')
#             Cqt_env_prof_out = xr.DataArray(Cqt_env_prof[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='Cqt_env_prof')
#
#
#
#             LM_cloud_av = xr.DataArray(LM_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='LijMij_cloud_prof')
#             LM_env_av = xr.DataArray(LM_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='LijMij_env_prof')
#             MM_cloud_av = xr.DataArray(MM_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='MijMij_cloud_prof')
#             MM_env_av = xr.DataArray(MM_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='MijMij_env_prof')
#
#             HR_th_cloud_av = xr.DataArray(HR_th_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='HjRj_th_cloud_prof')
#             HR_th_env_av = xr.DataArray(HR_th_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='HjRj_th_env_prof')
#             RR_th_cloud_av = xr.DataArray(RR_th_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='RjRj_th_cloud_prof')
#             RR_th_env_av = xr.DataArray(RR_th_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='RjRj_th_env_prof')
#
#             HR_qt_cloud_av = xr.DataArray(HR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='HjRj_qt_cloud_prof')
#             HR_qt_env_av = xr.DataArray(HR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='HjRj_qt_env_prof')
#             RR_qt_cloud_av = xr.DataArray(RR_qt_cloud_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='RjRj_qt_cloud_prof')
#             RR_qt_env_av = xr.DataArray(RR_qt_env_av[np.newaxis, ...], coords={'time': [nt], 'z': z_s},
#                                          dims=['time', "z"], name='RjRj_qt_env_prof')
#
#
#             if return_fields == False:
#
#                 return Cs_cloud_prof_out, Cs_env_prof_out, Cth_cloud_prof_out, Cth_env_prof_out, Cqt_cloud_prof_out, Cqt_env_prof_out, \
#                        LM_cloud_av, LM_env_av, MM_cloud_av, MM_env_av, \
#                        HR_th_cloud_av, HR_th_env_av, RR_th_cloud_av, RR_th_env_av, \
#                        HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av

        #else:

            # num_field_cloud = xr.DataArray(num_field_cloud,
            #                                  coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
            #                                  dims=["time", "x_p", "y_p", "z"], name='LijMij_cloud_field')
            # num_field_env = xr.DataArray(num_field_env,
            #                                  coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
            #                                  dims=["time", "x_p", "y_p", "z"], name='LijMij_env_field')
            # den_field_cloud = xr.DataArray(den_field_cloud,
            #                                  coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
            #                                  dims=["time", "x_p", "y_p", "z"], name='MijMij_cloud_field')
            # den_field_env = xr.DataArray(den_field_env,
            #                                  coords={'time': times, 'x_p': x_s, 'y_p': y_s, 'zn': zn_s},
            #                                  dims=["time", "x_p", "y_p", "z"], name='MijMij_env_field')

            # data_field_cloud
            # data_field_env
            # data_field_sq_cloud
            # data_field_sq_env


    #
    #     mask = cloudy_and_or(data_in = dataset_in, other_var = other_var_choice, var_thres=other_var_thres, \
    #                          less_greater_threas='greater', and_or='and', cloud_liquid_threshold=10 ** (-5), \
    #                          res_counter=res_count, return_all=return_all_masks)



