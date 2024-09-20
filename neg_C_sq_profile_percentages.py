import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import mask_cloud_vs_env as clo
import matplotlib.ticker as ticker
import numpy.ma as ma
import dynamic_functions as dyn
import functools

print = functools.partial(print, flush=True)

np.seterr(divide='ignore')



print('test')


BOMEX_homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/LM/update/BOMEX_m0020_g0800_all_'
BOMEX_dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_'

ARM_homedir = f'/work/scratch-pw3/apower/ARM/second_filt/LM/update/diagnostics_3d_ts_'
ARM_dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/neg_C_sq_profiles/'
save_dir = '/work/scratch-pw3/apower/neg_Csq_values/'
#'/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir, exist_ok = True)
os.makedirs(save_dir, exist_ok = True)

cloud_field = f'f(q_cloud_liquid_mass_on_p)_r'

fields = [ ['LM_field', 'MM_field'], ['HR_th_field', 'RR_th_field'],
           ['HR_th_L_field', 'RR_th_L_field'], ['HR_q_field', 'RR_q_field'] ]

bomex_res=['40_80', '160_320', '640_1280']
arm_res=['50_100', '200_400', '800_1600']
Deltas = ['4$\\Delta$', '16$\\Delta$', '64$\\Delta$']

times = ['14400', '18000', '25200', '32400', '39600']
z_i_all = [430, 795, 955, 1095, 1255]

dx_BOMEX=20
z_BOMEX = np.arange(0, 3020, 20)
z_i_BOMEX = [430] #1120


dx_ARM=25
z_ARM = np.arange(0, 4410, 10)
z_i_ARM = [795, 955, 1095, 1255]
ARM_times = ['18000', '25200', '32400', '39600']

BOMEX_ML_range = np.array([100, 400])
BOMEX_CL_range = np.array([500, 1500])

ARM_ML_range = np.array([ [100, 700], [100, 900], [100, 1000], [100, 1000] ])
ARM_CL_range = np.array([ [900, 1050], [1100, 1400], [1250, 1850], [1400, 2150] ])

BOMEX_ML_ind = BOMEX_ML_range/20
BOMEX_CL_ind = BOMEX_CL_range/20
ARM_ML_ind = ARM_ML_range/10
ARM_CL_ind = ARM_CL_range/10


list_of_C_latex = ['$C_s^2$', '$C_\\theta^2$', '$C_{\\theta_L^2}$', '$C_{q_t}^2$']
list_of_c_names = ['Cs', 'C_th', 'C_th_L', 'C_qt']


field_names = ['Cs_field', 'Cth_field', 'Cth_L_field', 'Cqt_field']
field_latex = ['$C_{s}$', '$C_{\\theta}$', '$C_{\\theta_L}$', '$C_{q_t}$']




def get_data_per_delta(dir_in, dir_cloud, C, time, res_in):

    data_s4 = xr.open_dataset(dir_in+f'{time}_{C}_{res_in[0]}.nc')
    data_s16 = xr.open_dataset(dir_in+f'{time}_{C}_{res_in[1]}.nc')
    data_s64 = xr.open_dataset(dir_in+f'{time}_{C}_{res_in[2]}.nc')

    data_cl4 = dir_cloud+f'{time}_gaussian_filter_ga00_gaussian_filter_ga00.nc'
    data_cl16 = dir_cloud+f'{time}_gaussian_filter_ga02_gaussian_filter_ga00.nc'
    data_cl64 = dir_cloud+f'{time}_gaussian_filter_ga04_gaussian_filter_ga00.nc'

    data_cl_list = [data_cl4, data_cl16, data_cl64]
    data_s_list = [data_s4, data_s16, data_s64]

    return data_s_list, data_cl_list





def cloud_and_env_masks(dataset_in, cloud_liquid_threshold=10**(-7), grid='p'):

    ds_in = xr.open_dataset(dataset_in)

    if f'f(q_cloud_liquid_mass_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(q_cloud_liquid_mass_on_{grid})_r']
    elif f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r']
    elif 'q_cloud_liquid_mass' in ds_in:
        q_in = ds_in['q_cloud_liquid_mass']


    q_cloud = q_in.data[-1,...]

    if cloud_liquid_threshold == 0:
        masked_q_cloud = ma.masked_less_equal(q_cloud, cloud_liquid_threshold)  # masking lower values
        masked_q_env = ma.masked_greater(q_cloud, cloud_liquid_threshold)
    else:
        masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold) #masking lower values
        masked_q_env = ma.masked_greater_equal(q_cloud, cloud_liquid_threshold) #masking larger values

    cloud_only_mask = ma.getmaskarray(masked_q_cloud)
    env_only_mask = ma.getmaskarray(masked_q_env)

    masked_q_cloud = None
    masked_q_env = None

    return cloud_only_mask, env_only_mask


labels_title = ['BOMEX', 'ARM 10:30L', 'ARM 12:30L', 'ARM 14:30L', 'ARM 16:30L']




def negs_in_field(plotdir, field, c, c_latex, z, z_i, time_in, nt_in, data_field_list, data_cl_list):

    deltas = ['4$\\Delta$', '16$\\Delta$', '64$\\Delta$']
    colours = ['tab:orange', 'tab:red', 'tab:cyan']

    fig1 = plt.figure(figsize=(4, 6))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(4, 6))
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(4, 6))
    ax3 = fig3.add_subplot(111)

    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = cloud_and_env_masks(data_cl_list[i])

        data_field_LM = data_field_list[i][f'{field[0]}'].data[-1,...]

        data_field_cloud_temp = ma.masked_array(data_field_LM, mask=cloud_only_mask)
        print('applied cloud mask, shape of cloud_only_fields = ', np.shape(data_field_cloud_temp))
        data_field_env_temp = ma.masked_array(data_field_LM, mask=env_only_mask)
        print('applied env mask, shape of env_only_fields = ', np.shape(data_field_env_temp))

        total_points_each_level = len(data_field_LM[:,0,0])*len(data_field_LM[0,:,0])

        data_field_cloud = ma.filled(data_field_cloud_temp, 0)
        data_field_env = ma.filled(data_field_env_temp, 0)

        data_field_cloud_temp = None
        data_field_env_temp = None



        print('shape of env is = ', np.shape(data_field_env), 'shape of cloud is = ', np.shape(data_field_cloud))


        counter_env = np.zeros(len(data_field_env[0, 0, :]), dtype=float)
        counter_cloud = np.zeros(len(data_field_cloud[0,0,:]), dtype=float)

        counter_cloud_no_messin = np.zeros(len(data_field_cloud[0, 0, :]), dtype=float)

        number_of_points_env = np.zeros(len(data_field_env[0, 0, :]), dtype=float)
        number_of_points_cloud = np.zeros(len(data_field_cloud[0,0,:]), dtype=float)

        counter_LM = np.zeros(len(data_field_env[0, 0, :]), dtype=float)

        # total_count = len(data_field_LM[:,0,0])*len(data_field_LM[0,:,0])*len(data_field_LM[0,0,:])

        for j in range(len(data_field_cloud[0,0,:])):

            counter_LM[j] = np.count_nonzero(data_field_LM[:,:,j] < 0)

            counter_cloud[j] = np.count_nonzero(data_field_cloud[:,:,j] < 0)
            counter_cloud_no_messin[j] = counter_cloud[j]

            counter_env[j] = np.count_nonzero(data_field_env[:, :, j] < 0)

            number_of_points_env[j] = np.count_nonzero(data_field_env[:,:,j]) #ma.MaskedArray.count(data_field_env[:,:,j])

            number_of_points_cloud[j] = np.count_nonzero(data_field_cloud[:,:,j]) #ma.MaskedArray.count(data_field_cloud[:,:,j])


            if counter_cloud[j] > number_of_points_cloud[j] or number_of_points_cloud[j] == 0:
                print(j)
                print('number of negatives in cloud = ', counter_cloud[j],
                      ' and total number of cloudy points is = ', number_of_points_cloud[j])
                counter_cloud_no_messin[j] = np.nan

        print('counted neg vals in whole domain')
        np.save(f'{save_dir}number_domain_neg_{c}_vs_z_{time_in}.npy', counter_LM)
        print('counted neg vals in cloud')
        np.save(f'{save_dir}number_neg_IC_{c}_vs_z_{time_in}.npy', counter_cloud)
        print('counted neg vals in env')
        np.save(f'{save_dir}number_neg_CFE_{c}_vs_z_{time_in}.npy', counter_env)
        print('counted points in cloud')
        np.save(f'{save_dir}number_total_IC_points_{c}_vs_z_{time_in}.npy', number_of_points_env)
        print('counted points in env')
        np.save(f'{save_dir}number_total_CFE_points_{c}_vs_z_{time_in}.npy', number_of_points_cloud)


        ax1.plot((counter_env/number_of_points_env)*100, z/z_i, label=f'{deltas[i]}', color=colours[i])
        ax1.plot((counter_cloud_no_messin/number_of_points_cloud)*100, z/z_i, linestyle='--', color=colours[i]) #label='$C_s$ IC')
        print(f'plotted profile for {deltas[i]}')

        ax2.plot(counter_env, z/z_i, label=f'{deltas[i]}', color=colours[i])
        ax2.plot(counter_cloud, z/z_i, linestyle='--', color=colours[i])

        ax3.plot((counter_env / total_points_each_level) * 100, z / z_i, label=f'{deltas[i]}', color=colours[i])
        ax3.plot((counter_cloud_no_messin / total_points_each_level) * 100, z / z_i, linestyle='--', color=colours[i])

    ax1.legend()
    ax2.legend()
    ax3.legend()

    # og_xtic = plt.xticks()
    # plt.xticks(og_xtic[0],
    #            np.round(np.linspace((0) * (20 / 480), (151) * (20 / 480), len(og_xtic[0])), 1))

    ax1.set_title(f"{labels_title[nt_in]}", fontsize=13)
    ax1.set_ylabel("$z/z_{ML}$ $z_{ML} = $"+f'{z_i}m', fontsize=16)
    ax1.set_xlabel(f"Percentage of Negative {c_latex} Values", fontsize=13)
    fig1.savefig(plotdir + f'percent_neg_{c}_vs_z_{time_in}.pdf', bbox_inches='tight')
    # ax1.clf()

    ax2.set_title(f"{labels_title[nt_in]}", fontsize=13)
    ax2.set_ylabel("$z/z_{ML}$ $z_{ML} = $"+f'{z_i}m', fontsize=16)
    ax2.set_xlabel(f"Number of Negative {c_latex} Values", fontsize=13)
    fig2.savefig(plotdir + f'number_neg_{c}_vs_z_{time_in}.pdf', bbox_inches='tight')

    ax3.set_title(f"{labels_title[nt_in]}", fontsize=13)
    ax3.set_ylabel("$z/z_{ML}$ $z_{ML} = $"+f'{z_i}m', fontsize=16)
    ax3.set_xlabel(f"% of Negative {c_latex} Across Domain", fontsize=13)
    fig3.savefig(plotdir + f'number_neg_{c}_vs_z_{time_in}.pdf', bbox_inches='tight')
    #ax2.clf()

    print(f'plotted all deltas neg vs z for {c}')


for iters in range(len(list_of_C_latex)):

    C = list_of_c_names[iters]
    field = fields[iters]
    c_lat = list_of_C_latex[iters]

    for nt, t in enumerate(times):
        if t == '14400':
            dir_in = BOMEX_homedir
            dir_cloud = BOMEX_dir_contour
            res_in = bomex_res
            z = z_BOMEX
            z_i = z_i_all[nt]
        else:
            dir_in = ARM_homedir
            dir_cloud = ARM_dir_contour
            res_in = arm_res
            z = z_ARM
            z_i = z_i_all[nt]

        data_C_list, data_cloud_list = get_data_per_delta(dir_in, dir_cloud, C, t, res_in)

        negs_in_field(plotdir, field, C, c_lat, z, z_i, t, nt, data_C_list, data_cloud_list)