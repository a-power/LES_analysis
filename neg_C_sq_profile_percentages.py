import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import mask_cloud_vs_env as clo
import matplotlib.ticker as ticker
import numpy.ma as ma
import dynamic_functions as dyn
import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)





BOMEX_homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/LM/update/BOMEX_m0020_g0800_all_'
BOMEX_dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_'

ARM_homedir = f'/work/scratch-pw3/apower/ARM/second_filt/LM/update/diagnostics_3d_ts_'
ARM_dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/neg_C_sq_profiles/'
#'/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir, exist_ok = True)

cloud_field = f'f(q_cloud_liquid_mass_on_p)_r'

fields = [ ['LM_field', 'MM_field'], ['HR_th_field', 'RR_th_field'],
           ['HR_th_L_field', 'RR_th_L_field'], ['HR_q_field', 'RR_q_field'] ]

bomex_res=['40_80', '160_320', '640_1280']
arm_res=['50_100', '200_400', '800_1600']
Deltas = ['4$\\Delta$', '16$\\Delta$', '64$\\Delta$']

times = ['14400', '18000', '25200', '32400', '39600']

dx_BOMEX=20
z_BOMEX = np.arange(0, 3020, 20)
z_i_BOMEX = 430 #1120


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







def negs_in_field(plotdir, field, c, z, z_i, data_field_list, data_cl_list):

    deltas = ['4$\\Delta$', '16$\\Delta$', '64$\\Delta$']
    colours = ['tab:orange', 'tab:red', 'tab:cyan']

    plt.figure(figsize=(7, 6))

    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = cloud_and_env_masks(data_cl_list[i])

        data_field_LM = data_field_list[i][f'{field[0]}'].data[-1,...]
        data_field_MM = data_field_list[i][f'{field[1]}'].data[-1, ...]
        print('imported LM and MM')

        C_field_sq = 0.5 * data_field_LM / data_field_MM
        print('calced c^2')

        data_field_LM = None
        data_field_MM = None

        data_field_cloud = ma.masked_array(C_field_sq, mask=cloud_only_mask)
        print('applied cloud mask')
        data_field_env = ma.masked_array(C_field_sq, mask=env_only_mask)
        print('applied env mask')

        C_field_sq = None

        print('shape of env is = ', np.shape(data_field_env), 'shape of cloud is = ', np.shape(data_field_cloud))

        number_of_points_env = ma.MaskedArray.count(data_field_env)
        print('counted points in cloud')
        number_of_points_cloud = ma.MaskedArray.count(data_field_cloud)
        print('counted points in env')

        counter_env = np.zeros(len(data_field_env[0, 0, :]))
        counter_cloud = np.zeros(len(data_field_cloud[0,0,:]))

        for j in range(len(data_field_cloud[0,0,:])):
            counter_cloud[j] = np.count_nonzero(data_field_cloud[:,:,j] < 0)
            print('counted neg vals in cloud')
            counter_env[j] = np.count_nonzero(data_field_env[:, :, j] < 0)
            print('counted neg vals in env')


        plt.plot((counter_env/number_of_points_env)*100, z/z_i, label=f'{deltas[i]}', color=colours[i])
        plt.plot((counter_cloud/number_of_points_cloud)*100, z/z_i, linestyle='--', color=colours[i]) #label='$C_s$ IC')
        print(f'plotted profile for {deltas[i]}')

    plt.legend()

    # og_xtic = plt.xticks()
    # plt.xticks(og_xtic[0],
    #            np.round(np.linspace((0) * (20 / 480), (151) * (20 / 480), len(og_xtic[0])), 1))

    plt.ylabel("$z/z_{ML}$", fontsize=16)
    plt.xlabel(f"Percentage of Negative {list_of_C_latex[C]} Values", fontsize=16)
    plt.savefig(plotdir + f'neg_{c}_vs_z.pdf', bbox_inches='tight')
    plt.clf()

    print(f'plotted all deltas neg vs z for {c}')

    plt.close('all')


for iters in range(len(list_of_C_latex)):

    C = list_of_c_names[iters]
    field = fields[iters]

    for nt, t in enumerate(times):
        if t == '14400':
            dir_in = BOMEX_homedir
            dir_cloud = BOMEX_dir_contour
            res_in = bomex_res
            z = z_BOMEX
            z_i = z_i_BOMEX
        else:
            dir_in = ARM_homedir
            dir_cloud = ARM_dir_contour
            res_in = arm_res
            z = z_ARM
            z_i = z_i_ARM

        data_C_list, data_cloud_list = get_data_per_delta(dir_in, dir_cloud, C, t, res_in)

        negs_in_field(plotdir, field, C, z, z_i, data_C_list, data_cloud_list)