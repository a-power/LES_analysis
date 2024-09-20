import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib.ticker as ticker
import numpy.ma as ma
import dynamic_functions as dyn
import functools


parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, default='Cs')
parser.add_argument('--n', type=int, default=0)

args = parser.parse_args()
chose_C = args.c
chose_case = args.n

print = functools.partial(print, flush=True)


print('test')



BOMEX_homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/LM/update/BOMEX_m0020_g0800_all_'
BOMEX_dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_'

ARM_homedir = f'/work/scratch-pw3/apower/ARM/second_filt/LM/update/diagnostics_3d_ts_'
ARM_dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_'

save_dir = '/work/scratch-pw3/apower/distrib_of_C/'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/distribs/'
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

BOMEX_ML_range = np.array([100, 400])
BOMEX_CL_range = np.array([500, 1500])

ARM_ML_range = np.array([ [100, 700], [100, 900], [100, 1000], [100, 1000] ])
ARM_CL_range = np.array([ [900, 1050], [1100, 1400], [1250, 1850], [1400, 2150] ])

BOMEX_ML_ind = BOMEX_ML_range/20
BOMEX_CL_ind = BOMEX_CL_range/20
ARM_ML_ind = ARM_ML_range/10
ARM_CL_ind = ARM_CL_range/10


set_bins=50

field_names = ['Cs_field', 'Cth_field', 'Cth_L_field', 'Cqt_field']
field_latex = ['$C_{s}$', '$C_{\\theta}$', '$C_{\\theta_L}$', '$C_{q_t}$']

field_names_sq = ['Cs_sq_field', 'Cth_sq_field', 'Cth_L_sq_field', 'Cqt_sq_field']
field_latex_sq = ['$C_{s}^2$', '$C_{\\theta}^2$', '$C_{\\theta_L}^2$', '$C_{q_t}^2$']



def get_data_per_delta(dir_in, dir_cloud, c_in, time, res_in):

#     data_s4 = xr.open_dataset(dir_in+f'{time}_{c_in}_{res_in[0]}.nc')
#     data_s16 = xr.open_dataset(dir_in+f'{time}_{c_in}_{res_in[1]}.nc')
#     data_s64 = xr.open_dataset(dir_in+f'{time}_{c_in}_{res_in[2]}.nc')
#
#     print('datasets Cs opened')
#
#     if c_in == 'Cs':
#         c_type = 0
#     elif c_in == 'C_th':
#         c_type = 1
#     elif c_in == 'C_th_L':
#         c_type = 2
#     elif c_in == 'C_qt':
#         c_type = 3
#
#
#     data_s4_LM = data_s4[f'{fields[c_type][0]}'].data[-1,...]
#     data_s16_LM = data_s16[f'{fields[c_type][0]}'].data[-1,...]
#     data_s64_LM = data_s64[f'{fields[c_type][0]}'].data[-1,...]
#
#     print('shape of LM is = ', np.shape(data_s64_LM))
#
#     data_s4_MM = data_s4[f'{fields[c_type][1]}'].data[-1,...]
#     data_s16_MM = data_s16[f'{fields[c_type][1]}'].data[-1,...]
#     data_s64_MM = data_s64[f'{fields[c_type][1]}'].data[-1,...]
#
#     data_s_list = [dyn.get_Cs_where(0.5*data_s4_LM/data_s4_MM), dyn.get_Cs_where(0.5*data_s16_LM/data_s16_MM),
#                    dyn.get_Cs_where(0.5*data_s64_LM/data_s64_MM)]
#
#     print(f'{c_in} calculated')
#
#     # np.save(f'{save_dir}{c_in}_case_{time}_deltas_field', data_s_list)
#     data_s4_LM = None
#     data_s16_LM = None
#     data_s64_LM = None
#     data_s4_MM = None
#     data_s16_MM = None
#     data_s64_MM = None
#
#     data_s4.close()
#     data_s16.close()
#     data_s64.close()
#     print(f'{c_in} saved')
#
#     return
#
# if chose_case == 0:
#     get_data_per_delta(BOMEX_homedir, BOMEX_dir_contour, chose_C, times[0], bomex_res)
# else:
#     get_data_per_delta(ARM_homedir, ARM_dir_contour, chose_C, times[chose_case], arm_res)


    #
    #
    # data_th4 = xr.open_dataset(dir_in+f'{time}_C_th_{res_in[0]}.nc')
    # data_th16 = xr.open_dataset(dir_in+f'{time}_C_th_{res_in[1]}.nc')
    # data_th64 = xr.open_dataset(dir_in+f'{time}_C_th_{res_in[2]}.nc')
    #
    # data_th4_LM = data_th4[f'{fields[1][0]}'].data[-1,...]
    # data_th16_LM = data_th16[f'{fields[1][0]}'].data[-1,...]
    # data_th64_LM = data_th64[f'{fields[1][0]}'].data[-1,...]
    # data_th4_MM = data_th4[f'{fields[1][1]}'].data[-1,...]
    # data_th16_MM = data_th16[f'{fields[1][1]}'].data[-1,...]
    # data_th64_MM = data_th64[f'{fields[1][1]}'].data[-1,...]
    #
    # data_th_list = [dyn.get_Cs_where(0.5*data_th4_LM/data_th4_MM), dyn.get_Cs_where(0.5*data_th16_LM/data_th16_MM),
    #                 dyn.get_Cs_where(0.5*data_th64_LM/data_th64_MM)]
    # data_th4_LM = None
    # data_th16_LM = None
    # data_th64_LM = None
    # data_th4_MM = None
    # data_th16_MM = None
    # data_th64_MM = None
    #
    # data_th4.close
    # data_th16.close
    # data_th64.close
    # print('Cth calculated')
    #
    #
    #
    #
    # data_th_L4 = xr.open_dataset(dir_in+f'{time}_C_th_L_{res_in[0]}.nc')
    # data_th_L16 = xr.open_dataset(dir_in+f'{time}_C_th_L_{res_in[1]}.nc')
    # data_th_L64 = xr.open_dataset(dir_in+f'{time}_C_th_L_{res_in[2]}.nc')
    #
    # data_th_L4_LM = data_th_L4[f'{fields[2][0]}'].data[-1, ...]
    # data_th_L16_LM = data_th_L16[f'{fields[2][0]}'].data[-1, ...]
    # data_th_L64_LM = data_th_L64[f'{fields[2][0]}'].data[-1, ...]
    # data_th_L4_MM = data_th_L4[f'{fields[2][1]}'].data[-1, ...]
    # data_th_L16_MM = data_th_L16[f'{fields[2][1]}'].data[-1, ...]
    # data_th_L64_MM = data_th_L64[f'{fields[2][1]}'].data[-1, ...]
    #
    # data_th_L_list = [dyn.get_Cs_where(0.5*data_th_L4_LM/data_th_L4_MM), dyn.get_Cs_where(0.5*data_th_L16_LM/data_th_L16_MM),
    #                   dyn.get_Cs_where(0.5*data_th_L64_LM/data_th_L64_MM)]
    # data_th_L4_LM = None
    # data_th_L16_LM = None
    # data_th_L64_LM = None
    # data_th_L4_MM = None
    # data_th_L16_MM = None
    # data_th_L64_MM = None
    #
    # data_th_L4.close()
    # data_th_L16.close()
    # data_th_L64.close()
    # print('Cth_L calculated')
    #
    #
    #
    # data_qt4 = xr.open_dataset(dir_in+f'{time}_C_qt_{res_in[0]}.nc')
    # data_qt16 = xr.open_dataset(dir_in+f'{time}_C_qt_{res_in[1]}.nc')
    # data_qt64 = xr.open_dataset(dir_in+f'{time}_C_qt_{res_in[2]}.nc')
    #
    # data_qt4_LM = data_qt4[f'{fields[3][0]}'].data[-1, ...]
    # data_qt16_LM = data_qt16[f'{fields[3][0]}'].data[-1, ...]
    # data_qt64_LM = data_qt64[f'{fields[3][0]}'].data[-1, ...]
    # data_qt4_MM = data_qt4[f'{fields[3][1]}'].data[-1, ...]
    # data_qt16_MM = data_qt16[f'{fields[3][1]}'].data[-1, ...]
    # data_qt64_MM = data_qt64[f'{fields[3][1]}'].data[-1, ...]
    #
    # data_qt_list = [dyn.get_Cs(0.5*data_qt4_LM/data_qt4_MM), dyn.get_Cs(0.5*data_qt16_LM/data_qt16_MM),
    #                 dyn.get_Cs(0.5*data_qt64_LM/data_qt64_MM)]
    # data_qt4_LM = None
    # data_qt16_LM = None
    # data_qt64_LM = None
    # data_qt4_MM = None
    # data_qt16_MM = None
    # data_qt64_MM = None
    #
    # data_qt4.close()
    # data_qt16.close()
    # data_qt64.close()
    # print('Cqt calculated')
    #
    # print('all C calculated')



#     data_s_list = [dyn.get_Cs_where(0.5*data_s4_LM/data_s4_MM), dyn.get_Cs_where(0.5*data_s16_LM/data_s16_MM),
#                    dyn.get_Cs_where(0.5*data_s64_LM/data_s64_MM)]

    #
    # np.save(f'{save_dir}{c_in}_case_{time}_deltas_field', data_s_list)

    data_s_list = np.load(f'{save_dir}Cs_case_{time}_deltas_field.npy')
    data_th_list = np.load(f'{save_dir}C_th_case_{time}_deltas_field.npy')
    data_th_L_list = np.load(f'{save_dir}C_th_L_case_{time}_deltas_field.npy')
    data_qt_list = np.load(f'{save_dir}C_qt_case_{time}_deltas_field.npy')


    data_cl4 = dir_cloud+f'{time}_gaussian_filter_ga00_gaussian_filter_ga00.nc'
    data_cl16 = dir_cloud+f'{time}_gaussian_filter_ga02_gaussian_filter_ga00.nc'
    data_cl64 = dir_cloud+f'{time}_gaussian_filter_ga04_gaussian_filter_ga00.nc'

    data_cl_list = [data_cl4, data_cl16, data_cl64]


    data_field_s_cloud_4D, data_field_s_env_4D, data_field_th_cloud_4D, data_field_th_env_4D, \
        data_field_th_L_cloud_4D, data_field_th_L_env_4D, data_field_qt_cloud_4D, data_field_qt_env_4D \
        = apply_masks(data_s_list[0], data_th_list[0], data_th_L_list[0], data_qt_list[0], data_cl_list[0])

    data_field_s_cloud_16D, data_field_s_env_16D, data_field_th_cloud_16D, data_field_th_env_16D, \
        data_field_th_L_cloud_16D, data_field_th_L_env_16D, data_field_qt_cloud_16D, data_field_qt_env_16D \
        = apply_masks(data_s_list[1], data_th_list[1], data_th_L_list[1], data_qt_list[1], data_cl_list[1])

    data_field_s_cloud_64D, data_field_s_env_64D, data_field_th_cloud_64D, data_field_th_env_64D, \
        data_field_th_L_cloud_64D, data_field_th_L_env_64D, data_field_qt_cloud_64D, data_field_qt_env_64D \
        = apply_masks(data_s_list[2], data_th_list[2], data_th_L_list[2], data_qt_list[2], data_cl_list[2])

    print('masks applied to C fields')

    data_s_list = None
    data_th_list = None
    data_th_L_list = None
    data_qt_list = None
    data_cl_list = None

    data_s_list_cloud = [data_field_s_cloud_4D, data_field_s_cloud_16D, data_field_s_cloud_64D]
    data_s_list_env = [data_field_s_env_4D, data_field_s_env_16D, data_field_s_env_64D]

    data_th_list_cloud = [data_field_th_cloud_4D, data_field_th_cloud_16D, data_field_th_cloud_64D]
    data_th_list_env = [data_field_th_env_4D, data_field_th_env_16D, data_field_th_env_64D]

    data_th_L_list_cloud = [data_field_th_L_cloud_4D, data_field_th_L_cloud_16D, data_field_th_L_cloud_64D]
    data_th_L_list_env = [data_field_th_L_env_4D, data_field_th_L_env_16D, data_field_th_L_env_64D]

    data_qt_list_cloud = [data_field_qt_cloud_4D, data_field_qt_cloud_16D, data_field_qt_cloud_64D]
    data_qt_list_env = [data_field_qt_env_4D, data_field_qt_env_16D, data_field_qt_env_64D]


    return data_s_list_cloud, data_s_list_env, data_th_list_cloud, data_th_list_env, \
        data_th_L_list_cloud, data_th_L_list_env, data_qt_list_cloud, data_qt_list_env





def cloud_and_env_masks(data_in, cloud_liquid_threshold=10**(-7), grid='p'):

    data_in_new = data_in
    ds_in = xr.open_dataset(data_in_new)

    if f'f(q_cloud_liquid_mass_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(q_cloud_liquid_mass_on_{grid})_r']
    elif f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r']
    elif 'q_cloud_liquid_mass' in ds_in:
        q_in = ds_in['q_cloud_liquid_mass']


    q_cloud = q_in.data[-1,...]
    print('shape of the cloud field is = ', np.shape(q_cloud))

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


def apply_masks(data_field_s, data_field_th, data_field_th_L, data_field_qt, data_cl_list):

    cloud_only_mask, env_only_mask = cloud_and_env_masks(data_cl_list)


    data_field_s_cloud = ma.masked_array(data_field_s, mask=cloud_only_mask)
    data_field_s_env = ma.masked_array(data_field_s, mask=env_only_mask)
    data_field_s=None
    print('applied mask to Cs')

    data_field_th_cloud = ma.masked_array(data_field_th, mask=cloud_only_mask)
    data_field_th_env = ma.masked_array(data_field_th, mask=env_only_mask)
    data_field_th=None
    print('applied mask to C_th')

    data_field_th_L_cloud = ma.masked_array(data_field_th_L, mask=cloud_only_mask)
    data_field_th_L_env = ma.masked_array(data_field_th_L, mask=env_only_mask)
    data_field_th_L =None
    print('applied mask to C_th_L')

    data_field_qt_cloud = ma.masked_array(data_field_qt, mask=cloud_only_mask)
    data_field_qt_env = ma.masked_array(data_field_qt, mask=env_only_mask)
    data_field_qt=None
    print('applied mask to C_qt')

    return data_field_s_cloud, data_field_s_env, data_field_th_cloud, data_field_th_env, \
        data_field_th_L_cloud, data_field_th_L_env, data_field_qt_cloud, data_field_qt_env





def plot_hist(plotdir_in, data1, data2, data3, data4, data5, region, bins_in=set_bins, what_plotting='C'):

    colours = ['tab:blue', 'tab:brown', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple',
               'tab:olive', 'tab:cyan', 'tab:gray', 'tab:pink']

    if what_plotting == 'C':
        fields_latex_in = field_latex
    else:
        fields_latex_in = field_latex_sq

    # ARM_ML_range = np.array([[100, 700], [100, 900], [100, 1000], [100, 1000]])
    # ARM_CL_range = np.array([[900, 1050], [1100, 1400], [1250, 1850], [1400, 2150]])

    # BOMEX_ML_range = np.array([100, 400])
    # BOMEX_CL_range = np.array([500, 1500])

    if region == 'ML':
        B1 = 5
        B2 = 20

        A1 = [10,10,10,10]
        A2 = [70,90,100,100]

    elif region == 'IC' or region == 'CFE':
        B1 = 25
        B2 = 75

        A1 = [90,110,125,140]
        A2 = [105,140,185,215]

    elif region == 'DA':
        B1 = 0
        B2 = -1

        A1 = [0, 0, 0, 0]
        A2 = [-1, -1, -1, -1]



    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 18), sharex='row', sharey='row')
    for i in range(4):
        for j in range(3):

            print('shape of data1  = ', np.shape(data1))

            ax[i,j].hist(data1[i][j,:,:,B1:B2].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[0],
                     weights=np.ones(ma.count(data1[i][j,:,:,B1:B2])) / ma.count(data1[i][j,:,:,B1:B2]), label='BOMEX')
            ax[i,j].hist(data2[i][j,:,:,A1[0]:A2[0]].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[1],
                     weights=np.ones(ma.count(data2[i][j,:,:,A1[0]:A2[0]])) / ma.count(data2[i][j,:,:,A1[0]:A2[0]]), label='ARM 10:30L')
            ax[i,j].hist(data3[i][j,:,:,A1[1]:A2[1]].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[2],
                     weights=np.ones(ma.count(data3[i][j,:,:,A1[1]:A2[1]])) / ma.count(data3[i][j,:,:,A1[1]:A2[1]]), label='ARM 12:30L')
            ax[i,j].hist(data4[i][j,:,:,A1[2]:A2[2]].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[3],
                     weights=np.ones(ma.count(data4[i][j,:,:,A1[2]:A2[2]])) / ma.count(data4[i][j,:,:,A1[2]:A2[2]]), label='ARM 14:30L')
            ax[i,j].hist(data5[i][j,:,:,A1[3]:A2[3]].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[4],
                     weights=np.ones(ma.count(data5[i][j,:,:,A1[3]:A2[3]])) / ma.count(data5[i][j,:,:,A1[3]:A2[3]]), label='ARM 16:30L')
            ax[i,j].set_xlabel(f"{fields_latex_in[i]}", fontsize=16)
            ax[0,j].set_title(f'{Deltas[j]}')
        ax[i,0].set_ylabel("Percentage of Occurrences", fontsize=16)


    # bottom_set, top_set = plt.ylim()
    # print('y_min = ', bottom_set, 'y_max = ', top_set)
    ax[4,0].legend(fontsize=12, loc='best')
    #plt.vlines(0, ymin=0, ymax=((1e9)), linestyles='dashed', colors='black', linewidths=0.5)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    plt.savefig(plotdir_in + f'hist_of_{what_plotting}_values_{region}.pdf',
                bbox_inches='tight')
    plt.clf()

    print(f'plotted {what_plotting}')




#dir_in, dir_cloud, c_in, time, res_in

BOMEX_s_list_cloud, BOMEX_s_list_env, BOMEX_th_list_cloud, BOMEX_th_list_env, \
        BOMEX_th_L_list_cloud, BOMEX_th_L_list_env, BOMEX_qt_list_cloud, BOMEX_qt_list_env = \
    get_data_per_delta(BOMEX_homedir, BOMEX_dir_contour, 'C', times[0], bomex_res)
BOMEX_env = [BOMEX_s_list_env, BOMEX_th_list_env, BOMEX_th_L_list_env, BOMEX_qt_list_env]
BOMEX_IC = [BOMEX_s_list_cloud, BOMEX_th_list_cloud, BOMEX_th_L_list_cloud, BOMEX_qt_list_cloud]
# np.save(f'{save_dir}BOMEX_env.npy', BOMEX_env)
# np.save(f'{save_dir}BOMEX_IC.npy', BOMEX_IC)

ARM1_s_list_cloud, ARM1_s_list_env, ARM1_th_list_cloud, ARM1_th_list_env, \
        ARM1_th_L_list_cloud, ARM1_th_L_list_env, ARM1_qt_list_cloud, ARM1_qt_list_env = \
    get_data_per_delta(ARM_homedir, ARM_dir_contour, 'C', times[1], arm_res)
ARM1_env = [ARM1_s_list_env, ARM1_th_list_env, ARM1_th_L_list_env, ARM1_qt_list_env]
ARM1_IC = [ARM1_s_list_cloud, ARM1_th_list_cloud, ARM1_th_L_list_cloud, ARM1_qt_list_cloud]
# np.save(f'{save_dir}ARM1_env.npy', ARM1_env)
# np.save(f'{save_dir}ARM1_IC.npy', ARM1_IC)

ARM2_s_list_cloud, ARM2_s_list_env, ARM2_th_list_cloud, ARM2_th_list_env, \
        ARM2_th_L_list_cloud, ARM2_th_L_list_env, ARM2_qt_list_cloud, ARM2_qt_list_env = \
    get_data_per_delta(ARM_homedir, ARM_dir_contour, 'C', times[2], arm_res)
ARM2_env = [ARM2_s_list_env, ARM2_th_list_env, ARM2_th_L_list_env, ARM2_qt_list_env]
ARM2_IC = [ARM2_s_list_cloud, ARM2_th_list_cloud, ARM2_th_L_list_cloud, ARM2_qt_list_cloud]
# np.save(f'{save_dir}ARM2_env.npy', ARM2_env)
# np.save(f'{save_dir}ARM2_IC.npy', ARM2_IC)

ARM3_s_list_cloud, ARM3_s_list_env, ARM3_th_list_cloud, ARM3_th_list_env, \
        ARM3_th_L_list_cloud, ARM3_th_L_list_env, ARM3_qt_list_cloud, ARM3_qt_list_env = \
    get_data_per_delta(ARM_homedir, ARM_dir_contour, 'C', times[3], arm_res)
ARM3_env = [ARM3_s_list_env, ARM3_th_list_env, ARM3_th_L_list_env, ARM3_qt_list_env]
ARM3_IC = [ARM3_s_list_cloud, ARM3_th_list_cloud, ARM3_th_L_list_cloud, ARM3_qt_list_cloud]
# np.save(f'{save_dir}ARM3_env.npy', ARM3_env)
# np.save(f'{save_dir}ARM3_IC.npy', ARM3_IC)

ARM4_s_list_cloud, ARM4_s_list_env, ARM4_th_list_cloud, ARM4_th_list_env, \
        ARM4_th_L_list_cloud, ARM4_th_L_list_env, ARM4_qt_list_cloud, ARM4_qt_list_env = \
    get_data_per_delta(ARM_homedir, ARM_dir_contour, 'C', times[4], arm_res)
ARM4_env = [ARM4_s_list_env, ARM4_th_list_env, ARM4_th_L_list_env, ARM4_qt_list_env]
ARM4_IC = [ARM4_s_list_cloud, ARM4_th_list_cloud, ARM4_th_L_list_cloud, ARM4_qt_list_cloud]
# np.save(f'{save_dir}ARM4_env.npy', ARM4_env)
# np.save(f'{save_dir}ARM4_IC.npy', ARM4_IC)


plot_hist(plotdir, BOMEX_env, ARM1_env, ARM2_env, ARM3_env, ARM4_env, region='ML')
plot_hist(plotdir,  BOMEX_env, ARM1_env, ARM2_env, ARM3_env, ARM4_env, region='IC')
plot_hist(plotdir, BOMEX_IC, ARM1_IC, ARM2_IC, ARM3_IC, ARM4_IC, region='CFE')