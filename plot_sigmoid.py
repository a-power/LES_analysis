import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--case_in', type=str, default='ARM')
parser.add_argument('--var', type=str, default='TKE') # 'w', 'TKE'
parser.add_argument('--var2', type=str, default=None)# 'th', 'q_total'
parser.add_argument('--log_a', default=True)

args = parser.parse_args()
case = args.case_in
set_var = args.var
set_var2 = args.var2
set_log_axis = args.log_a

mygrid = 'p'
Deltas = ['-1', '0', '1', '2', '3', '4', '5']
Delta_labels = ['$\\Delta$', '2$\\Delta$', '4$\\Delta$', '8$\\Delta$', '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_filters/contours/BOMEX_m0020_g0800_all_'
    og_unfilt = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_filters/contours/BOMEX_m0020_g0800_all_'
    plotdir = '/home/users/si818415/phd/plots/'
    outdir = '/home/users/si818415/phd/data/'

    times = [ '14400' ]

    zn_set = np.arange(0, 3020, 20)
    dx=20

    # todd_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
    # prof_file = todd_dir + 'BOMEX_m0020_g0800_all_14400.nc'

    z_cl_r_ind_set = [[33, 77]] # set
    z_cl_r_ind_calc = [[22, 109]]  # calc

    z_ml_r_ind_list = [[10, 22]] # 22 as calc but the profs
    Delta_values = [20, 2 * 20, 4 * 20, 8 * 20, 16 * 20, 32 * 20, 64 * 20]

    os.makedirs(outdir, exist_ok=True)


if case == 'ARM':
    data_path = '/work/scratch-pw3/apower/ARM/corrected_sigmas/diagnostics_3d_ts_'
    og_unfilt = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_'
    outdir = '/work/scratch-pw3/apower/ARM/corrected_sigmas/data/test/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/sigmoid/'
    bomex_info_in = '/gws/nopw/j04/paracon_rdg/users/apower/BOMEX/data/'
    times = [ '18000', '25200', '32400', '39600' ]

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    dx=25

    z_cl_r_ind_set = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [15, 75], [15, 80], [15, 85], [15, 90] ]

    Delta_values = [25, 2 * 25, 4 * 25, 8 * 25, 16 * 25, 32 * 25, 64 * 25]
    Delta_values_BOMEX = [20, 2 * 20, 4 * 20, 8 * 20, 16 * 20, 32 * 20, 64 * 20]

    ml_heights = np.array([820, 970, 1120, 1270])
    ml_height_bomex = 440

    os.makedirs(outdir, exist_ok=True)



os.makedirs(plotdir, exist_ok = True)


def calc_var_mean(var_in, dir_in, time_in, layer_set, Deltas, dir_og_unfilt = og_unfilt):

    #var_mean_high_res = f'{var_in}_wind_mean'
    var_mean = np.zeros(( len(Deltas) ))

    for d, Del in enumerate(Deltas):
        if Del == '-1':
            dataset_in = dir_og_unfilt + f'3d_ts_{time_in}.nc'
            var_name = f'{var_in}'
            if var_in == 'q_total':
                vara = 'q_vapour'
                varb = 'q_cloud_liquid_mass'
        else:
            dataset_in = dir_in + f'{time_in}_gaussian_filter_ga0{Del}.nc'
            var_name = f'f({var_in}_on_{mygrid})_r'

        data_set = xr.open_dataset(dataset_in)
        if Del == '-1' and var_in == 'q_total':
            var_mean[d] = np.mean(data_set[vara].data[..., layer_set] + data_set[varb].data[..., layer_set])
        else:
            var_mean[d] = np.mean(data_set[var_name].data[..., layer_set])

    return var_mean

def calc_variance(var, dir, time, layer, Delta_list, dir_og_unfilt = og_unfilt):

    var_mean = calc_var_mean(var, dir, time, layer, Delta_list)
    var_variance = np.zeros(( len(Deltas) ))

    for d, Del in enumerate(Deltas):
        if Del == '-1':
            dataset_in = dir_og_unfilt + f'3d_ts_{time}.nc'
            var_name = f'{var}'
        else:
            dataset_in = dir + f'{time}_gaussian_filter_ga0{Del}.nc'
            var_name = f'f({var}_on_{mygrid})_r'

        data_set = xr.open_dataset(dataset_in)
        var_variance[d] = np.mean( (data_set[var_name].data[..., layer] - var_mean[d])**2 )

    return var_variance


def calc_covariance(var1, var2, dir, time, layer, Delta_list, var3=None, dir_og_unfilt = og_unfilt):


    var_mean1 = calc_var_mean(var1, dir, time, layer, Delta_list)
    var_mean2 = calc_var_mean(var2, dir, time, layer, Delta_list)
    if var3 != None:
        var_mean3 = calc_var_mean(var3, dir, time, layer, Delta_list)

    var_covariance = np.zeros(( len(Deltas) ))

    for d, Del in enumerate(Deltas):
        if Del == '-1':
            dataset_in = dir_og_unfilt + f'3d_ts_{time}.nc'
            var_name1 = f'{var1}'
            var_name2 = f'{var2}'
            var_name3 = f'{var3}'
            if var2 == 'q_total':
                vara = 'q_vapour'
                varb = 'q_cloud_liquid_mass'
        else:
            dataset_in = dir + f'{time}_gaussian_filter_ga0{Del}.nc'
            var_name1 = f'f({var1}_on_{mygrid})_r'
            var_name2 = f'f({var2}_on_{mygrid})_r'
            var_name3 = f'f({var3}_on_{mygrid})_r'

        data_set = xr.open_dataset(dataset_in)
        if Del == '-1' and var2 == 'q_total':
            var_covariance[d] = np.mean( (data_set[var_name1].data[..., layer] - var_mean1[d]) * \
                             ( (data_set[vara].data[..., layer] + data_set[varb].data[..., layer]) - var_mean2[d]) )
        elif var3 != None:
            var_covariance[d] = np.mean( 0.5 * ( (data_set[var_name1].data[..., layer] - var_mean1[d] )**2 + \
                                        (data_set[var_name2].data[..., layer] - var_mean2[d])**2 + \
                                        (data_set[var_name3].data[..., layer] - var_mean3[d]) ** 2 ) )
        else:
            var_covariance[d] = np.mean( (data_set[var_name1].data[..., layer] - var_mean1[d]) * \
                                     (data_set[var_name2].data[..., layer] - var_mean2[d]))

    return var_covariance


def plot_sigmoid(variable, data_dir, time_list, delta_list, layer, z_l_r_ind_list_in, var2=None, log_axis=True, \
                 dir_unfilt = og_unfilt):

    col_list = ['k', 'r', 'b', 'g', 'y', 'm', 'tab:gray']

    plt.figure(figsize=(6, 6))

    for t, time_str in enumerate(time_list):

        sg_dataset_path = dir_unfilt + f'ts_{time_str}.nc'
        sg_dataset = xr.open_dataset(sg_dataset_path)

        clock_time_int = 05.30 + int(time_str) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        z_l_r = z_l_r_ind_list_in[t]
        z_l_mid_layer = int( (z_l_r[0] + z_l_r[1])/2 )
        print('axis index for ', layer, f'at time {time_str} is ', z_l_mid_layer)
        if variable == 'TKE':
            sg_var = np.mean(sg_dataset['tkesg_mean'].data[..., z_l_mid_layer])
            if os.path.exists(outdir + f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy'):
                var_sig = np.load(outdir+f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy')
            else:
                # var_u = calc_variance('u', data_dir, time_str, z_l_mid_layer, delta_list)
                # var_v = calc_variance('v', data_dir, time_str, z_l_mid_layer, delta_list)
                # var_w = calc_variance('w', data_dir, time_str, z_l_mid_layer, delta_list)
                # var_sig = 0.5 * (var_u + var_v + var_w)
                # var_u = None
                # var_v = None
                # var_w = None
                var_sig = calc_covariance('u', 'v', data_dir, time_str, z_l_mid_layer, delta_list, 'w')
                np.save(outdir+f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy', var_sig)

        else:
            if var2 != None:

                if os.path.exists(outdir + f'sigmoid_{case}_{variable}_{var2}_{layer}_{time_str}.npy'):
                    var_sig = np.load(outdir+f'sigmoid_{case}_{variable}_{var2}_{layer}_{time_str}.npy')
                else:
                    var_sig = calc_covariance(variable, var2, data_dir, time_str, z_l_mid_layer, delta_list)
                    np.save(outdir+f'sigmoid_{case}_{variable}_{var2}_{layer}_{time_str}.npy', var_sig)
                if var2 == 'q_total':
                    var_sig = var_sig*1000 # kg to g
            else:
                if variable == 'w':
                    sg_var = np.mean(sg_dataset[f'{variable}sg_mean'].data[..., z_l_mid_layer])
                if os.path.exists(outdir + f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy'):
                    var_sig = np.load(outdir+f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy')
                else:
                    var_sig = calc_variance(variable, data_dir, time_str, z_l_mid_layer, delta_list)
                    np.save(outdir+f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy', var_sig)

        if log_axis == True:
            plt.semilogx(Delta_values, var_sig, col_list[t], label=f'{clock_time}')
        else:
            plt.plot(Delta_values, var_sig, col_list[t], label=f'{clock_time}')
        left, right = plt.xlim()

        if var2 == None:
            total_var = sg_var + var_sig[0]
            plt.hlines(total_var, xmin=left, xmax=right, colors=col_list[t], linestyles='--')

    # plt.errorbar(res[0] / z_i, w_var[0] / w_var[0], yerr=var_err[0] / w_var[0], label=str(res[0]) + 'm',
    #              color=col_list[0], ecolor='green', fmt='o', capsize=5)
    # for i in range(len(res) - 1):
    #     plt.errorbar(res[i + 1] / z_i, w_var[i + 1] / w_var[1], yerr=var_err[i + 1] / w_var[1],
    #                  label=str(res[i + 1]) + 'm',
    #                  color=col_list[i + 1], ecolor='green', fmt='o', capsize=5)

    plt.xlabel("Filter Scale", fontsize=14)
    plt.title(f'{layer}', fontsize=16)
    if var2 == None:
        if variable == 'w':
            plt.ylabel("$\\overline{ w'^2 }$ $(m^2 s^{-2})$", fontsize=16)
        elif variable == 'TKE':
            plt.ylabel("RKE $(m^2 s^{-2})$", fontsize=16)
    else:
        if variable == 'w' and var2 == 'th':
            plt.ylabel("$\\overline{ w' \\theta' }$ $(K m s^{-1})$", fontsize=16)
        elif variable == 'w' and var2 == 'q_total':
            plt.ylabel("$\\overline{ w' q_t' }$ $(g kg^{-1} m s^{-1})$", fontsize=16)

    plt.ylim(ymin=0.0)
    plt.xlim(xmin=Delta_values[0]-1, xmax=Delta_values[-1]+10)
    if case == 'ARM':
        plt.legend(fontsize=12, loc='best')
    plt.xticks(Delta_values, Delta_labels)

    if var2 != None:
        plt.savefig(plotdir + f'{case}_{variable}_{var2}_sigmoid_{layer}_log_{set_log_axis}.pdf', bbox_inches='tight')
    else:
        plt.savefig(plotdir + f'{case}_{variable}_sigmoid_{layer}_log_{set_log_axis}.pdf', bbox_inches='tight')




def plot_all_sigmoid(variable, data_dir, extra_case_npy, time_list, delta_list, layer, z_l_r_ind_list_in, var2=None):

    col_list = ['k', 'r', 'b', 'g', 'y', 'm', 'tab:gray']

    plt.figure(figsize=(6, 6))

    for t, time_str in enumerate(time_list):

        clock_time_int = 05.30 + int(time_str) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        z_l_r = z_l_r_ind_list_in[t]
        z_l_mid_layer = int( (z_l_r[0] + z_l_r[1])/2 )
        print('axis index for ', layer, f'at time {time_str} is ', z_l_mid_layer)

        if var2 != None:
            var_sig = np.load(outdir+f'sigmoid_{case}_{variable}_{var2}_{layer}_{time_str}.npy')
        else:
            var_sig = np.load(outdir+f'sigmoid_{case}_{variable}_{layer}_{time_str}.npy')
        plt.semilogx(Delta_values/ml_heights[t], var_sig/float(var_sig[0]), col_list[t], label=f'ARM at {clock_time}')

    e_case = np.load(extra_case_npy+f'BOMEX_{variable}_{layer}.npy')
    plt.semilogx(np.array(Delta_values_BOMEX)/float(ml_height_bomex), e_case/float(e_case[0]), col_list[t+1], label=f'BOMEX')

    # plt.errorbar(res[0] / z_i, w_var[0] / w_var[0], yerr=var_err[0] / w_var[0], label=str(res[0]) + 'm',
    #              color=col_list[0], ecolor='green', fmt='o', capsize=5)
    # for i in range(len(res) - 1):
    #     plt.errorbar(res[i + 1] / z_i, w_var[i + 1] / w_var[1], yerr=var_err[i + 1] / w_var[1],
    #                  label=str(res[i + 1]) + 'm',
    #                  color=col_list[i + 1], ecolor='green', fmt='o', capsize=5)

    plt.xlabel("$\\widehat{\\bar{\\Delta}} / z_{ML}$", fontsize=14)
    plt.title(f'{layer}', fontsize=16)
    if var2 == None:
        if variable == 'w':
            plt.ylabel("$\\overline{ w'^2 }$ $(m^2 s^{-2})$", fontsize=16)
        elif variable == 'TKE':
            plt.ylabel("TKE $(m^2 s^{-2})$", fontsize=16)
    else:
        if variable == 'w' and var2 == 'th':
            plt.ylabel("$\\overline{ w' \\theta' }$ $(K m s^{-1})$", fontsize=16)
        elif variable == 'w' and var2 == 'q_total':
            plt.ylabel("$\\overline{ w' \\q_t' }$ $(K m s^{-1})$", fontsize=16)
    # plt.ylim(ymax=1.1, ymin=0.0)
    # plt.xlim(xmax=4e3, xmin=3e0)
    plt.legend(fontsize=12, loc='best')
    # plt.xticks(Delta_values, Delta_labels)

    if var2 != None:
        plt.savefig(plotdir + f'all_{variable}_{var2}_sigmoids_{layer}_log_{set_log_axis}.pdf', bbox_inches='tight')
    else:
        plt.savefig(plotdir + f'all_{variable}_sigmoids_{layer}_log_{set_log_axis}.pdf', bbox_inches='tight')



#
plot_sigmoid(set_var, data_path, times, Deltas, 'Mid ML', z_ml_r_ind_list, set_var2, set_log_axis)
plot_sigmoid(set_var, data_path, times, Deltas, 'Mid CL', z_cl_r_ind_set, set_var2, set_log_axis)


# plot_all_sigmoid(set_var, data_path, bomex_info_in, times, Deltas, 'Mid ML', z_ml_r_ind_list, set_var2)
# plot_all_sigmoid(set_var, data_path, bomex_info_in, times, Deltas, 'Mid CL', z_ml_r_ind_list, set_var2)
