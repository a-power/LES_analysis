import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--case_in', type=str, default='ARM')
parser.add_argument('--var', type=str, default='TKE')
parser.add_argument('--log_a', default=True)

args = parser.parse_args()
case = args.case_in
set_var = args.var
set_log_axis = args.log_a

mygrid = 'p'
Deltas = ['0', '1', '2', '3', '4', '5']
Delta_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$', '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_filters/contours/BOMEX_m0020_g0800_all_'
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
    Delta_values = [2 * 20, 4 * 20, 8 * 20, 16 * 20, 32 * 20, 64 * 20]

    os.makedirs(outdir, exist_ok=True)


if case == 'ARM':
    data_path = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/diagnostics_3d_ts_'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/sigmoid/'
    bomex_info_in = '/gws/nopw/j04/paracon_rdg/users/apower/BOMEX/data/'
    times = [ '18000', '25200', '32400', '39600' ]

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    dx=25

    z_cl_r_ind_set = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ]

    Delta_values = [2 * 25, 4 * 25, 8 * 25, 16 * 25, 32 * 25, 64 * 25]
    Delta_values_BOMEX = [2 * 20, 4 * 20, 8 * 20, 16 * 20, 32 * 20, 64 * 20]

    ml_heights = np.array([820, 970, 1120, 1270])
    ml_height_bomex = 440



os.makedirs(plotdir, exist_ok = True)


def calc_var_mean(var_in, dir_in, time_in, layer_set, Deltas):

    var_mean_high_res = f'{var_in}_wind_mean'
    var_name = f'f({var_in}_on_{mygrid})_r'

    var_mean = np.zeros(( len(Deltas) ))

    for d, Del in enumerate(Deltas):
        dataset_in = dir_in + f'{time_in}_gaussian_filter_ga0{Del}_gaussian_filter_ga00.nc'

        data_set = xr.open_dataset(dataset_in)
        var_mean[d] = np.mean(data_set[var_name].data[..., layer_set])

    return var_mean

def calc_variance(var, dir, time, layer, Delta_list):

    var_name = f'f({var}_on_{mygrid})_r'

    var_mean = calc_var_mean(var, dir, time, layer, Delta_list)
    var_varience = np.zeros(( len(Deltas) ))

    for i in range(len(Delta_list)):
        dataset_in = dir + f'{time}_gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'

        data_set = xr.open_dataset(dataset_in)
        var_varience[i] = np.mean( (data_set[var_name].data[..., layer] - var_mean[i])**2 )

    return var_varience


def plot_sigmoid(variable, data_dir, time_list, delta_list, layer, z_l_r_ind_list_in, log_axis=True):

    col_list = ['k', 'r', 'b', 'g', 'y', 'm', 'tab:gray']

    plt.figure(figsize=(6, 6))

    for t, time_str in enumerate(time_list):

        clock_time_int = 05.30 + int(time_str) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        z_l_r = z_l_r_ind_list_in[t]
        z_l_mid_layer = int( (z_l_r[0] + z_l_r[1])/2 )
        print('axis index for ', layer, f'at time {time_str} is ', z_l_mid_layer)
        if variable == 'TKE':
            sig_u = calc_variance('u', data_dir, time_str, z_l_mid_layer, delta_list)
            sig_v = calc_variance('v', data_dir, time_str, z_l_mid_layer, delta_list)
            sig_w = calc_variance('w', data_dir, time_str, z_l_mid_layer, delta_list)

            var_sig = 0.5 * (sig_u + sig_v + sig_w)
        else:
            var_sig = calc_variance(variable, data_dir, time_str, z_l_mid_layer, delta_list)

        if case == 'BOMEX':
            np.save(outdir+f'BOMEX_{variable}_{layer}.npy', var_sig)

        if log_axis == True:
            plt.semilogx(Delta_values, var_sig, col_list[t], label=f'{clock_time}')
        else:
            plt.plot(Delta_values, var_sig, col_list[t], label=f'{clock_time}')

    # plt.errorbar(res[0] / z_i, w_var[0] / w_var[0], yerr=var_err[0] / w_var[0], label=str(res[0]) + 'm',
    #              color=col_list[0], ecolor='green', fmt='o', capsize=5)
    # for i in range(len(res) - 1):
    #     plt.errorbar(res[i + 1] / z_i, w_var[i + 1] / w_var[1], yerr=var_err[i + 1] / w_var[1],
    #                  label=str(res[i + 1]) + 'm',
    #                  color=col_list[i + 1], ecolor='green', fmt='o', capsize=5)

    plt.xlabel("Filter Scale", fontsize=14)
    plt.title(f'{layer}', fontsize=16)
    if variable == 'w':
        plt.ylabel("$\\overline{ w'^2 }$ $(m^2 s^{-1})$", fontsize=16)
    elif variable == 'TKE':
        plt.ylabel("TKE $(m^2 s^{-1})$", fontsize=16)
    # plt.ylim(ymax=1.1, ymin=0.0)
    # plt.xlim(xmax=4e3, xmin=3e0)
    if case == 'ARM':
        plt.legend(fontsize=12, loc='best')
    plt.xticks(Delta_values, Delta_labels)

    plt.savefig(plotdir + f'{case}_{variable}_sigmoid_{layer}_log_{set_log_axis}.pdf')  # ("../plots/5m_w_variance_subgrid.png")



def plot_all_sigmoid(variable, data_dir, extra_case_npy, time_list, delta_list, layer, z_l_r_ind_list_in):

    col_list = ['k', 'r', 'b', 'g', 'y', 'm', 'tab:gray']

    plt.figure(figsize=(6, 6))

    for t, time_str in enumerate(time_list):

        clock_time_int = 05.30 + int(time_str) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        z_l_r = z_l_r_ind_list_in[t]
        z_l_mid_layer = int( (z_l_r[0] + z_l_r[1])/2 )
        print('axis index for ', layer, f'at time {time_str} is ', z_l_mid_layer)

        if variable == 'TKE':
            sig_u = calc_variance('u', data_dir, time_str, z_l_mid_layer, delta_list)
            sig_v = calc_variance('v', data_dir, time_str, z_l_mid_layer, delta_list)
            sig_w = calc_variance('w', data_dir, time_str, z_l_mid_layer, delta_list)

            var_sig = 0.5 * (sig_u + sig_v + sig_w)
        else:
            var_sig = calc_variance(variable, data_dir, time_str, z_l_mid_layer, delta_list)

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
    if variable == 'w':
        plt.ylabel("$\\overline{ w'^2} / \overline{ w'^2_{total} }$", fontsize=16)
    elif variable == 'TKE':
        plt.ylabel("$TKE / TKE_{total}$", fontsize=16)
    # plt.ylim(ymax=1.1, ymin=0.0)
    # plt.xlim(xmax=4e3, xmin=3e0)
    plt.legend(fontsize=12, loc='best')
    # plt.xticks(Delta_values, Delta_labels)

    plt.savefig(plotdir + f'all_{variable}_sigmoids_{layer}_log_{set_log_axis}.pdf')



#
# plot_sigmoid(set_var, data_path, times, Deltas, 'Mid ML', z_ml_r_ind_list, set_log_axis)
# plot_sigmoid(set_var, data_path, times, Deltas, 'Mid CL', z_cl_r_ind_set, set_log_axis)


plot_all_sigmoid(set_var, data_path, bomex_info_in, times, Deltas, 'Mid ML', z_ml_r_ind_list)
plot_all_sigmoid(set_var, data_path, bomex_info_in, times, Deltas, 'Mid CL', z_ml_r_ind_list)
