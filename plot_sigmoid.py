import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--case_in', type=str, default='ARM')
args = parser.parse_args()
case = args.case_in

mygrid = 'p'
Deltas = ['0', '1', '2', '3', '4', '5']
Delta_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$', '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_data/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/BOMEX/sigmoid/'
    times_analysed = [ '14400' ]

    zn_set = np.arange(0, 3020, 20)
    dx=20

    # todd_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
    # prof_file = todd_dir + 'BOMEX_m0020_g0800_all_14400.nc'

    z_cl_r_ind_set = [33, 77] # set
    z_cl_r_ind_calc = [22, 109]  # calc

    z_ML_r_ind = [10, 22] # 22 as calc but the profs


if case == 'ARM':
    data_path = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/diagnostics_3d_ts_'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/sigmoid/'
    times = [ '18000', '25200', '32400', '39600' ]

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    dx=25

    z_cl_r_ind_set_list = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ]



os.makedirs(plotdir, exist_ok = True)


def calc_var_mean(var_in, dir_in, time_in, layer_set, Deltas):

    var_mean_high_res = f'{var_in}_wind_mean'
    var_name = f'f({var_in}_on_{mygrid})_r'

    var_mean = np.zeros(( len(Deltas) ))

    for i in range(Deltas):
        dataset_in = dir_in + f'{time_in}_gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'

        data_set = xr.open_dataset(dataset_in)
        var_mean[i] = np.mean(data_set[var_name].data[..., layer_set])

    return var_mean

def calc_variance(var, dir, time, layer, Delta_list):

    var_name = f'f({var}_on_{mygrid})_r'

    var_mean = calc_var_mean(var, dir, time, layer, Delta_list)
    var_varience = np.zeros(( len(Deltas) ))

    for i in range(Delta_list):
        dataset_in = dir + f'{time}_gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'

        data_set = xr.open_dataset(dataset_in)
        var_varience[i] = np.mean( (data_set[var_name].data[..., layer] - var_mean[i])**2 )

    return var_varience


def plot_sigmoid(variable, data_dir, time_list, delta_list, z_level):

    col_list = ['k', 'r', 'b', 'g', 'y', 'm', 'tab:gray']

    plt.figure(figsize=(6, 6))

    for t, time_str in enumerate(time_list):

        var_sig = calc_variance(variable, data_dir, time_str, z_level, delta_list)
        plt.semilogx(Delta_labels, var_sig, col_list[t], label=f'{time_str}')

    # plt.errorbar(res[0] / z_i, w_var[0] / w_var[0], yerr=var_err[0] / w_var[0], label=str(res[0]) + 'm',
    #              color=col_list[0], ecolor='green', fmt='o', capsize=5)
    # for i in range(len(res) - 1):
    #     plt.errorbar(res[i + 1] / z_i, w_var[i + 1] / w_var[1], yerr=var_err[i + 1] / w_var[1],
    #                  label=str(res[i + 1]) + 'm',
    #                  color=col_list[i + 1], ecolor='green', fmt='o', capsize=5)

    plt.xlabel("Filter Scale", fontsize=22)
    if variable == 'w':
        plt.ylabel("$\\overline{ w'^2 }$", fontsize=16)
    # plt.ylim(ymax=1.1, ymin=0.0)
    # plt.xlim(xmax=4e3, xmin=3e0)
    plt.legend(fontsize=12, loc='best')
    #plt.xticks(np.array([0.01, 0.1, 1]), [0.01, 0.1, 1])

    plt.savefig(plotdir + f'{variable}_sigmoid.pdf')  # ("../plots/5m_w_variance_subgrid.png")