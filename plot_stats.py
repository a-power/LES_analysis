import numpy as np
import matplotlib.pyplot as plt
import mask_cloud_vs_env as clo
import numpy.ma as ma
import dynamic_functions as dyn
import xarray as xr
import os
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--case_in', type=str, default='ARM')
parser.add_argument('--time_it', type=int, default=0)

args = parser.parse_args()
case = args.case_in

mygrid = 'p'
Deltas = ['0', '1', '2', '3', '4', '5']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_data/smoothed_LM_HR_fields/stats/'
    times_analysed = [ '14400' ]

    zn_set = np.arange(0, 3020, 20)
    dx=20

    # todd_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
    # prof_file = todd_dir + 'BOMEX_m0020_g0800_all_14400.nc'

    z_cl_r_ind_set = [33, 77] # set
    z_cl_r_ind_calc = [22, 109]  # calc

    z_ML_r_ind = [10, 22] # 22 as calc but the profs


if case == 'ARM':
    data_path = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/smoothed_LM_HR_fields/stats/'
    times = [ '18000', '25200', '32400', '39600' ]

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    dx=25

    z_cl_r_ind_set_list = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ]

    z_cl_r_ind_set = z_cl_r_ind_set_list[args.time_it]
    z_ml_r_ind_set = z_ml_r_ind_list[args.time_it]

def plt_all_D_mean_sd():

    for t, time_in in enumerate(times):

        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 6))
        fig.tight_layout(pad=0.5)

        for c_n, smag in enumerate(['Cs', 'C_th', 'C_qt']):
            for d, Delta_in in enumerate(Deltas):

                file_name = data_path + f'{smag}_{time_in}_delta_{str(Delta_in)}.csv'

                # header = ['param', 'partition', 'layer_range', 'C_mean', 'C_st_dev',
                #           'C_med', 'C_25', 'C_75', 'C_min', 'C_max', 'C_number']

                # rows = [C_dom, C_ML, C_CL_calc, C_CL_set, C_CS, C_IC, C_CU, C_CC,
                #         C_sq_dom, C_sq_ML, C_sq_CL_calc, C_sq_CL_set, C_sq_CS, C_sq_IC, C_sq_CU, C_sq_CC]

                partition_name = ['domain', 'ML', 'CL_calc', 'CL_set', 'CS', 'IC', 'CU', 'CC']

                with open('employee_birthday.txt', mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 1 or line_count == len(partition_name):
                            Cs_domain_mean = row[3]
                            Cs_domain_sd = row[4]
                        line_count += 1






                    for it in range(len(Cs[:,0])):
                        ax[0].plot(Cs[it,:], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
                        ax[1].plot(Cth[it, :], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
                        ax[2].plot(Cqt[it, :], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
                    if C_sq_to_C == True:
                        if C_or_LM == 'C':
                            ax[0].set_xlabel('$C_{s}$', fontsize=16)
                            ax[1].set_xlabel('$C_{\\theta}$', fontsize=16)
                            ax[2].set_xlabel('$C_{qt}$', fontsize=16)
                        elif C_or_LM == 'LM':
                            ax[0].set_xlabel(f'$LM$', fontsize=16)
                            ax[1].set_xlabel('$HR_{\\theta}$', fontsize=16)
                            ax[2].set_xlabel('$HR_{qt}$', fontsize=16)
                        elif C_or_LM == 'MM':
                            ax[0].set_xlabel(f'$MM$', fontsize=16)
                            ax[1].set_xlabel('$RR_{\\theta}$', fontsize=16)
                            ax[2].set_xlabel('$RR_{qt}$', fontsize=16)
                        else:
                            print('not a recognised LM/MM/C')
                    else:
                        ax[0].set_xlabel('$C^2_{s}$', fontsize=16)
                        ax[1].set_xlabel('$C^2_{\\theta}$', fontsize=16)
                        ax[2].set_xlabel('$C^2_{qt}$', fontsize=16)
                    ax[0].legend(fontsize=13, loc='upper right')
                    ax[1].legend(fontsize=13, loc='upper right')
                    ax[2].legend(fontsize=13, loc='upper right')

                    if C_or_LM == 'C':
                        left0, right0 = ax[0].set_xlim()
                        left1, right1 = ax[1].set_xlim()
                        left2, right2 = ax[2].set_xlim()

                        set_right = max(right0, right1, right2)
                        set_left = left0
                    else:
                        print(np.shape(Cs))
                        x_ax_max_Cs = np.amax(Cs[:, 10:80])
                        x_ax_max_Cth = np.amax(Cth[:, 10:80])
                        x_ax_max_Cqt = np.amax(Cqt[:, 10:80])

                        x_ax_min_Cs = np.amin(Cs[:, 10:80])
                        x_ax_min_Cth = np.amin(Cth[:, 10:80])
                        x_ax_min_Cqt = np.amin(Cqt[:, 10:80])

                        set_right = max(x_ax_max_Cs, x_ax_max_Cth, x_ax_max_Cqt)
                        set_left = -1 #min(x_ax_min_Cs, x_ax_min_Cth, x_ax_min_Cqt)

                    print('for all Delta profs, min is = ', set_left, 'max is =', set_right)

                    ax[0].set_xlim(right = set_right, left = set_left)
                    ax[1].set_xlim(right = set_right, left = set_left)
                    ax[2].set_xlim(right = set_right, left = set_left)

                    ax[0].axhline(z_CL_r_m[0]/z_i, set_left, 1, color='k', linestyle='-.')
                    ax[1].axhline(z_CL_r_m[0]/z_i, set_left, 1, color='k', linestyle='-.')
                    ax[2].axhline(z_CL_r_m[0]/z_i, set_left, 1, color='k', linestyle='-.')

                    ax[0].axhline(z_CL_r_m[1]/z_i, set_left, 1, color='k', linestyle='dashed')
                    ax[1].axhline(z_CL_r_m[1]/z_i, set_left, 1, color='k', linestyle='dashed')
                    ax[2].axhline(z_CL_r_m[1]/z_i, set_left, 1, color='k', linestyle='dashed')

                    if interp==True:
                        ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
                        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}prof_scaled_interp_z.pdf', bbox_inches='tight')
                    else:
                        ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
                        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}prof_scaled_zn.pdf', bbox_inches='tight')
                    plt.close()
`