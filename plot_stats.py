import numpy as np
import matplotlib.pyplot as plt
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
Delta_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$', '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_data/smoothed_LM_HR_fields/stats/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/stats/'
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
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/BOMEX/stats/'
    times = [ '18000', '25200', '32400', '39600' ]

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    dx=25

    z_cl_r_ind_set_list = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ]

    z_cl_r_ind_set = z_cl_r_ind_set_list[args.time_it]
    z_ml_r_ind_set = z_ml_r_ind_list[args.time_it]


os.makedirs(plotdir, exist_ok = True)

def plt_all_D_mean_sd():

    col_list = ['k', 'b', 'r', 'y'] # 'm', 'k', 'tab:gray', 'b']

    for t, time_in in enumerate(times):

        clock_time_int = 05.30 + int(time_in) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        if case == 'BOMEX':
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
        elif case == 'ARM':
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4, 12))
        fig.tight_layout(pad=0.5)

        C_domain_mean = np.zeros((3, len(Deltas) ))
        C_domain_sd = np.zeros((3, len(Deltas) ))

        for c_n, smag in enumerate(['Cs', 'C_th', 'C_qt']):
            for d, Delta_in in enumerate(Deltas):

                file_name = data_path + f'{smag}_{time_in}_delta_{str(Delta_in)}.csv'

                # header = ['param', 'partition', 'layer_range', 'C_mean', 'C_st_dev',
                #           'C_med', 'C_25', 'C_75', 'C_min', 'C_max', 'C_number']

                # rows = [C_dom, C_ML, C_CL_calc, C_CL_set, C_CS, C_IC, C_CU, C_CC,
                #         C_sq_dom, C_sq_ML, C_sq_CL_calc, C_sq_CL_set, C_sq_CS, C_sq_IC, C_sq_CU, C_sq_CC]

                # partition_name = ['domain', 'ML', 'CL_calc', 'CL_set', 'CS', 'IC', 'CU', 'CC']

                with open(file_name) as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 1: #or line_count == len(partition_name)+1: #definietly +1, have counted
                            print('Smagorinsky parameter being plotted is ', row[0])
                            C_domain_mean[c_n, d] = row[3]
                            C_domain_sd[c_n, d] = row[4]
                            break
                        line_count += 1


            ax[c_n].errorbar(Delta_labels, C_domain_mean[c_n, ...], yerr=C_domain_sd[c_n, ...],
                             color='black', ecolor='black', capsize=5)


        ax[0].set_ylabel('$C_{s}$'+f' at {clock_time}', fontsize=16)
        ax[1].set_ylabel('$C_{\\theta}$'+f' at {clock_time}', fontsize=16)
        ax[2].set_ylabel('$C_{qt}$'+f' at {clock_time}', fontsize=16)

        bottom0, top0 = ax[0].set_ylim()
        bottom1, top1 = ax[1].set_ylim()
        bottom2, top2 = ax[2].set_ylim()

        set_top = max(top0, top1, top2)
        set_bottom = bottom0

        ax[0].set_ylim(bottom = set_bottom, top = set_top)
        ax[1].set_ylim(bottom = set_bottom, top = set_top)
        ax[2].set_ylim(bottom = set_bottom, top = set_top)

        ax[0].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
        ax[1].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
        ax[2].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)

        plt.savefig(plotdir + f'C_vs_Delta_st_dev_3_ax_time_{time_in}.pdf', bbox_inches='tight')
        plt.close()




        if case == 'ARM':
            fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
            ax[c_n].errorbar(Delta_labels, C_domain_mean[c_n, ...], yerr=C_domain_sd[c_n, ...],
                             color=col_list[t], ecolor=col_list[t], capsize=5)

        ax2[0].set_ylabel('$C_{s}$', fontsize=16)
        ax2[1].set_ylabel('$C_{\\theta}$', fontsize=16)
        ax2[2].set_ylabel('$C_{qt}$', fontsize=16)

        bottom0, top0 = ax2[0].set_ylim()
        bottom1, top1 = ax2[1].set_ylim()
        bottom2, top2 = ax2[2].set_ylim()

        set_top = max(top0, top1, top2)
        set_bottom = bottom0

        ax2[0].set_ylim(bottom=set_bottom, top=set_top)
        ax2[1].set_ylim(bottom=set_bottom, top=set_top)
        ax2[2].set_ylim(bottom=set_bottom, top=set_top)

        ax2[0].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
        ax2[1].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
        ax2[2].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)

        plt.savefig(plotdir + f'C_vs_Delta_st_dev_3_ax_time_{time_in}.pdf', bbox_inches='tight')
        plt.close()