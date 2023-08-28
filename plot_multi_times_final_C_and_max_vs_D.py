
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn
import argparse

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

parser = argparse.ArgumentParser()
#parser.add_argument('--times', type=str, default='28800')
parser.add_argument('--case_in', type=str, default='ARM')
args = parser.parse_args()

case = args.case_in


beta=True
what_plotting='_0' # '_beta'
C_or_LM = 'C' # 'C', 'LM', or 'MM'. C_sq_to_C == True for LM and MM

x_lim_list = [0.355, 0.355, 0.355, 0.355, 0.255, 0.07]

if case == 'ARM':

    homedir = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/smoothed_LM_HR_fields/C_profs/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/C_beta_profiles/'
    profiles_dir = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_ts_'


    zn_set = np.arange(0, 4410, 10)
    z_set = np.arange(-5, 4405, 10)
    #z_ML = 10 #z_ML_calc
    z_ML_bottom = 20

    z_cl_r_t_list = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_t_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ] #z_ml_range_calc

    set_time = ['18000', '25200', '32400', '39600']

    th_name = 'th_v'

elif case == 'BOMEX':

    beta=True
    what_plotting='_0'
    C_or_LM = 'C' # 'C', 'LM', or 'MM'. C_sq_to_C == True for LM and MM

    if beta == True:
        homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/smoothed_LM_HR_fields/C_profs/'
        plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/scale_dep_plots/C_beta_profiles/fitting_relations/'
    else:
        homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/C_profs_cloud_1e-7/'
        plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/profiles_cloud_1e-7/diff_C_calc/'

    file_name = 'BOMEX_m0020_g0800_all_14400_gaussian_filter_LM_'


    zn_set = np.arange(0, 3020, 20)
    z_set = np.arange(-10, 3010, 20)
    z_ML = 490 # the w'th' 1D MONC prof suggests 440m

    z_cl_r_Bomex = [49, 73]
    z_ml_r_Bomex = [10, 22]

    set_time = ['14400']

    th_name = 'th'

else:
    print('need to def case')

os.makedirs(plotdir, exist_ok = True)


if C_or_LM == 'C':
    C_or_LM_profs = [['LM', 'HR_th', 'HR_qt'],
                     ['MM', 'RR_th', 'RR_qt']]
elif C_or_LM == 'LM':
    C_or_LM_profs = ['LM', 'HR_th', 'HR_qt']
elif C_or_LM == 'MM':
    C_or_LM_profs = ['MM', 'RR_th', 'RR_qt']
else:
    print("C_or_LM must equal 'C', 'LM', or 'MM', case sensitive with quote marks, not", C_or_LM,
          "as is curtrently being specified")


os.makedirs(plotdir, exist_ok = True)


#######################################################################################################################


def calc_z_ML_and_CL(file_path, time_stamp=-1):

    prof_data = xr.open_dataset(file_path)

    wth_prof = prof_data['wtheta_cn_mean'].data[time_stamp, ...]
    wth_prof_list = wth_prof.tolist()
    z_ML_out = wth_prof_list.index(np.min(wth_prof))

    z_cloud = prof_data['total_cloud_fraction'].data[time_stamp, ...]
    z_cloud_where = np.where(z_cloud > 1e-7)
    z_ind = np.arange(0, len(z_cloud))
    z_cloud_ind = z_ind[z_cloud_where]
    #print('z_cloud_ind =', z_cloud_ind)
    z_min_CL = np.min(z_cloud_ind)
    z_max_CL = np.max(z_cloud_ind)
    z_CL = [ z_min_CL, z_max_CL ]

    zn_out = prof_data['zn'].data[...] # timeless parameter?

    return z_ML_out, z_CL, zn_out


def interp_z(var_in, z_from=z_set, z_to=zn_set):
    interp_var = np.zeros_like(var_in)
    for n in range(len(var_in[:,0])):
        for k in range(len(z_from)-1):
            interp_var[n,k] = var_in[n,k] + (z_to[k] - z_from[k])*( (var_in[n,k+1] - var_in[n,k]) / (z_from[k+1] - z_from[k]) )
    return interp_var


def plot_C_all_Deltas(Cs, Cth, Cqt, z, z_i, z_CL_r_index, labels_in, interp=False, C_sq_to_C = False, time_in='14400'):

    clock_time_int = 05.30 + int(time_in)/(60*60)
    clock_time = str(clock_time_int)+'0L'

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    if interp==True:
        Cs = interp_z(Cs)
        Cth = interp_z(Cth)
        Cqt = interp_z(Cqt)
    if C_sq_to_C == True:
        if C_or_LM == 'C':
            Cs = dyn.get_Cs(Cs)
            Cth = dyn.get_Cs(Cth)
            Cqt = dyn.get_Cs(Cqt)
        name='_'
    else:
        if C_or_LM == 'C':
            name='_sq_'
        else:
            name = '_'

    fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(4,18))

    fig.tight_layout(pad=0.5)

    for it in range(len(Cs[:,0])):
        ax[0].plot(Cs[it,:], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
        ax[1].plot(Cth[it, :], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
        ax[2].plot(Cqt[it, :], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
    if C_sq_to_C == True:
        if C_or_LM == 'C':
            ax[0].set_xlabel('$C_{s}$ at time ' + clock_time, fontsize=16)
            ax[1].set_xlabel('$C_{\\theta}$ at time ' + clock_time, fontsize=16)
            ax[2].set_xlabel('$C_{qt}$ at time ' + clock_time, fontsize=16)
        elif C_or_LM == 'LM':
            ax[0].set_xlabel(f'$LM$ at time {clock_time}', fontsize=16)
            ax[1].set_xlabel('$HR_{\\theta}$ at time ' + clock_time, fontsize=16)
            ax[2].set_xlabel('$HR_{qt}$ at time ' + clock_time, fontsize=16)
        elif C_or_LM == 'MM':
            ax[0].set_xlabel(f'$MM$ at time {clock_time}', fontsize=16)
            ax[1].set_xlabel('$RR_{\\theta}$ at time ' + clock_time, fontsize=16)
            ax[2].set_xlabel('$RR_{qt}$ at time ' + clock_time, fontsize=16)
        else:
            print('not a recognised LM/MM/C')
    else:
        ax[0].set_xlabel('$C^2_{s}$ at time ' + clock_time, fontsize=16)
        ax[1].set_xlabel('$C^2_{\\theta}$ at time ' + clock_time, fontsize=16)
        ax[2].set_xlabel('$C^2_{qt}$ at time ' + clock_time, fontsize=16)

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
        print('np.shape(Cs) =', np.shape(Cs))
        x_ax_max_Cs = np.amax(Cs[:, 10:80])
        x_ax_max_Cth = np.amax(Cth[:, 10:80])
        x_ax_max_Cqt = np.amax(Cqt[:, 10:80])

        x_ax_min_Cs = np.amin(Cs[:, 10:80])
        x_ax_min_Cth = np.amin(Cth[:, 10:80])
        x_ax_min_Cqt = np.amin(Cqt[:, 10:80])

        set_right = max(x_ax_max_Cs, x_ax_max_Cth, x_ax_max_Cqt)
        set_left = -1 #min(x_ax_min_Cs, x_ax_min_Cth, x_ax_min_Cqt)

    ax[0].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')
    ax[1].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')
    ax[2].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')

    ax[0].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')
    ax[1].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')
    ax[2].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')

    print('for all Delta profs, min is = ', set_left, 'max is =', set_right)

    ax[0].set_xlim(right = set_right, left = set_left)
    ax[1].set_xlim(right = set_right, left = set_left)
    ax[2].set_xlim(right = set_right, left = set_left)

    plt.tight_layout()

    if interp==True:
        ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
        ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
        ax[2].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)


        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}_time{time_in}_prof_scaled_interp_z.png',
                    bbox_inches='tight')
        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}_time{time_in}_prof_scaled_interp_z.pdf',
                    bbox_inches='tight')
    else:
        ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
        ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
        ax[2].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}_time{time_in}_prof_scaled_zn.png',
                    bbox_inches='tight')
        plt.savefig(plotdir + f'{C_or_LM}{what_plotting}{name}_time{time_in}_prof_scaled_zn.pdf',
                    bbox_inches='tight')
    plt.close()


def plot_Pr_all_Deltas(Cs, Cth, Cqt, z, z_i, z_CL_r_index, labels_in, time_in, interp=False):

    clock_time_int = 05.30 + int(time_in) / (60 * 60)
    clock_time = str(clock_time_int) + '0L'

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']
    # NOTE youre feeding in C^2 not C

    if interp == True:
        Cs = interp_z(Cs)
        Cth = interp_z(Cth)
        Cqt = interp_z(Cqt)

    if case == 'BOMEX':
        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(9, 6))
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharey=False, figsize=(5, 12))

    for it in range(len(Cs[:, 0])):
        ax[0].plot(Cs[it, :] / Cth[it, :], z / z_i, color=colours[it],
                   label='$\\widehat{\\bar{\\Delta}} = $' + labels_in[it])
        ax[1].plot(Cs[it, :] / Cqt[it, :], z / z_i, color=colours[it],
                   label='$\\widehat{\\bar{\\Delta}} = $' + labels_in[it])

        bottom0, top0 = ax[0].set_ylim()
        bottom1, top1 = ax[1].set_ylim()
        set_bottom = min(bottom0, bottom1)
        set_top = max(top0, top1)
        ax[0].set_ylim(set_bottom, set_top)
        ax[1].set_ylim(set_bottom, set_top)

        set_left = -0.5
        set_right = 3.5

        ax[0].set_xlim(set_left, set_right)
        ax[1].set_xlim(set_left, set_right)

        ax[0].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')
        ax[1].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')

        ax[0].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')
        ax[1].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')


        ax[0].axvline(0.7, set_bottom, set_top, color='k', linestyle='dashed')
        ax[1].axvline(0.7, set_bottom, set_top, color='k', linestyle='dashed')

        ax[0].set_xlabel('$Pr$ at time ' + clock_time, fontsize=16)
        ax[1].set_xlabel('$Sc_{qt}$ at time ' + clock_time, fontsize=16)

        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')

        ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)
        ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)


        # left0, right0 = ax[0].set_xlim()
        # left1, right1 = ax[1].set_xlim()
        # left2, right2 = ax[2].set_xlim()
        #
        # set_right = max(right0, right1, right2)
        # set_left = min
        #
        #
        # ax[0].set_xlim(right = set_right, left = set_left)
        # ax[1].set_xlim(right = set_right, left = set_left)
        # ax[2].set_xlim(right = set_right, left = set_left)

    fig.tight_layout(pad=0.5)

    plt.savefig(plotdir + f'Pr_prof_{time_in}.pdf', bbox_inches='tight')
    plt.close()






def plot_condit_C_each_Deltas(Cs_in, Cth_in, Cqt_in, z, z_i, z_CL_r_index, deltas, delta_label, interp=False, C_sq_to_C = True,
                      labels_in = ['total', 'cloud-free', 'in-cloud', 'cloud updraft', 'cloud core'],
                              time_in='`14400', set_x_lim_list=[0.355, 0.355, 0.355, 0.355, 0.255, 0.07], Pr_in=True):


    clock_time_int = 05.30 + int(time_in)/(60*60)
    clock_time = str(clock_time_int)+'0L'

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    print('np.shape(Cs) = ', np.shape(Cs_in)) # C shape is [conditionals, Delta, z]

    Cs_temp = np.zeros_like(Cs_in)
    Cth_temp = np.zeros_like(Cs_in)
    Cqt_temp = np.zeros_like(Cs_in)
    Cs = np.zeros_like(Cs_in)
    Cth = np.zeros_like(Cs_in)
    Cqt = np.zeros_like(Cs_in)

    if Pr_in == True:
        Pr = Cs_in/Cth_in # not C here actually denotes C^2
        Sc = Cs_in/Cqt_in

    print('np.shape(Cs_in)[1] = ', np.shape(Cs_in)[1])
    for it in range(np.shape(Cs_in)[1]): #loop over Deltas
        print('it = ', it)

        if interp==True:
            Cs_temp[:,it, :] = interp_z(Cs_in[:,it, :])
            Cth_temp[:,it, :] = interp_z(Cth_in[:,it, :])
            Cqt_temp[:,it, :] = interp_z(Cqt_in[:,it, :])
        else:
            Cs_temp[:,it, :] = Cs_in[:,it, :].copy()
            Cth_temp[:,it, :] = Cth_in[:,it, :].copy()
            Cqt_temp[:,it, :] = Cqt_in[:,it, :].copy()

        if C_sq_to_C == True:
            if C_or_LM == 'C':
                Cs[:,it, :] = dyn.get_Cs(Cs_temp[:,it, :])
                Cth[:,it, :] = dyn.get_Cs(Cth_temp[:,it, :])
                Cqt[:,it, :] = dyn.get_Cs(Cqt_temp[:,it, :])
            else:
                Cs[:, it, :] = Cs_temp[:, it, :]
                Cth[:, it, :] = Cth_temp[:, it, :]
                Cqt[:, it, :] = Cqt_temp[:, it, :]
            name='_'
        else:
            name = '_sq_'
            Cs = Cs_temp.copy()
            Cth = Cth_temp.copy()
            Cqt = Cqt_temp.copy()

        fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(4,18))
        fig.tight_layout(pad=0.5)

        for nt in range(np.shape(Cs)[0]):

            set_x_lim = set_x_lim_list[nt]

            ax[0].plot(Cs[nt, it, :], z/z_i, color=colours[nt], label=labels_in[nt])
            ax[1].plot(Cth[nt, it, :], z/z_i, color=colours[nt], label=labels_in[nt])
            ax[2].plot(Cqt[nt, it, :], z/z_i, color=colours[nt], label=labels_in[nt])
        if C_sq_to_C == True:
            if C_or_LM == 'C':
                ax[0].set_xlabel('$C_{s}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it]  + ' at time ' + clock_time, fontsize=16)
                ax[1].set_xlabel('$C_{\\theta}$ for $\\widehat{\\bar{\\Delta}} = $' + delta_label[it]  + ' at time '  + clock_time,
                                 fontsize=16)
                ax[2].set_xlabel('$C_{qt}$ for $\\widehat{\\bar{\\Delta}} = $' + delta_label[it]  + ' at time ' + clock_time,
                                 fontsize=16)
            else:
                ax[0].set_xlabel(f'${C_or_LM}$'+' for $\\widehat{\\bar{\\Delta}} = $' + delta_label[it] + ' at time ' +clock_time,
                                 fontsize=16)
                ax[1].set_xlabel(f'${C_or_LM}$'+'$_{\\theta}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it] + ' at time ' +clock_time,
                                 fontsize=16)
                ax[2].set_xlabel(f'${C_or_LM}$'+'$_{qt}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it] + ' at time ' +clock_time,
                                 fontsize=16)
        else:
            ax[0].set_xlabel('$C^2_{s}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it] + ' at time ' + clock_time, fontsize=16)
            ax[1].set_xlabel('$C^2_{\\theta}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it] + ' at time ' + clock_time, fontsize=16)
            ax[2].set_xlabel('$C^2_{qt}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it] + ' at time ' + clock_time, fontsize=16)

        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')
        ax[2].legend(fontsize=13, loc='upper right')

        if C_or_LM == 'C':
            left0, right0 = ax[0].set_xlim()
            left1, right1 = ax[1].set_xlim()
            left2, right2 = ax[2].set_xlim()

            set_right = set_x_lim #max(right0, right1, right2)
            set_left = left0
        else:
            print('np.shape(Cs) = ', np.shape(Cs))
            x_ax_max_Cs = np.nanmax(Cs[:, it, 10:80])
            x_ax_max_Cth = np.nanmax(Cth[:, it, 10:80])
            x_ax_max_Cqt = np.nanmax(Cqt[:, it, 10:80])

            x_ax_min_Cs = np.nanmin(Cs[:, it, 10:80])
            x_ax_min_Cth = np.nanmin(Cth[:, it, 10:80])
            x_ax_min_Cqt = np.nanmin(Cqt[:, it, 10:80])

            set_right = max(x_ax_max_Cs, x_ax_max_Cth, x_ax_max_Cqt)
            set_left = -1 #min(x_ax_min_Cs, x_ax_min_Cth, x_ax_min_Cqt)

            print('For it (Delta)  =', it, 'min = ', set_left, 'max = ', set_right)

        print('for condit profs, min is = ', set_left, 'max is =', set_right)

        ax[0].set_xlim(right=set_right, left=set_left)
        ax[1].set_xlim(right=set_right, left=set_left)
        ax[2].set_xlim(right=set_right, left=set_left)

        ax[0].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')
        ax[1].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')
        ax[2].axhline(z_CL_r_index[0], set_left, set_right, color='k', linestyle='-.')

        ax[0].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')
        ax[1].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')
        ax[2].axhline(z_CL_r_index[1], set_left, set_right, color='k', linestyle='dashed')

        print('deltas[it] =', deltas[it])

        plt.tight_layout()

        if interp==True:
            ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
            ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
            ax[2].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)

            plt.savefig(plotdir + f'{C_or_LM}{name}condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled_interp_z.png',
                        bbox_inches='tight')
            plt.savefig(plotdir + f'{C_or_LM}{name}condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled_interp_z.pdf',
                        bbox_inches='tight')
        else:
            ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
            ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)
            ax[2].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = "+ str(z_i) + "m)", fontsize=16)

            plt.savefig(plotdir + f'{C_or_LM}{name}condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled_zn.png',
                        bbox_inches='tight')
            plt.savefig(plotdir + f'{C_or_LM}{name}condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled_zn.pdf',
                        bbox_inches='tight')
        plt.close()

        if Pr_in == True:

            fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(4, 11))
            fig.tight_layout(pad=0.5)

            for nt in range(np.shape(Pr)[0]):

                ax[0].plot(Pr[nt, it, :], z / z_i, color=colours[nt], label=labels_in[nt])
                ax[1].plot(Sc[nt, it, :], z / z_i, color=colours[nt], label=labels_in[nt])

            ax[0].set_xlabel('$Pr$ for $\\widehat{\\bar{\\Delta}} = $' + delta_label[it] + ' at time ' + clock_time,
                             fontsize=16)
            ax[1].set_xlabel('$Sc_{qt}$ for $\\widehat{\\bar{\\Delta}} = $' + delta_label[it] + ' at time ' + clock_time,
                             fontsize=16)

            ax[0].legend(fontsize=13, loc='upper right')
            ax[1].legend(fontsize=13, loc='upper right')

            # left0, right0 = ax[0].set_xlim()
            # left1, right1 = ax[1].set_xlim()
            #
            # set_right = max(right0, right1)
            # set_left = min(left0, left1)

            # ax[0].set_xlim(right=set_right, left=set_left)
            # ax[1].set_xlim(right=set_right, left=set_left)

            ax[0].set_xlim(-0.5, 3.5)
            ax[1].set_xlim(-0.5, 3.5)

            ax[0].axhline(z_CL_r_index[0], -0.5, 3.5, color='k', linestyle='-.')
            ax[1].axhline(z_CL_r_index[0], -0.5, 3.5, color='k', linestyle='-.')

            ax[0].axhline(z_CL_r_index[1], -0.5, 3.5, color='k', linestyle='dashed')
            ax[1].axhline(z_CL_r_index[1], -0.5, 3.5, color='k', linestyle='dashed')

            ax[0].set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
            ax[1].set_xticks([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

            bottom0, top0 = ax[0].set_ylim()
            bottom1, top1 = ax[1].set_ylim()
            set_bottom = min(bottom0, bottom1)
            set_top = max(top0, top1)

            ax[0].axvline(0.7, set_bottom, set_top, color='k', linestyle='dashed')
            ax[1].axvline(0.7, set_bottom, set_top, color='k', linestyle='dashed')

            ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(z_i) + "m)", fontsize=16)
            ax[1].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(z_i) + "m)", fontsize=16)

            fig.tight_layout(pad=0.5)

            plt.savefig(plotdir + f'Pr_condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled.png',
                        bbox_inches='tight')
            plt.savefig(plotdir + f'Pr_condit_prof_D={deltas[it]}{what_plotting}_time{time_in}_scaled.pdf',
                        bbox_inches='tight')
            plt.close()


def cal_max_Cs(C_list, z_ml_r, z_cl_r):

    print('when calc the max values, shape of C list is ', np.shape(C_list))

    max_C = np.zeros((np.shape(C_list)[0]+1, np.shape(C_list)[1]))

    for i in range(np.shape(C_list)[0]):
        for nD in range(np.shape(C_list)[1]):
            if i == 0:
                max_C[i, nD] = np.max(C_list[i, nD, z_ml_r[0]:z_ml_r[1]])
                max_C[i+1, nD] = np.max(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])
            else:
                max_C[i+1, nD] = np.max(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])

    print('shape of max C is ', np.shape(max_C))
    return max_C

def cal_mean_Cs(C_list, z_ml_r, z_cl_r):

    print('when calc the mean values, shape of C list is ', np.shape(C_list))

    mean_C = np.zeros((np.shape(C_list)[0]+1, np.shape(C_list)[1]))

    for i in range(np.shape(C_list)[0]):
        for nD in range(np.shape(C_list)[1]):
            if i == 0:
                mean_C[i, nD] = np.mean(C_list[i, nD, z_ml_r[0]:z_ml_r[1]])
                mean_C[i+1, nD] = np.mean(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])
            else:
                mean_C[i+1, nD] = np.mean(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])

    print('shape of mean C is ', np.shape(mean_C))
    return mean_C

def get_max_l_from_C(max_C_cond, deltas_num, grid_spacing):
    Delta_res = deltas_num*grid_spacing
    max_l_cond = np.zeros_like(max_C_cond)
    for it in range(np.shape(max_C_cond)[1]):
        max_l_cond[:,it] = max_C_cond[:,it] * Delta_res[it]
    print('shape of max l is ', np.shape(max_l_cond))
    return max_l_cond

def plot_max_C_l_vs_Delta(Cs_max_in, Cth_max_in, Cqt_max_in, Delta, y_ax, max_mean='mean', time_in = '14400'):


    clock_time_int = 05.30 + int(time_in)/(60*60)
    clock_time = str(clock_time_int)+'0L'

    my_lines = ['solid', 'solid', 'dotted', 'dashed', 'dashed', 'dashed']
    #labels = ['ML domain', 'CL domain', 'CL: clear sky', 'in-cloud', 'cloudy updraft', 'cloud core']
    labels = ['ML', 'CL', 'CS', 'IC', 'CU', 'CC']

    colours = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    if y_ax == 'C':
        y_labels = ['$C_{s}$', '$C_{\\theta}$', '$C_{qt}$']
    elif y_ax == 'l':
        y_labels = ['$l_{s}$ (m)', '$l_{\\theta}$ (m)', '$l_{qt}$ (m)']
    elif y_ax == 'Pr':
        y_labels = ['$Pr$', '$Sc_{qt}$']
    else:
        print('y_ax input not recognised')

    if y_ax == 'Pr':
        fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(4, 8))
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(4, 12))
    fig.tight_layout(pad=0.5)

    for it in range(np.shape(Cs_max_in)[0]):
        ax[0].plot(Delta, Cs_max_in[it,...], color=colours[it], linestyle=my_lines[it], label=labels[it])
        ax[1].plot(Delta, Cth_max_in[it,...], color=colours[it], linestyle=my_lines[it], label=labels[it])
        if y_ax != 'Pr':
            ax[2].plot(Delta, Cqt_max_in[it,...], color=colours[it], linestyle=my_lines[it], label=labels[it])

    if y_ax == 'C':
        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')
        ax[2].legend(fontsize=13, loc='upper right')
    elif y_ax == 'l':
        ax[0].legend(fontsize=13, loc='best')
        ax[1].legend(fontsize=13, loc='best')
        ax[2].legend(fontsize=13, loc='best')
    elif y_ax == 'Pr':
        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')

    bottom0, top0 = ax[0].set_ylim()
    bottom1, top1 = ax[1].set_ylim()
    if y_ax != 'Pr':
        bottom2, top2 = ax[2].set_ylim()

    if max_mean == 'mean':
        set_top = max(top0, top1)
        if y_ax != 'Pr':
            temp_top = set_top
            set_top = max(temp_top, top2)  # 0.255

    elif max_mean == 'max':
        set_top = max(top0, top1)
        if y_ax != 'Pr':
            temp_top = set_top
            set_top = max(temp_top, top2)  # 0.305

    ax[0].set_ylim(top=set_top)
    ax[1].set_ylim(top=set_top)
    if y_ax != 'Pr':
        ax[2].set_ylim(top=set_top)

    if y_ax == 'C':
        ax[0].set_ylabel('$C_{s}$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$C_{\\theta}$ at '+ clock_time, fontsize=14)
        ax[2].set_ylabel('$C_{qt}$ at '+ clock_time, fontsize=14)
    elif y_ax == 'l':
        ax[0].set_ylabel('$l_{mix}$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$l_{\\theta}$ at '+ clock_time, fontsize=14)
        ax[2].set_ylabel('$l_{qt}$ at '+ clock_time, fontsize=14)
    elif y_ax == 'Pr':
        ax[0].set_ylabel('$Pr$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$Sc_{qt}$ at '+ clock_time, fontsize=14)

    # ax[0].set_title(y_labels[0], fontsize=16)
    # ax[1].set_title(y_labels[1], fontsize=16)
    # ax[2].set_title(y_labels[2], fontsize=16)

    ax[0].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
    ax[1].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
    if y_ax != 'Pr':
        ax[2].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)

    plt.tight_layout()

    plt.savefig(plotdir+f'{max_mean}_{y_ax}{what_plotting}_time{time_in}_prof.png', bbox_inches='tight')
    plt.savefig(plotdir + f'{max_mean}_{y_ax}{what_plotting}_time{time_in}_prof.pdf', bbox_inches='tight')
    plt.close()



def plot_max_mean_dom_av_vs_Delta(Cs_val_in, Cth_val_in, Cqt_val_in, Cs_err, Cth_err, Cqt_err,
                                  Delta, y_ax, max_mean='mean', time_in = '14400'):

    clock_time_int = 05.30 + int(time_in)/(60*60)
    clock_time = str(clock_time_int)+'0L'

    if y_ax == 'C':
        y_labels = ['$C_{s}$', '$C_{\\theta}$', '$C_{qt}$']
    elif y_ax == 'l':
        y_labels = ['$l_{s}$ (m)', '$l_{\\theta}$ (m)', '$l_{qt}$ (m)']
    elif y_ax == 'Pr':
        y_labels = ['$Pr$', '$Sc_{qt}$']
    else:
        print('y_ax input not recognised')

    if y_ax == 'Pr':
        fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(4, 8))
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(4, 12))
    fig.tight_layout(pad=0.5)

    ax[0].plot(Delta, Cs_val_in, 'k')
    ax[1].plot(Delta, Cth_val_in, 'k')
    if y_ax != 'Pr':
        ax[2].plot(Delta, Cqt_val_in, 'k')

    # for i, D_in in enumerate(Delta):
    #     ax[0].errorbar(D_in[i], Cs_val_in[i], yerr=Cs_err[i], color='k', ecolor='k', fmt='.', capsize=7)
    #     ax[1].errorbar(D_in[i], Cth_val_in[i], yerr=Cth_err[i], color='k', ecolor='k', fmt='.', capsize=7)
    #     if y_ax != 'Pr':
    #         ax[2].errorbar(D_in[i], Cqt_val_in[i], yerr=Cqt_err[i], color='k', ecolor='k', fmt='.', capsize=7)

    if y_ax == 'C':
        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')
        ax[2].legend(fontsize=13, loc='upper right')
    elif y_ax == 'l':
        ax[0].legend(fontsize=13, loc='best')
        ax[1].legend(fontsize=13, loc='best')
        ax[2].legend(fontsize=13, loc='best')
    elif y_ax == 'Pr':
        ax[0].legend(fontsize=13, loc='upper right')
        ax[1].legend(fontsize=13, loc='upper right')
    else:
        print('y_ax not recognised')

    bottom0, top0 = ax[0].set_ylim()
    bottom1, top1 = ax[1].set_ylim()
    if y_ax != 'Pr':
        bottom2, top2 = ax[2].set_ylim()

    if max_mean == 'mean':
       set_top = max(top0, top1)
       if y_ax != 'Pr':
           temp_top = set_top
           set_top = max(temp_top, top2) #0.255

    elif max_mean == 'max':
       set_top= max(top0, top1)
       if y_ax != 'Pr':
           temp_top = set_top
           set_top = max(temp_top, top2) #0.305

    ax[0].set_ylim(top=set_top)
    ax[1].set_ylim(top=set_top)
    if y_ax != 'Pr':
        ax[2].set_ylim(top=set_top)

    if y_ax == 'C':
        ax[0].set_ylabel('$C_{s}$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$C_{\\theta}$ at '+ clock_time, fontsize=14)
        ax[2].set_ylabel('$C_{qt}$ at '+ clock_time, fontsize=14)
    elif y_ax == 'l':
        ax[0].set_ylabel('$l_{mix}$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$l_{\\theta}$ at '+ clock_time, fontsize=14)
        ax[2].set_ylabel('$l_{qt}$ at '+ clock_time, fontsize=14)
    elif y_ax == 'Pr':
        ax[0].set_ylabel('$Pr$ at '+ clock_time, fontsize=14)
        ax[1].set_ylabel('$Sc_{qt}$ at '+ clock_time, fontsize=14)

    # ax[0].set_title(y_labels[0], fontsize=16)
    # ax[1].set_title(y_labels[1], fontsize=16)
    # ax[2].set_title(y_labels[2], fontsize=16)

    ax[0].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
    ax[1].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
    if y_ax != 'Pr':
        ax[2].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)

    plt.tight_layout()

    plt.savefig(plotdir+f'{max_mean}_{y_ax}{what_plotting}_time{time_in}_prof.png', bbox_inches='tight')
    plt.savefig(plotdir + f'{max_mean}_{y_ax}{what_plotting}_time{time_in}_prof.pdf', bbox_inches='tight')
    plt.close()



#######################################################################################################################


for itr, time_stamp in enumerate(set_time):

    z_cl_range = z_cl_r_t_list[itr]
    z_ml_range = z_ml_r_t_list[itr]

    file_name = f"diagnostics_3d_ts_{time_stamp}_gaussian_filter_C_"
    mydir = homedir + file_name
    prof_file = profiles_dir+time_stamp+'.nc'

    if beta == True:
        if what_plotting == '_0' or what_plotting == '_beta':
            data_2D_0 = xr.open_dataset(mydir + f'2D_0.nc')
            data_4D_0 = xr.open_dataset(mydir + f'4D_0.nc')
            data_8D_0 = xr.open_dataset(mydir + f'8D_0.nc')
            data_16D_0 = xr.open_dataset(mydir + f'16D_0.nc')
            data_32D_0 = xr.open_dataset(mydir + f'32D_0.nc')
            data_64D_0 = xr.open_dataset(mydir + f'64D_0.nc')

        if what_plotting == '_1' or what_plotting == '_beta':
            data_2D_1 = xr.open_dataset(mydir + f'2D_1.nc')
            data_4D_1 = xr.open_dataset(mydir + f'4D_1.nc')
            data_8D_1 = xr.open_dataset(mydir + f'8D_1.nc')
            data_16D_1 = xr.open_dataset(mydir + f'16D_1.nc')
            data_32D_1 = xr.open_dataset(mydir + f'32D_1.nc')
            data_64D_1 = xr.open_dataset(mydir + f'64D_1.nc')

        if C_or_LM == 'C':
            if what_plotting == '_0':
                data_list = [data_2D_0, data_4D_0, data_8D_0, data_16D_0, data_32D_0, data_64D_0]
                set_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                              '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
                deltas_in = ['2D', '4D', '8D', '16D', '32D', '64D']
                delta_numbers = np.array((2, 4, 8, 16, 32, 64))
            elif what_plotting == '_1':
                data_list = [data_2D_1, data_4D_1, data_8D_1, data_16D_1, data_32D_1, data_64D_1]
                set_labels = ['4$\\Delta$', '8$\\Delta$', '16$\\Delta$',
                              '32$\\Delta$', '64$\\Delta$', '$128\\Delta$']
                deltas_in = ['4D', '8D', '16D', '32D', '64D', '128D']
                delta_numbers = np.array((4, 8, 16, 32, 64, 128))
            else:
                data_list = [[data_2D_0, data_4D_0, data_8D_0, data_16D_0, data_32D_0, data_64D_0],
                        [data_2D_1, data_4D_1, data_8D_1, data_16D_1, data_32D_1, data_64D_1]]
                set_labels = ['$\\Delta$', '2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                              '16$\\Delta$', '32$\\Delta$']
                deltas_in = ['D' '2D', '4D', '8D', '16D', '32D']
                delta_numbers = np.array((1, 2, 4, 8, 16, 32))
        else:
            data_list = [data_2D_0, data_4D_0, data_8D_0, data_16D_0, data_32D_0, data_64D_0]
            set_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                          '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
            deltas_in = ['2D', '4D', '8D', '16D', '32D', '64D']
            delta_numbers = np.array((2, 4, 8, 16, 32, 64))


    else:
        data_2D = xr.open_dataset(mydir + f'2D.nc')
        data_4D = xr.open_dataset(mydir + f'4D.nc')
        data_8D = xr.open_dataset(mydir + f'8D.nc')
        data_16D = xr.open_dataset(mydir + f'16D.nc')
        data_32D = xr.open_dataset(mydir + f'32D.nc')
        data_64D = xr.open_dataset(mydir + f'64D.nc')

        data_list = [data_2D, data_4D, data_8D, data_16D, data_32D, data_64D]
        set_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                      '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']



    ##################################################################################################################

    #index of 0 at the start is to get rid of the dummy time index thats required to save the files
    if what_plotting=='_beta':
        Cs_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cth_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cqt_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cs_env_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cth_env_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cqt_env_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cs_cloud_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cth_cloud_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cqt_cloud_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cs_w_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cth_w_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cqt_w_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cs_w_th_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cth_w_th_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))
        Cqt_w_th_sq_temp = np.zeros((2, len(data_list[1]), len(zn_set)))

        print('shape of initialised C_temps = ', np.shape(Cs_sq_temp))

        for j in range(2):
            for i in range(len(data_list[1])):
                print('j = ', j)
                print('i = ', i)

                my_C_or_LM_profs = C_or_LM_profs

                print('shape of C data being imported =', np.shape(data_list[i]['Cs_sq_prof'].data))

                Cs_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[0]}_prof'].data[0, ...]
                Cth_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[1]}_prof'].data[0, ...]
                Cqt_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[2]}_prof'].data[0, ...]

                Cs_env_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[0]}_env_prof'].data[0, ...]
                Cth_env_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[1]}_env_prof'].data[0, ...]
                Cqt_env_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[2]}_env_prof'].data[0, ...]

                Cs_cloud_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[0]}_cloud_prof'].data[0, ...]
                Cth_cloud_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[1]}_cloud_prof'].data[0, ...]
                Cqt_cloud_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[2]}_cloud_prof'].data[0, ...]

                Cs_w_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[0]}_w_prof'].data[0, ...]
                Cth_w_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[1]}_w_prof'].data[0, ...]
                Cqt_w_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[2]}_w_prof'].data[0, ...]

                Cs_w_th_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[0]}_w_{th_name}_prof'].data[0, ...]
                Cth_w_th_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[1]}_w_{th_name}_prof'].data[0, ...]
                Cqt_w_th_sq_temp[j, i, :] = data_list[i][f'{my_C_or_LM_profs[2]}_w_{th_name}_prof'].data[0, ...]

        Cs_sq = Cs_sq_temp[0,...] / dyn.beta_calc(Cs_sq_temp[0,...], Cs_sq_temp[1,...])
        Cth_sq = Cth_sq_temp[0, ...] / dyn.beta_calc(Cth_sq_temp[0, ...], Cth_sq_temp[1, ...])
        Cqt_sq = Cqt_sq_temp[0, ...] / dyn.beta_calc(Cqt_sq_temp[0, ...], Cqt_sq_temp[1, ...])

        Cs_env_sq = Cs_env_sq_temp[0, ...] / dyn.beta_calc(Cs_env_sq_temp[0, ...], Cs_env_sq_temp[1, ...])
        Cth_env_sq = Cth_env_sq_temp[0, ...] / dyn.beta_calc(Cth_env_sq_temp[0, ...], Cth_env_sq_temp[1, ...])
        Cqt_env_sq = Cqt_env_sq_temp[0, ...] / dyn.beta_calc(Cqt_env_sq_temp[0, ...], Cqt_env_sq_temp[1, ...])

        Cs_cloud_sq = Cs_cloud_sq_temp[0, ...] / dyn.beta_calc(Cs_cloud_sq_temp[0, ...], Cs_cloud_sq_temp[1, ...])
        Cth_cloud_sq = Cth_cloud_sq_temp[0, ...] / dyn.beta_calc(Cth_cloud_sq_temp[0, ...], Cth_cloud_sq_temp[1, ...])
        Cqt_cloud_sq = Cqt_cloud_sq_temp[0, ...] / dyn.beta_calc(Cqt_cloud_sq_temp[0, ...], Cqt_cloud_sq_temp[1, ...])

        Cs_w_sq = Cs_w_sq_temp[0, ...] / dyn.beta_calc(Cs_w_sq_temp[0, ...], Cs_w_sq_temp[1, ...])
        Cth_w_sq = Cth_w_sq_temp[0, ...] / dyn.beta_calc(Cth_w_sq_temp[0, ...], Cth_w_sq_temp[1, ...])
        Cqt_w_sq = Cqt_w_sq_temp[0, ...] / dyn.beta_calc(Cqt_w_sq_temp[0, ...], Cqt_w_sq_temp[1, ...])

        Cs_w_th_sq = Cs_w_th_sq_temp[0, ...]  / dyn.beta_calc(Cs_w_th_sq_temp[0, ...], Cs_w_th_sq_temp[1, ...])
        Cth_w_th_sq = Cth_w_th_sq_temp[0, ...] / dyn.beta_calc(Cth_w_th_sq_temp[0, ...], Cth_w_th_sq_temp[1, ...])
        Cqt_w_th_sq = Cqt_w_th_sq_temp[0, ...] / dyn.beta_calc(Cqt_w_th_sq_temp[0, ...], Cqt_w_th_sq_temp[1, ...])


    else:
        Cs_sq = np.zeros((len(data_list), len(zn_set)))
        Cth_sq = np.zeros((len(data_list), len(zn_set)))
        Cqt_sq = np.zeros((len(data_list), len(zn_set)))
        Cs_env_sq = np.zeros((len(data_list), len(zn_set)))
        Cth_env_sq = np.zeros((len(data_list), len(zn_set)))
        Cqt_env_sq = np.zeros((len(data_list), len(zn_set)))
        Cs_cloud_sq = np.zeros((len(data_list), len(zn_set)))
        Cth_cloud_sq = np.zeros((len(data_list), len(zn_set)))
        Cqt_cloud_sq = np.zeros((len(data_list), len(zn_set)))
        Cs_w_sq = np.zeros((len(data_list), len(zn_set)))
        Cth_w_sq = np.zeros((len(data_list), len(zn_set)))
        Cqt_w_sq = np.zeros((len(data_list), len(zn_set)))
        Cs_w_th_sq = np.zeros((len(data_list), len(zn_set)))
        Cth_w_th_sq = np.zeros((len(data_list), len(zn_set)))
        Cqt_w_th_sq = np.zeros((len(data_list), len(zn_set)))

        for i in range(len(data_list)):
            if C_or_LM == 'C':

                Cs_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][0]}_prof'].data[0, ...] /
                                     data_list[i][f'{C_or_LM_profs[1][0]}_prof'].data[0, ...] )
                Cth_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][1]}_prof'].data[0, ...] /
                                      data_list[i][f'{C_or_LM_profs[1][1]}_prof'].data[0, ...] )
                Cqt_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][2]}_prof'].data[0, ...] /
                                      data_list[i][f'{C_or_LM_profs[1][2]}_prof'].data[0, ...] )

                Cs_env_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][0]}_env_prof'].data[0, ...] /
                                         data_list[i][f'{C_or_LM_profs[1][0]}_env_prof'].data[0, ...] )
                Cth_env_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][1]}_env_prof'].data[0, ...] /
                                          data_list[i][f'{C_or_LM_profs[1][1]}_env_prof'].data[0, ...] )
                Cqt_env_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][2]}_env_prof'].data[0, ...] /
                                          data_list[i][f'{C_or_LM_profs[1][2]}_env_prof'].data[0, ...] )

                Cs_cloud_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][0]}_cloud_prof'].data[0, ...] /
                                           data_list[i][f'{C_or_LM_profs[1][0]}_cloud_prof'].data[0, ...] )
                Cth_cloud_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][1]}_cloud_prof'].data[0, ...] /
                                            data_list[i][f'{C_or_LM_profs[1][1]}_cloud_prof'].data[0, ...] )
                Cqt_cloud_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][2]}_cloud_prof'].data[0, ...] /
                                            data_list[i][f'{C_or_LM_profs[1][2]}_cloud_prof'].data[0, ...] )

                Cs_w_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][0]}_w_prof'].data[0, ...] /
                                       data_list[i][f'{C_or_LM_profs[1][0]}_w_prof'].data[0, ...] )
                Cth_w_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][1]}_w_prof'].data[0, ...] /
                                        data_list[i][f'{C_or_LM_profs[1][1]}_w_prof'].data[0, ...] )
                Cqt_w_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][2]}_w_prof'].data[0, ...] /
                                        data_list[i][f'{C_or_LM_profs[1][2]}_w_prof'].data[0, ...] )

                Cs_w_th_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][0]}_w_{th_name}_prof'].data[0, ...] /
                                          data_list[i][f'{C_or_LM_profs[1][0]}_w_{th_name}_prof'].data[0, ...] )
                Cth_w_th_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][1]}_w_{th_name}_prof'].data[0, ...] /
                                           data_list[i][f'{C_or_LM_profs[1][1]}_w_{th_name}_prof'].data[0, ...] )
                Cqt_w_th_sq[i, :] = 0.5 * (data_list[i][f'{C_or_LM_profs[0][2]}_w_{th_name}_prof'].data[0, ...] /
                                           data_list[i][f'{C_or_LM_profs[1][2]}_w_{th_name}_prof'].data[0, ...] )
            else:
                Cs_sq[i, :] = data_list[i][f'{C_or_LM_profs[0]}_prof'].data[0, ...]
                Cth_sq[i, :] = data_list[i][f'{C_or_LM_profs[1]}_prof'].data[0, ...]
                Cqt_sq[i, :] = data_list[i][f'{C_or_LM_profs[2]}_prof'].data[0, ...]

                Cs_env_sq[i, :] = data_list[i][f'{C_or_LM_profs[0]}_env_prof'].data[0, ...]
                Cth_env_sq[i, :] = data_list[i][f'{C_or_LM_profs[1]}_env_prof'].data[0, ...]
                Cqt_env_sq[i, :] = data_list[i][f'{C_or_LM_profs[2]}_env_prof'].data[0, ...]

                Cs_cloud_sq[i, :] = data_list[i][f'{C_or_LM_profs[0]}_cloud_prof'].data[0, ...]
                Cth_cloud_sq[i, :] = data_list[i][f'{C_or_LM_profs[1]}_cloud_prof'].data[0, ...]
                Cqt_cloud_sq[i, :] = data_list[i][f'{C_or_LM_profs[2]}_cloud_prof'].data[0, ...]

                Cs_w_sq[i, :] = data_list[i][f'{C_or_LM_profs[0]}_w_prof'].data[0, ...]
                Cth_w_sq[i, :] = data_list[i][f'{C_or_LM_profs[1]}_w_prof'].data[0, ...]
                Cqt_w_sq[i, :] = data_list[i][f'{C_or_LM_profs[2]}_w_prof'].data[0, ...]

                Cs_w_th_sq[i, :] = data_list[i][f'{C_or_LM_profs[0]}_w_{th_name}_prof'].data[0, ...]
                Cth_w_th_sq[i, :] = data_list[i][f'{C_or_LM_profs[1]}_w_{th_name}_prof'].data[0, ...]
                Cqt_w_th_sq[i, :] = data_list[i][f'{C_or_LM_profs[2]}_w_{th_name}_prof'].data[0, ...]
        print('shape of Cs_sq = ', np.shape(Cs_sq))

    ########################################################################################################################

    z_ML_index, z_cl_range_calc, zn_arr = calc_z_ML_and_CL(prof_file)
    z_ML = zn_set[z_ML_index]

    # print('zn_set = ', zn_set)
    # print('zn_arr = ', zn_arr)

    z_ml_range_calc = [z_ML_bottom, z_ML_index]




    #plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, interp=True)
    #plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, labels_in=set_labels)

    #plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, interp=True, C_sq_to_C = True)
    print('shape of Cs_sq being fed into fn:', np.shape(Cs_sq))
    plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, z_cl_range, labels_in=set_labels,
                      C_sq_to_C = True, time_in=time_stamp)

    print('saved C plots to ', plotdir)

    plot_Pr_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, z_cl_range, labels_in=set_labels, time_in=time_stamp)



    #################################################################################


    Cs_sq_cond = [Cs_sq, Cs_env_sq, Cs_cloud_sq, Cs_w_sq, Cs_w_th_sq]
    Cth_sq_cond = [Cth_sq, Cth_env_sq, Cth_cloud_sq, Cth_w_sq, Cth_w_th_sq]
    Cqt_sq_cond = [Cqt_sq, Cqt_env_sq, Cqt_cloud_sq, Cqt_w_sq, Cqt_w_th_sq]

    Cs_sq_cond = np.reshape(Cs_sq_cond, ( np.shape(Cs_sq_cond)[0], np.shape(Cs_sq_cond)[1], np.shape(Cs_sq_cond)[2] ))
    Cth_sq_cond = np.reshape(Cth_sq_cond, ( np.shape(Cth_sq_cond)[0], np.shape(Cth_sq_cond)[1], np.shape(Cth_sq_cond)[2] ))
    Cqt_sq_cond = np.reshape(Cqt_sq_cond, ( np.shape(Cqt_sq_cond)[0], np.shape(Cqt_sq_cond)[1], np.shape(Cqt_sq_cond)[2] ))

    np.save(homedir+f'Cs_sq_cond_{time_stamp}.npy', Cs_sq_cond)
    np.save(homedir+f'Cth_sq_cond_{time_stamp}.npy', Cth_sq_cond)
    np.save(homedir+f'Cqt_sq_cond_{time_stamp}.npy', Cqt_sq_cond)




    #plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, z_set, z_ML, interp=True, C_sq_to_C = False)
    # plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, zn_set, z_ML,
    #                           deltas = deltas_in, delta_label = set_labels, interp=False, C_sq_to_C = False)

    #plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, z_set, z_ML, interp=True, C_sq_to_C = True)
    plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, zn_set, z_ML, z_cl_range,
                              deltas = deltas_in, delta_label = set_labels, interp=False,
                              C_sq_to_C = True, time_in=time_stamp, set_x_lim_list=x_lim_list)


    ##################################################################################################################




    max_Cs_sq_cond = cal_max_Cs(Cs_sq_cond, z_ml_range, z_cl_range)
    max_Cth_sq_cond = cal_max_Cs(Cth_sq_cond, z_ml_range, z_cl_range)
    max_Cqt_sq_cond = cal_max_Cs(Cqt_sq_cond, z_ml_range, z_cl_range)

    max_Cs_cond = dyn.get_Cs(max_Cs_sq_cond)
    max_Cth_cond = dyn.get_Cs(max_Cth_sq_cond)
    max_Cqt_cond = dyn.get_Cs(max_Cqt_sq_cond)




    mean_Cs_sq_cond = cal_mean_Cs(Cs_sq_cond, z_ml_range, z_cl_range)
    mean_Cth_sq_cond = cal_mean_Cs(Cth_sq_cond, z_ml_range, z_cl_range)
    mean_Cqt_sq_cond = cal_mean_Cs(Cqt_sq_cond, z_ml_range, z_cl_range)

    mean_Cs_cond = dyn.get_Cs(mean_Cs_sq_cond)
    mean_Cth_cond = dyn.get_Cs(mean_Cth_sq_cond)
    mean_Cqt_cond = dyn.get_Cs(mean_Cqt_sq_cond)




    Pr_partit_profs = Cs_sq_cond / Cth_sq_cond
    Sc_partit_profs = Cs_sq_cond / Cqt_sq_cond

    max_Pr_cond = cal_max_Cs(Pr_partit_profs, z_ml_range, z_cl_range)
    max_Sc_cond = cal_max_Cs(Sc_partit_profs, z_ml_range, z_cl_range)

    mean_Pr_cond = cal_mean_Cs(Pr_partit_profs, z_ml_range, z_cl_range)
    mean_Sc_cond = cal_mean_Cs(Sc_partit_profs, z_ml_range, z_cl_range)





    #Cs_sq in ML, Cs_sq in CL, Cs_env_sq, Cs_cloud_sq, Cs_w_sq, Cs_w_th_sq




    plot_max_C_l_vs_Delta(max_Cs_cond, max_Cth_cond, max_Cqt_cond, Delta = set_labels, y_ax = 'C', max_mean='max',
                          time_in = time_stamp)
    plot_max_C_l_vs_Delta(get_max_l_from_C(max_Cs_cond, delta_numbers, 25), get_max_l_from_C(max_Cth_cond, delta_numbers, 25),
                          get_max_l_from_C(max_Cqt_cond, delta_numbers, 25), Delta = set_labels, y_ax = 'l',
                          max_mean='max', time_in = time_stamp)


    plot_max_C_l_vs_Delta(mean_Cs_cond, mean_Cth_cond, mean_Cqt_cond, Delta = set_labels, y_ax = 'C', max_mean='mean',
                          time_in = time_stamp)
    plot_max_C_l_vs_Delta(get_max_l_from_C(mean_Cs_cond, delta_numbers, 25), get_max_l_from_C(mean_Cth_cond, delta_numbers, 25),
                          get_max_l_from_C(mean_Cqt_cond, delta_numbers, 25), Delta = set_labels, y_ax = 'l',
                          max_mean='mean', time_in = time_stamp)

    plot_max_C_l_vs_Delta(mean_Pr_cond, max_Sc_cond, None, Delta=set_labels, y_ax='Pr', max_mean='max',
                          time_in=time_stamp)
    plot_max_C_l_vs_Delta(mean_Pr_cond, mean_Sc_cond, None, Delta=set_labels, y_ax='Pr', max_mean='mean',
                          time_in=time_stamp)


    print('z_ml_range = ', z_ml_range)
    print('z_cl_range = ', z_cl_range)



    # #########################################################################################################################
    #
    #
    # Pr_th_beta = dyn.Pr(Cs_beta_sq, Cth_beta_sq)
    # Pr_th_2D = dyn.Pr(Cs_sq_2, Cth_sq_2)
    # Pr_th_4D = dyn.Pr(Cs_sq_4, Cth_sq_4)
    # Pr_th_8D = dyn.Pr(Cs_sq_8, Cth_sq_8)
    # Pr_th_16D = dyn.Pr(Cs_sq_16, Cth_sq_16)
    # Pr_th_32D = dyn.Pr(Cs_sq_32, Cth_sq_32)
    # Pr_th_64D = dyn.Pr(Cs_sq_64, Cth_sq_64)



