import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

beta=True
what_plotting='_beta'

if beta == True:
    scale_invar_dir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/C_profs_cloud_1e-7/'
    homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/smoothed_LM_HR_fields/C_profs/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/scale_dep_plots/C_beta_profiles/'

else:
    homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/C_profs_cloud_1e-7/'
    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/profiles_cloud_1e-7/'

file_name = 'BOMEX_m0020_g0800_all_14400_gaussian_filter_C_'
mydir = homedir + file_name


os.makedirs(plotdir, exist_ok = True)

if beta == True:
    data_2D = xr.open_dataset(scale_invar_dir + file_name + f'2D.nc')
    data_4D = xr.open_dataset(scale_invar_dir + file_name + f'4D.nc')

    data_2D_0 = xr.open_dataset(mydir + f'2D_0.nc')
    data_4D_0 = xr.open_dataset(mydir + f'4D_0.nc')
    data_8D_0 = xr.open_dataset(mydir + f'8D_0.nc')
    data_16D_0 = xr.open_dataset(mydir + f'16D_0.nc')
    data_32D_0 = xr.open_dataset(mydir + f'32D_0.nc')
    data_64D_0 = xr.open_dataset(mydir + f'64D_0.nc')

    data_2D_1 = xr.open_dataset(mydir + f'2D_1.nc')
    data_4D_1 = xr.open_dataset(mydir + f'4D_1.nc')
    data_8D_1 = xr.open_dataset(mydir + f'8D_1.nc')
    data_16D_1 = xr.open_dataset(mydir + f'16D_1.nc')
    data_32D_1 = xr.open_dataset(mydir + f'32D_1.nc')
    data_64D_1 = xr.open_dataset(mydir + f'64D_1.nc')

    if what_plotting == '_0':
        data_list = [data_2D, data_2D_0, data_4D_0, data_8D_0, data_16D_0, data_32D_0, data_64D_0]
        set_labels = ['2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                      '16$\\Delta$', '32$\\Delta$', '64$\\Delta$', '$128\\Delta$']
        deltas_in = ['2D', '4D', '8D', '16D', '32D', '64D', '128D']
        delta_numbers = np.array((2, 4, 8, 16, 32, 64, 128))
    elif what_plotting == '_1':
        data_list = [data_4D, data_2D_1, data_4D_1, data_8D_1, data_16D_1, data_32D_1, data_64D_1]
        set_labels = ['4$\\Delta$', '8$\\Delta$', '16$\\Delta$',
                      '32$\\Delta$', '64$\\Delta$', '$128\\Delta$', '$256\\Delta$']
        deltas_in = ['4D', '8D', '16D', '32D', '64D', '128D', '256D']
        delta_numbers = np.array((4, 8, 16, 32, 64, 128, 256))
    else:
        data_list = [[data_2D, data_2D_0, data_4D_0, data_8D_0, data_16D_0, data_32D_0, data_64D_0],
                [data_4D, data_2D_1, data_4D_1, data_8D_1, data_16D_1, data_32D_1, data_64D_1]]
        set_labels = ['$\\Delta$', '2$\\Delta$', '4$\\Delta$', '8$\\Delta$',
                      '16$\\Delta$', '32$\\Delta$', '64$\\Delta$']
        deltas_in = ['D' '2D', '4D', '8D', '16D', '32D', '64D']
        delta_numbers = np.array((1, 2, 4, 8, 16, 32, 64))


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

zn_set = np.arange(0, 3020, 20)
z_set = np.arange(-10, 3010, 20)
z_ML = 490

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

            print('shape of C data being imported =', np.shape(data_list[j][i]['Cs_sq_prof'].data))

            Cs_sq_temp[j, i, :] = data_list[j][i]['Cs_sq_prof'].data[0, ...]
            Cth_sq_temp[j, i, :] = data_list[j][i]['Cth_sq_prof'].data[0, ...]
            Cqt_sq_temp[j, i, :] = data_list[j][i]['Cqt_sq_prof'].data[0, ...]

            Cs_env_sq_temp[j, i, :] = data_list[j][i]['Cs_sq_env_prof'].data[0, ...]
            Cth_env_sq_temp[j, i, :] = data_list[j][i]['Cth_sq_env_prof'].data[0, ...]
            Cqt_env_sq_temp[j, i, :] = data_list[j][i]['Cqt_sq_env_prof'].data[0, ...]

            Cs_cloud_sq_temp[j, i, :] = data_list[j][i]['Cs_sq_cloud_prof'].data[0, ...]
            Cth_cloud_sq_temp[j, i, :] = data_list[j][i]['Cth_sq_cloud_prof'].data[0, ...]
            Cqt_cloud_sq_temp[j, i, :] = data_list[j][i]['Cqt_sq_cloud_prof'].data[0, ...]

            Cs_w_sq_temp[j, i, :] = data_list[j][i]['Cs_sq_w_prof'].data[0, ...]
            Cth_w_sq_temp[j, i, :] = data_list[j][i]['Cth_sq_w_prof'].data[0, ...]
            Cqt_w_sq_temp[j, i, :] = data_list[j][i]['Cqt_sq_w_prof'].data[0, ...]

            Cs_w_th_sq_temp[j, i, :] = data_list[j][i]['Cs_sq_w_th_prof'].data[0, ...]
            Cth_w_th_sq_temp[j, i, :] = data_list[j][i]['Cth_sq_w_th_prof'].data[0, ...]
            Cqt_w_th_sq_temp[j, i, :] = data_list[j][i]['Cqt_sq_w_th_prof'].data[0, ...]

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

        Cs_sq[i,:] = data_list[i]['Cs_sq_prof'].data[0, ...]
        Cth_sq[i,:] = data_list[i]['Cth_sq_prof'].data[0, ...]
        Cqt_sq[i,:] = data_list[i]['Cqt_sq_prof'].data[0, ...]

        Cs_env_sq[i,:] = data_list[i]['Cs_sq_env_prof'].data[0, ...]
        Cth_env_sq[i,:] = data_list[i]['Cth_sq_env_prof'].data[0, ...]
        Cqt_env_sq[i,:] = data_list[i]['Cqt_sq_env_prof'].data[0, ...]

        Cs_cloud_sq[i,:] = data_list[i]['Cs_sq_cloud_prof'].data[0, ...]
        Cth_cloud_sq[i,:] = data_list[i]['Cth_sq_cloud_prof'].data[0, ...]
        Cqt_cloud_sq[i,:] = data_list[i]['Cqt_sq_cloud_prof'].data[0, ...]

        Cs_w_sq[i,:] = data_list[i]['Cs_sq_w_prof'].data[0, ...]
        Cth_w_sq[i,:] = data_list[i]['Cth_sq_w_prof'].data[0, ...]
        Cqt_w_sq[i,:] = data_list[i]['Cqt_sq_w_prof'].data[0, ...]

        Cs_w_th_sq[i,:] = data_list[i]['Cs_sq_w_th_prof'].data[0, ...]
        Cth_w_th_sq[i,:] = data_list[i]['Cth_sq_w_th_prof'].data[0, ...]
        Cqt_w_th_sq[i,:] = data_list[i]['Cqt_sq_w_th_prof'].data[0, ...]

########################################################################################################################

def interp_z(var_in, z_from=z_set, z_to=zn_set):
    interp_var = np.zeros_like(var_in)
    for n in range(len(var_in[:,0])):
        for k in range(len(z_from)-1):
            interp_var[n,k] = var_in[n,k] + (z_to[k] - z_from[k])*( (var_in[n,k+1] - var_in[n,k]) / (z_from[k+1] - z_from[k]) )
    return interp_var


def plot_C_all_Deltas(Cs, Cth, Cqt, z, z_i, labels_in, interp=False, C_sq_to_C = False):

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    if interp==True:
        Cs = interp_z(Cs)
        Cth = interp_z(Cth)
        Cqt = interp_z(Cqt)
    if C_sq_to_C == True:
        Cs = dyn.get_Cs(Cs)
        Cth = dyn.get_Cs(Cth)
        Cqt = dyn.get_Cs(Cqt)
        name='_'
    else:
        name='_sq_'

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,6))

    fig.tight_layout(pad=0.5)

    for it in range(len(Cs[:,0])):
        ax[0].plot(Cs[it,:], z/z_i, color=colours[it])
        ax[1].plot(Cth[it, :], z/z_i, color=colours[it])
        ax[2].plot(Cqt[it, :], z/z_i, color=colours[it], label='$\\widehat{\\bar{\\Delta}} = $'+labels_in[it])
    if C_sq_to_C == True:
        ax[0].set_xlabel('$C_{s}$', fontsize=16)
        ax[1].set_xlabel('$C_{\\theta}$', fontsize=16)
        ax[2].set_xlabel('$C_{qt}$', fontsize=16)
    else:
        ax[0].set_xlabel('$C^2_{s}$', fontsize=16)
        ax[1].set_xlabel('$C^2_{\\theta}$', fontsize=16)
        ax[2].set_xlabel('$C^2_{qt}$', fontsize=16)
    ax[2].legend(fontsize=13, loc='best')

    left0, right0 = ax[0].set_xlim()
    left1, right1 = ax[1].set_xlim()
    left2, right2 = ax[2].set_xlim()

    set_right = max(right0, right1, right2)

    ax[0].set_xlim(right = set_right)
    ax[1].set_xlim(right = set_right)
    ax[2].set_xlim(right = set_right)

    if interp==True:
        ax[0].set_ylabel("z/z$_{ML}$", fontsize=16)
        plt.savefig(plotdir + f'C{what_plotting}{name}prof_scaled_interp_z.png', bbox_inches='tight')
    else:
        ax[0].set_ylabel("zn/z$_{ML}$", fontsize=16)
        plt.savefig(plotdir + f'C{what_plotting}{name}prof_scaled_zn.png', bbox_inches='tight')
    plt.close()


#plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, interp=True)
plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, labels_in=set_labels)

#plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, interp=True, C_sq_to_C = True)
plot_C_all_Deltas(Cs_sq, Cth_sq, Cqt_sq, zn_set, z_ML, labels_in=set_labels, C_sq_to_C = True)

print('saved C plots to ', plotdir)



#################################################################################


Cs_sq_cond = [Cs_sq, Cs_env_sq, Cs_cloud_sq, Cs_w_sq, Cs_w_th_sq]
Cth_sq_cond = [Cth_sq, Cth_env_sq, Cth_cloud_sq, Cth_w_sq, Cth_w_th_sq]
Cqt_sq_cond = [Cqt_sq, Cqt_env_sq, Cqt_cloud_sq, Cqt_w_sq, Cqt_w_th_sq]

Cs_sq_cond = np.reshape(Cs_sq_cond, ( np.shape(Cs_sq_cond)[0], np.shape(Cs_sq_cond)[1], np.shape(Cs_sq_cond)[2] ))
Cth_sq_cond = np.reshape(Cth_sq_cond, ( np.shape(Cth_sq_cond)[0], np.shape(Cth_sq_cond)[1], np.shape(Cth_sq_cond)[2] ))
Cqt_sq_cond = np.reshape(Cqt_sq_cond, ( np.shape(Cqt_sq_cond)[0], np.shape(Cqt_sq_cond)[1], np.shape(Cqt_sq_cond)[2] ))

def plot_condit_C_each_Deltas(Cs_in, Cth_in, Cqt_in, z, z_i, interp=False, C_sq_to_C = True, deltas, delta_label,
                      labels_in = ['total', 'cloud-free', 'in-cloud', 'cloud updraft', 'cloud core']):

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    print('np.shape(Cs) = ', np.shape(Cs_in)) # C shape is [conditionals, Delta, z]

    Cs_temp = np.zeros_like(Cs_in)
    Cth_temp = np.zeros_like(Cs_in)
    Cqt_temp = np.zeros_like(Cs_in)
    Cs = np.zeros_like(Cs_in)
    Cth = np.zeros_like(Cs_in)
    Cqt = np.zeros_like(Cs_in)

    print('np.shape(Cs_in)[1] = ', np.shape(Cs_in)[1])
    for it in range(np.shape(Cs_in)[1]):
        print('it = ', it)

        if interp==True:
            Cs_temp[:,it, :] = interp_z(Cs_in[:,it, :])
            Cth_temp[:,it, :] = interp_z(Cth_in[:,it, :])
            Cqt_temp[:,it, :] = interp_z(Cqt_in[:,it, :])
        else:
            Cs_temp = Cs_in.copy()
            Cth_temp = Cth_in.copy()
            Cqt_temp = Cqt_in.copy()

        if C_sq_to_C == True:
            Cs[:,it, :] = dyn.get_Cs(Cs_temp[:,it, :])
            Cth[:,it, :] = dyn.get_Cs(Cth_temp[:,it, :])
            Cqt[:,it, :] = dyn.get_Cs(Cqt_temp[:,it, :])
            name='_'
        else:
            name='_sq_'
            Cs = Cs_temp.copy()
            Cth = Cth_temp.copy()
            Cqt = Cqt_temp.copy()


        fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,6))
        print('np.shape(Cs) = ', np.shape(Cs))

        fig.tight_layout(pad=0.5)

        for nt in range(np.shape(Cs)[0]):
            ax[0].plot(Cs[nt, it, :], z/z_i, color=colours[nt])
            ax[1].plot(Cth[nt, it, :], z/z_i, color=colours[nt])
            ax[2].plot(Cqt[nt, it, :], z/z_i, color=colours[nt], label=labels_in[nt])
        if C_sq_to_C == True:
            ax[0].set_xlabel('$C_{s}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
            ax[1].set_xlabel('$C_{\\theta}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
            ax[2].set_xlabel('$C_{qt}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
        else:
            ax[0].set_xlabel('$C^2_{s}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
            ax[1].set_xlabel('$C^2_{\\theta}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
            ax[2].set_xlabel('$C^2_{qt}$ for $\\widehat{\\bar{\\Delta}} = $'+delta_label[it], fontsize=16)
        ax[2].legend(fontsize=13, loc='upper right')

        left0, right0 = ax[0].set_xlim()
        left1, right1 = ax[1].set_xlim()
        left2, right2 = ax[2].set_xlim()

        set_right = max(right0, right1, right2)

        ax[0].set_xlim(right = set_right)
        ax[1].set_xlim(right = set_right)
        ax[2].set_xlim(right = set_right)

        print('deltas[it] =', deltas[it])

        if interp==True:
            ax[0].set_ylabel("z/z$_{ML}$", fontsize=16)
            plt.savefig(plotdir + f'C{name}condit_prof_D={deltas[it]}{what_plotting}_scaled_interp_z.png',
                        bbox_inches='tight')
        else:
            ax[0].set_ylabel("zn/z$_{ML}$", fontsize=16)
            plt.savefig(plotdir + f'C{name}condit_prof_D={deltas[it]}{what_plotting}_scaled_zn.png',
                        bbox_inches='tight')
        plt.close()


#plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, z_set, z_ML, interp=True, C_sq_to_C = False)
plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, zn_set, z_ML,
                          deltas = deltas_in, delta_label = set_labels, interp=False, C_sq_to_C = False)

#plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, z_set, z_ML, interp=True, C_sq_to_C = True)
plot_condit_C_each_Deltas(Cs_sq_cond, Cth_sq_cond, Cqt_sq_cond, zn_set, z_ML,
                          deltas = deltas_in, delta_label = set_labels, interp=False, C_sq_to_C = True)


##################################################################################################################


z_cl_r = [50, 75]
z_ml_r = [6, 20]

def cal_max_Cs(C_list):

    print('when calc the max values, shape of C list is ', np.shape(C_list))

    max_C = np.zeros((np.shape(C_list)[0]+1, np.shape(C_list)[1]))

    for i in range(np.shape(C_list)[0]):
        for nD in range(np.shape(C_list)[1]):
            print('nD = ', nD)
            if i == 0:
                max_C[i, nD] = np.max(C_list[i, nD, z_ml_r[0]:z_ml_r[1]])
                max_C[i+1, nD] = np.max(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])
            else:
                max_C[i+1, nD] = np.max(C_list[i, nD, z_cl_r[0]:z_cl_r[1]])

    print('shape of max C is ', np.shape(max_C))
    return max_C

max_Cs_sq_cond = cal_max_Cs(Cs_sq_cond)
max_Cth_sq_cond = cal_max_Cs(Cth_sq_cond)
max_Cqt_sq_cond = cal_max_Cs(Cqt_sq_cond)

max_Cs_cond = dyn.get_Cs(max_Cs_sq_cond)
max_Cth_cond = dyn.get_Cs(max_Cth_sq_cond)
max_Cqt_cond = dyn.get_Cs(max_Cqt_sq_cond)

def get_max_l_from_C(max_C_cond, deltas_num):
    Delta_res = deltas_num*20
    max_l_cond = np.zeros_like(max_C_cond)
    for it in range(np.shape(max_C_cond)[1]):
        max_l_cond[:,it] = max_C_cond[:,it] * Delta_res[it]
    print('shape of max l is ', np.shape(max_l_cond))
    return max_l_cond

#Cs_sq in ML, Cs_sq in CL, Cs_env_sq, Cs_cloud_sq, Cs_w_sq, Cs_w_th_sq

def plot_max_C_l_vs_Delta(Cs_max_in, Cth_max_in, Cqt_max_in, Delta, y_ax):

    my_lines = ['solid', 'solid', 'dotted', 'dashed', 'dashed', 'dashed']
    labels = ['ML domain', 'CL domain', 'CL: clear sky', 'in-cloud', 'cloudy updraft', 'cloud core']

    colours = ['k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink']

    if y_ax == 'C':
        y_labels = ['$C_{s}$', '$C_{\\theta}$', '$C_{qt}$']
    else:
        y_labels = ['$l_{s}$ (m)', '$l_{\\theta}$ (m)', '$l_{qt}$ (m)']

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))
    fig.tight_layout(pad=0.5)

    for it in range(np.shape(Cs_max_in)[0]):
        ax[0].plot(Delta, Cs_max_in[it,...], color=colours[it], linestyle=my_lines[it])
        ax[1].plot(Delta, Cth_max_in[it,...], color=colours[it], linestyle=my_lines[it])
        ax[2].plot(Delta, Cqt_max_in[it,...], color=colours[it], linestyle=my_lines[it], label=labels[it])

    if y_ax == 'C':
        ax[2].legend(fontsize=13, loc='upper right')
    else:
        ax[2].legend(fontsize=13, loc='best')

    bottom0, top0 = ax[0].set_ylim()
    bottom1, top1 = ax[1].set_ylim()
    bottom2, top2 = ax[2].set_ylim()

    set_top = max(top0, top1, top2)

    ax[0].set_ylim(top=set_top)
    ax[1].set_ylim(top=set_top)
    ax[2].set_ylim(top=set_top)

    ax[0].set_ylabel('Smagorinsky Parameter', fontsize=14)

    ax[0].set_title(y_labels[0], fontsize=16)
    ax[1].set_title(y_labels[1], fontsize=16)
    ax[2].set_title(y_labels[2], fontsize=16)

    ax[1].set_xlabel('Filter scale $\\widehat{\\bar{\\Delta}}$', fontsize=14)
    plt.savefig(plotdir+f'max_{y_ax}{what_plotting}_prof.png', bbox_inches='tight')
    plt.close()




plot_max_C_l_vs_Delta(max_Cs_cond, max_Cth_cond, max_Cqt_cond, Delta = set_labels, y_ax = 'C')
plot_max_C_l_vs_Delta(get_max_l_from_C(max_Cs_cond, delta_numbers), get_max_l_from_C(max_Cth_cond, delta_numbers),
                      get_max_l_from_C(max_Cqt_cond, delta_numbers), Delta = set_labels, y_ax = 'l')



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



