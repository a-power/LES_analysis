import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import os


def get_cloud_only(CT_or_CB_field, dist_from_surf_threas=1):

    mask_no_cloud = ma.masked_less_equal(CT_or_CB_field, dist_from_surf_threas)

    return mask_no_cloud


plotting = 'SAHCs_vs_SA_Smag' # 'og_vs_HCs' 'HCs_vs_SAHCs' 'SAHCs_vs_SA_Smag' 'SAHCs_vs_SAHCsCth_L'



path_ARM25 = '/storage/silver/greybls/si818415/arm_2d_25m/diagnostics_ts_'
path_MONC_alt_HCs = '/storage/silver/scenario/si818415/altered_MONC/'
path_MONC_stand = '/storage/silver/scenario/si818415/og_monc/'

plotdir = f'/home/users/si818415/phd/plots/MONC_alt/mean_profs/{plotting}/'
os.makedirs(plotdir, exist_ok = True)

list_timestamps = [18000, 25200, 32400, 39600]
#list_timestamps = [17400, 18000, 24600, 25200, 31800, 32400, 39000, 39600]
#list_timestamps = np.arange(600, 40200, 600)

var_list = ['wtheta_cn_mean', 'wtheta_ad_mean', 'wtsg_mean',
            'wqv_cn_mean', 'wqv_ad_mean', 'wqv_sg_mean', 'w_qt',
            'ww_mean', 'wwsg_mean', 'qt_qt', 'sqt_qt',
            'theta_mean', 'total_cloud_fraction', 'tkesg_mean',
            'viscosity_coef_mean', 'diffusion_coef_mean', 'dissipation_mean'] #,
            # 'resolved_buoyant_production', 'resolved_shear_production', 'resolved_turbulent_transport'] #,
            # 'subgrid_buoyant_production', 'subgrid_shear_stress', 'subgrid_turbulent_transport']

# 'tke_tendency', 'tkesg_mean'

zn = np.arange(0, 4410, 10)
zn_440 = np.arange(0, 4400, 10)
zn_40 = np.arange(0, 4410, 40)



colour_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

line_list = ['--', '--', '--', ':', ':', ':']
model_param = ['Smag 0.23', 'Smag 0.137', 'Smag 0.11', 'Smag 0.075',
               '$C_s$ prof', 'S-A $C_s$ prof', '$C_s C_{\\theta_L}$ prof'] #'HCs $\\widehat{\\bar{\\Delta}}'

cloudtop25 = np.zeros((len(list_timestamps)))
for t, times in enumerate(list_timestamps):

    ds_in = xr.open_dataset(path_ARM25 + f'{str(times)}.nc')
    cloud25 = np.mean(ds_in['total_cloud_fraction'].data, axis=0)
    for i in range(len(cloud25)-1, 0, -1):
        if cloud25[i] != 0:
            cloudtop25[t] = zn[i]
            break



if plotting == 'og_vs_HCs':

    for nv, var in enumerate(var_list):
        print(var)
        for nt, time in enumerate(list_timestamps):

            clock_time_int = 05.30 + int(time) / (60 * 60)
            clock_time = str(clock_time_int) + '0L'

            var_prof = np.zeros( (7, len(zn) ) )
            var_prof_40 = np.zeros((7, len(zn_40)))
            var_prof_440 = np.zeros((7, len(zn_440)))

            for n in range(7):
                print(n)
                if n <3:
                    path_in = path_MONC_stand + f'{2 ** (n)}00m/'
                    if n==2:
                        filein = f'arm_{str(time)}.nc'
                        ds_in = xr.open_dataset(path_in + filein)
                        var_prof_40[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    else:
                        filein = f'arm_{str(time)}.nc'
                        ds_in = xr.open_dataset(path_in + filein)
                        var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis = 0)


                elif n<6:

                    path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'
                    filein = f'arm_{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    if n == 4:
                        var_prof_440[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    else:
                        var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis = 0)

                else:
                    path_in = path_ARM25
                    filein = f'{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[6, :] = np.mean(ds_in[f'{var}'].data, axis=0)



            plt.plot(figsize=(5, 8))

            plt.plot(var_prof[6, :], zn, 'k', linewidth=2,
                     label='LES $\\Delta$ = 25m')

            for i in range(6):
                #plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
                if i < 3:
                    if i == 2:
                        plt.plot(var_prof_40[i, :], zn_40/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                    else:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                else:
                    if i == 4:
                        plt.plot(var_prof_440[i, :], zn_440/cloudtop25[nt], colour_cycle[i % 3],
                                 label='$\\Delta$'+f' = {(2**(i-3))}00m', linestyle='--')
                            # marker='*')
                    else:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3],
                             label='$\\Delta$'+f' = {(2**(i-3))}00m', linestyle='--')
                            # marker='*')

            plt.title(f'{clock_time}: Smag 0.23 (dot) vs '+'$C_s$ prof (dash)')
            plt.tight_layout(pad=0.5)
            plt.gcf().set_size_inches(5.5, 7)
            plt.legend(fontsize=13, loc='upper right')

            bottom, top = plt.ylim()
            plt.ylim(0, 1.3)

            plt.xlabel(f'{var}', fontsize=14)
            plt.ylabel('$z$/$z_{CT}$', fontsize=14)

            plt.tight_layout()

            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_stand.png', bbox_inches='tight')
            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_stand.pdf', bbox_inches='tight')
            plt.close()




elif plotting == 'HCs_vs_SAHCs':

    for nv, var in enumerate(var_list):
        print(var)
        for nt, time in enumerate(list_timestamps):

            clock_time_int = 05.30 + int(time) / (60 * 60)
            clock_time = str(clock_time_int) + '0L'

            var_prof = np.zeros((7, len(zn)))
            var_prof_40 = np.zeros((7, len(zn_40)))
            var_prof_440 = np.zeros((7, len(zn_440)))

            for n in range(7):
                print(n)
                if n < 3:
                    if n == 0:
                        path_in = path_MONC_alt_HCs + f'/{2 ** (n)}00m/'
                    else:
                        path_in = path_MONC_alt_HCs + f'/{2 ** (n)}00m/SA/'

                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)


                elif n < 6:

                    path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'
                    filein = f'arm_{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    if n == 4:
                        var_prof_440[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    else:
                        var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)

                else:
                    path_in = path_ARM25
                    filein = f'{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[6, :] = np.mean(ds_in[f'{var}'].data, axis=0)

            plt.plot(figsize=(5, 8))

            plt.plot(var_prof[6, :], zn, 'k', linewidth=2,
                     label='LES $\\Delta$ = 25m')

            for i in range(6):
                # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
                if i < 3:
                    plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='--',
                                 label='$\\Delta$' + f' = {(2 ** (i))}00m')
                else:
                    if i == 4:
                        plt.plot(var_prof_440[i, :], zn_440/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                        # marker='*')
                    else:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                        # marker='*')

            plt.title(f'{clock_time}:'+' $C_s$ prof (dot) vs S-A $C_s$ prof (dash)')
            plt.tight_layout(pad=0.5)
            plt.gcf().set_size_inches(5.5, 7)
            plt.legend(fontsize=13, loc='upper right')

            bottom, top = plt.ylim()
            plt.ylim(0, 1.3)

            plt.xlabel(f'{var}', fontsize=14)
            plt.ylabel('$z$/$z_{CT}$', fontsize=14)

            plt.tight_layout()

            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_SAHCs.png', bbox_inches='tight')
            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_SAHCs.pdf', bbox_inches='tight')
            plt.close()

elif plotting == 'SAHCs_vs_SA_Smag':

    for nv, var in enumerate(var_list):
        print(var)
        for nt, time in enumerate(list_timestamps):

            clock_time_int = 05.30 + int(time) / (60 * 60)
            clock_time = str(clock_time_int) + '0L'

            var_prof = np.zeros((7, len(zn)))
            var_prof_40 = np.zeros((7, len(zn_40)))
            var_prof_440 = np.zeros((7, len(zn_440)))

            for n in range(7):
                print(n)
                Cs_val = ['/', '/Cs_0_11/', '/dz_40m/Cs0_075/']
                if n < 3:
                    path_in = path_MONC_stand + f'{2 ** (n)}00m{Cs_val[n]}'
                    if n == 2:
                        filein = f'arm_{str(time)}.nc'
                        ds_in = xr.open_dataset(path_in + filein)
                        var_prof_40[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    else:
                        filein = f'arm_{str(time)}.nc'
                        ds_in = xr.open_dataset(path_in + filein)
                        var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)


                elif n < 6:

                    if n == 3:
                        path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'
                    else:
                        path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/SA/'

                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)

                    # if n == 4:
                    #     var_prof_440[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    # else:
                    var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)

                else:
                    path_in = path_ARM25
                    filein = f'{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[6, :] = np.mean(ds_in[f'{var}'].data, axis=0)

            plt.plot(figsize=(5, 8))

            plt.plot(var_prof[6, :], zn, 'k', linewidth=2,
                     label='LES $\\Delta$ = 25m')

            for i in range(6):
                # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
                if i < 3:
                    if i == 2:
                        plt.plot(var_prof_40[i, :], zn_40/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                    if i == 1:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle=':')
                else:
                    if i == 4:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='--',
                                 label='$\\Delta$' + f' = {(2 ** (i-3))}00m')
                        # marker='*')
                    else:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='--',
                                 label='$\\Delta$' + f' = {(2 ** (i-3))}00m')
                        # marker='*')

            plt.title(f'{clock_time}: S-A Smag (dot) vs S-A $C_s$ prof (dash)')
            plt.tight_layout(pad=0.5)
            plt.gcf().set_size_inches(5.5, 7)
            plt.legend(fontsize=13, loc='upper right')

            bottom, top = plt.ylim()
            plt.ylim(0, 1.3)
            #plt.ylim(bottom=600, top=3600)
            plt.xlabel(f'{var}', fontsize=14)
            plt.ylabel('$z$/$z_{CT}$', fontsize=14)

            plt.tight_layout()

            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_SA_Smag_vs_SAHCs.png', bbox_inches='tight')
            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_SA_Smag_vs_SAHCs.pdf', bbox_inches='tight')
            plt.close()


elif plotting == 'SAHCs_vs_SAHCsCth_L':


    for nv, var in enumerate(var_list):
        print(var)
        for nt, time in enumerate(list_timestamps):

            clock_time_int = 05.30 + int(time) / (60 * 60)
            clock_time = str(clock_time_int) + '0L'

            var_prof = np.zeros((7, len(zn)))
            var_prof_40 = np.zeros((7, len(zn_40)))
            var_prof_440 = np.zeros((7, len(zn_440)))

            for n in range(7):
                print(n)
                if n < 3:
                    if n == 0:
                        path_in = path_MONC_alt_HCs + '200m/HCth_L/'
                    elif n == 1:
                        path_in = path_MONC_alt_HCs + '400m/HCth_L/'
                    elif n == 2:
                        path_in = path_MONC_alt_HCs + '400m/HCth_L/SA/'


                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)


                elif n < 6:

                    if n == 3:
                        path_in = path_MONC_alt_HCs + '200m/'
                    elif n == 4:
                        path_in = path_MONC_alt_HCs + '400m/'
                    elif n == 5:
                        path_in = path_MONC_alt_HCs + '400m/SA/'

                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)

                    if n == 3:
                        var_prof_440[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                    else:
                        var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)

                else:
                    path_in = path_ARM25
                    filein = f'{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[6, :] = np.mean(ds_in[f'{var}'].data, axis=0)

            plt.plot(figsize=(5, 8))

            plt.plot(var_prof[6, :], zn, 'k', linewidth=2,
                     label='LES $\\Delta$ = 25m')

            for i in range(6):
                # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
                if i < 3:
                    if i == 2:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='-',
                                 label='$\\Delta$' + ' = 400m, S-A $C_s$ & $C_{\\theta_L}$ profs')
                    else:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='--',
                                 label='$\\Delta$' + f' = {(2 ** (i+1))}00m, ' + '$C_s$ & $C_{\\theta_L}$ profs')

                else:
                    if i == 5:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle='-.',
                                 label='$\\Delta$' + ' = 400m, S-A $C_s$ prof')
                    elif i == 4:
                        plt.plot(var_prof[i, :], zn/cloudtop25[nt], colour_cycle[i % 3], linestyle=':',
                                 label='$\\Delta$' + ' = 400m, $C_s$ prof')
                    elif i == 3:
                        plt.plot(var_prof_440[i, :], zn_440, colour_cycle[i % 3], linestyle=':',
                                 label='$\\Delta$' + ' = 200m, $C_s$ prof')
                        # marker='*')

            plt.title(f'{clock_time}:' + ' $C_s$ prof vs $C_s$ & $C_{\\theta_L}$ profs')
            plt.tight_layout(pad=0.5)
            plt.gcf().set_size_inches(5.5, 7)
            plt.legend(fontsize=13, loc='upper right')

            bottom, top = plt.ylim()
            plt.ylim(0, 1.3)
            #plt.ylim(bottom=600, top = 3600)

            plt.xlabel(f'{var}', fontsize=14)
            plt.ylabel('$z$/$z_{CT}$', fontsize=14)

            plt.tight_layout()

            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_SAHCs_vs_SAHCsCth_L.png', bbox_inches='tight')
            plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_SAHCs_vs_SAHCsCth_L.pdf', bbox_inches='tight')
            plt.close()





