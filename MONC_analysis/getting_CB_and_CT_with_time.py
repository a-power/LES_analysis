import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import datetime

plot_choice = 'all_Cs_at_D_200' #'og_HCs'  'HCs_HCsSA'  'all_Cs_at_D_200'

def get_cloud_only(CT_or_CB_field, dist_from_surf_threas=10):

    mask_no_cloud_temp = ma.masked_less_equal(CT_or_CB_field, dist_from_surf_threas)
    mask_no_cloud =np.ma.filled(mask_no_cloud_temp, np.nan)

    return mask_no_cloud




path_ARM25 = '/storage/silver/greybls/si818415/arm_2d_25m/diagnostics_ts_'
path_MONC_alt_HCs = '/storage/silver/scenario/si818415/altered_MONC/'
path_MONC_stand = '/storage/silver/scenario/si818415/og_monc/'

plotdir = '/home/users/si818415/phd/plots/MONC_alt/'

list_timestamps = np.arange(600, 40200, 600)
#[17400, 18000, 24600, 25200, 31800, 32400, 39000, 39600]

#np.ndarray.tolist( np.arange(17400, 40000, 600) )

colour_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
line_list = [':', ':', ':', '--', '--', '--']
model_param = ['Stand', 'Stand', 'Stand', 'HCs', 'HCs', 'HCs'] #'HCs $\\widehat{\\bar{\\Delta}}'

zn = np.arange(-5, 4400, 10)

def get_25m_ref(filein, var, len_ts, stepsize, nt_per_file):

    print(filein)

    ref_tstamps = np.arange(1200, 39600, stepsize)

    test_ds_in = xr.open_dataset(filein + f'{ref_tstamps[0]}.nc')
    len_zn = len(test_ds_in[f'{var}'].data[0, :])

    ref_25m = np.zeros((len_ts, len_zn))

    for ts, time_stamp in enumerate(ref_tstamps):
        ds_in = xr.open_dataset(filein+ f'{time_stamp}.nc')

        for nt in range(nt_per_file):
            ref_25m[nt_per_file*ts + nt, :] = ds_in[f'{var}'].data[nt, :]

    return ref_25m, len_zn


def get_CT_and_CB(ts_of_cloud_frac_prof, len_ts, len_zn_in):

    CT_ref_25m = np.zeros(len_ts)
    CB_ref_25m = np.zeros(len_ts)

    for nt in range(len_ts):
        for i in range(len_zn_in):
            if ts_of_cloud_frac_prof[nt, (len_zn_in-1)-i] >= 0.001:
                CT_ref_25m[nt] = zn[(len_zn_in-1)-i]
            if ts_of_cloud_frac_prof[nt, i] >= 0.001:
                CB_ref_25m[nt] = zn[i]

    CT_ref_25m[CT_ref_25m==0] = np.nan
    CB_ref_25m[CB_ref_25m==0] = np.nan

    #CT_ref_25m = savgol_filter(CT_ref_25m, 5, 3)
    #CB_ref_25m = savgol_filter(CB_ref_25m, 5, 3)

    return CB_ref_25m, CT_ref_25m

plot_ref_tstamps = np.arange(1200, 39600, 60)

file_in_25m = path_ARM25

ts_cloud_prof, len_zn_25 = get_25m_ref(file_in_25m, 'total_cloud_fraction', 640, 1200, 20)

CB_LES_25m, CT_LES_25m = get_CT_and_CB(ts_cloud_prof, 640, len_zn_25)


if plot_choice == 'HCs_HCsSA':

    CB_mean_height_ts = np.zeros( (6, 640))  # len(list_timestamps)) )
    CT_mean_height_ts = np.zeros( (6, 640))  # len(list_timestamps)) )

    for n in range(6):
        if n == 1 or n == 2: #unalt
            # path_in = path_MONC_stand + f'{2 ** n}00m/'
            path_in = path_MONC_alt_HCs + f'{2 ** (n)}00m/SA/'
        elif n == 0:
            path_in = path_MONC_alt_HCs + f'{2 ** (n)}00m/'  # SA/'
        else:
            path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'#SA/'


        ts_cloud_prof, len_zn_out = get_25m_ref(path_in + 'arm_', 'total_cloud_fraction', 640, 600, 10)
        CB_mean_height_ts[n, :], CT_mean_height_ts[n, :] = get_CT_and_CB(ts_cloud_prof, 640, len_zn_out)

        # for nt, time in enumerate(list_timestamps):
        #
        #     filein = f'arm_3d_{str(time)}.nc'
        #
        #     ds_in = xr.open_dataset(path_in + filein)
        #     CB_field = ds_in['clbas'].data
        #     CT_field = ds_in['cltop'].data
        #
        #     CB_cloud_only = get_cloud_only(CB_field)
        #     CT_cloud_only = get_cloud_only(CT_field)
        #
        #     CB_mean_height_ts[n, nt] = np.nanpercentile(CB_cloud_only, 5)
        #     CT_mean_height_ts[n, nt] = np.nanpercentile(CT_cloud_only, 95)




    plt.plot(figsize=(12, 4))

    plt.plot(plot_ref_tstamps, CB_LES_25m, 'k')
    plt.plot(plot_ref_tstamps, CT_LES_25m, 'k', label='25m LES')

    for i in range(6):
        # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
        if i >= 3:
            plt.plot(list_timestamps, CB_mean_height_ts[i, :], colour_cycle[i % 3], linewidth=1.5)
            plt.plot(list_timestamps, CT_mean_height_ts[i, :], colour_cycle[i % 3], linewidth=1.5,
                     label=f'$\\Delta$ = {2 ** ((i-3))}00m')
    for i in range(3):
            plt.plot(list_timestamps, CB_mean_height_ts[i, :], colour_cycle[i % 3], linestyle='--', marker='x')
            plt.plot(list_timestamps, CT_mean_height_ts[i, :], colour_cycle[i % 3], linestyle='--', marker='x')

    plt.tight_layout(pad=0.5)
    plt.gcf().set_size_inches(10, 5.5)
    plt.legend(fontsize=13, loc='upper left')


    bottom, top = plt.ylim()
    plt.ylim(bottom=0, top = 4000)


    # time_label_temp = "%.2f"%(05.50 + og_xtic[0]/(60*60))
    # time_label_temp_min = (( np.round(05.50 + og_xtic[0]/(60*60), 2 ) - "%.2f"%(05.50 + og_xtic[0]/(60*60)) )*60 )/100
    # time_label = np.round(time_label_temp + time_label_temp_min, 2)

    time_label = []

    x_tick_loc = np.arange(32400-19800, 61200-19800, 3600)

    for i in range(len(x_tick_loc)):
        print(x_tick_loc[i])
        time_label.append(datetime.timedelta(seconds=(int(x_tick_loc[i]) + 19800)))

    og_xtic = plt.xticks()
    print(og_xtic)

    plt.xticks(x_tick_loc, time_label)
    plt.xlim(32400 - 19800, 61200 - 19800)

    plt.xlabel('Local time (hh:mm:ss)', fontsize=14)
    plt.ylabel('z (m)', fontsize=14)
    plt.title('Cloud top and base height for HCs (solid) vs HCsSA (x)', fontsize=14)

    plt.tight_layout()

    plt.savefig(plotdir+f'ARM_cloud_top_and_base_ts_HCs_HCsSA.png', bbox_inches='tight')
    plt.savefig(plotdir + f'ARM_cloud_top_and_base_ts_HCs_HCsSA.pdf', bbox_inches='tight')
    plt.close()


elif plot_choice == 'og_HCs':

    CB_mean_height_ts = np.zeros( (6, 640))  # len(list_timestamps)) )
    CT_mean_height_ts = np.zeros( (6, 640))  # len(list_timestamps)) )

    for n in range(6):
        if n < 3: #unalt
            path_in = path_MONC_stand + f'{2 ** n}00m/'
        else:
            path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'

        ts_cloud_prof, len_zn_out = get_25m_ref(path_in + 'arm_', 'total_cloud_fraction', 640, 600, 10)

        CB_mean_height_ts[n, :], CT_mean_height_ts[n, :] = get_CT_and_CB(ts_cloud_prof, 640, len_zn_out)

        # for nt, time in enumerate(list_timestamps):
        #
        #     filein = f'arm_3d_{str(time)}.nc'
        #
        #     ds_in = xr.open_dataset(path_in + filein)
        #     CB_field = ds_in['clbas'].data
        #     CT_field = ds_in['cltop'].data
        #
        #     CB_cloud_only = get_cloud_only(CB_field)
        #     CT_cloud_only = get_cloud_only(CT_field)
        #
        #     CB_mean_height_ts[n, nt] = np.nanpercentile(CB_cloud_only, 5)
        #     CT_mean_height_ts[n, nt] = np.nanpercentile(CT_cloud_only, 95)




    plt.plot(figsize=(12, 4))

    plt.plot(plot_ref_tstamps, CB_LES_25m, 'k')
    plt.plot(plot_ref_tstamps, CT_LES_25m, 'k', label='25m LES')

    for i in range(6):
        # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
        if i < 3:
            plt.plot(list_timestamps, CB_mean_height_ts[i, :], colour_cycle[i % 3],
                     linestyle=':', marker='*')
            plt.plot(list_timestamps, CT_mean_height_ts[i, :], colour_cycle[i % 3],
                     linestyle=':', marker='*')

        else:
            plt.plot(list_timestamps, CB_mean_height_ts[i, :], colour_cycle[i % 3], linewidth=2.5)
            plt.plot(list_timestamps, CT_mean_height_ts[i, :], colour_cycle[i % 3], linewidth=2.5,
                     label=f'$\\Delta$ = {2 ** ((i-3))}00m')

    plt.tight_layout(pad=0.5)
    plt.gcf().set_size_inches(10, 5.5)
    plt.legend(fontsize=13, loc='upper left')


    bottom, top = plt.ylim()
    plt.ylim(bottom=0, top = 4000)

    time_label = []

    x_tick_loc = np.arange(32400-19800, 61200-19800, 3600)

    for i in range(len(x_tick_loc)):
        print(x_tick_loc[i])
        time_label.append(datetime.timedelta(seconds=(int(x_tick_loc[i]) + 19800)))

    og_xtic = plt.xticks()
    print(og_xtic)

    plt.xticks(x_tick_loc, time_label)
    plt.xlim(32400 - 19800, 61200 - 19800)

    plt.xlabel('Local time (hh:mm:ss)', fontsize=14)
    plt.ylabel('z (m)', fontsize=14)
    plt.title('Cloud top and base height for HCs (solid) vs SCs0.23 (star)', fontsize=14)

    plt.tight_layout()

    plt.savefig(plotdir+f'ARM_cloud_top_and_base_ts_og_HCs.png', bbox_inches='tight') #_HCsSA
    plt.savefig(plotdir + f'ARM_cloud_top_and_base_ts_og_HCs.pdf', bbox_inches='tight')
    plt.close()



elif plot_choice == 'all_Cs_at_D_200':

    plt.plot(plot_ref_tstamps, CB_LES_25m, 'k')#, linewidth=2)
    plt.plot(plot_ref_tstamps, CT_LES_25m, 'k', label='25m LES')

    # CB_mean_height_ts = np.zeros( (7, len(list_timestamps)) )
    # CT_mean_height_ts = np.zeros( (7, len(list_timestamps)) )

    CB_mean_height_ts = np.zeros( (7, 640) )
    CT_mean_height_ts = np.zeros( (7, 640) )

    for n in range(6):
        if n == 0: #unalt
            path_in = path_MONC_stand + f'200m/'
        elif n == 1:
            path_in = path_MONC_stand + f'200m/Cs_0_137/'
        elif n == 2:
            path_in = path_MONC_stand + f'200m/Cs_0_11/'
        elif n == 3:
            path_in = path_MONC_alt_HCs + f'200m/'
        elif n == 4:
            path_in = path_MONC_alt_HCs + f'200m/SA/'
        elif n == 5:
            path_in = path_MONC_alt_HCs + '200m/HCth_L/'

        # for nt, time in enumerate(list_timestamps):

        if n == 5:
            for nt, time in enumerate(list_timestamps):
                if time > 34800:
                        CB_mean_height_ts[n, nt] = np.nan
                        CT_mean_height_ts[n, nt] = np.nan
                else:
                    filein = f'arm_3d_{str(time)}.nc'

                    ds_in = xr.open_dataset(path_in + filein)
                    CB_field = ds_in['clbas'].data
                    CT_field = ds_in['cltop'].data

                    CB_cloud_only = get_cloud_only(CB_field)
                    CT_cloud_only = get_cloud_only(CT_field)

                    CB_mean_height_ts[n, nt] = np.nanpercentile(CB_cloud_only, 5)
                    CT_mean_height_ts[n, nt] = np.nanpercentile(CT_cloud_only, 95)
        else:


            # filein = f'arm_3d_{str(time)}.nc'

            ts_cloud_prof, len_zn_out = get_25m_ref(path_in + 'arm_', 'total_cloud_fraction', 640, 600, 10)

            CB_mean_height_ts[n, :], CT_mean_height_ts[n, :] = get_CT_and_CB(ts_cloud_prof, 640, len_zn_out)

            # CB_field = ds_in['clbas'].data
            # CT_field = ds_in['cltop'].data
            #
            # CB_cloud_only = get_cloud_only(CB_field)
            # CT_cloud_only = get_cloud_only(CT_field)
            #
            # CB_mean_height_ts[n, nt] = np.nanpercentile(CB_cloud_only, 5)
            # CT_mean_height_ts[n, nt] = np.nanpercentile(CT_cloud_only, 95)




    plt.plot(figsize=(12, 4))
    for i in range(6):
        # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
        if i == 0:
             print(f'S$C_s$0.23:, len(plot_ref_tstamps) = {len(plot_ref_tstamps)}, and len(CB_mean_height_ts = {CB_mean_height_ts[i, :]}')
        #     plt.plot(plot_ref_tstamps, CB_mean_height_ts[i, :], colour_cycle[0], linestyle='--')#, marker='*')
        #     plt.plot(plot_ref_tstamps, CT_mean_height_ts[i, :], colour_cycle[0], linestyle='--',
        #              label='S$C_s$0.23') #f'$\\Delta$ = {2 ** ((i-3))}00m'
        # elif i == 1:
        #     plt.plot(plot_ref_tstamps, CB_mean_height_ts[i, :], colour_cycle[1], linestyle='--')#, marker='*')
        #     plt.plot(plot_ref_tstamps, CT_mean_height_ts[i, :], colour_cycle[1], linestyle='--',
        #              label='S$C_s$0.137')
        elif i == 2:
            plt.plot(plot_ref_tstamps, CB_mean_height_ts[i, :], colour_cycle[2], linestyle='--')#, marker='*')
            plt.plot(plot_ref_tstamps, CT_mean_height_ts[i, :], colour_cycle[2], linestyle='--',
                     label='S$C_s$0.11')
        # elif i == 3:
        #     plt.plot(plot_ref_tstamps, CB_mean_height_ts[i, :], colour_cycle[3]) #, linestyle='--')#, linewidth=2)
        #     plt.plot(plot_ref_tstamps, CT_mean_height_ts[i, :], colour_cycle[3],
        #              label='H$C_s$')
        elif i == 4:
            plt.plot(plot_ref_tstamps, CB_mean_height_ts[i, :], colour_cycle[4]) #, linestyle='--')#, marker='x')
            plt.plot(plot_ref_tstamps, CT_mean_height_ts[i, :], colour_cycle[4],
                     label='H$C_s$SA')
        # elif i == 5:
        #     plt.plot(list_timestamps, CB_mean_height_ts[i, :len(list_timestamps)], colour_cycle[5]) #, linestyle='--')#, marker='^')
        #     plt.plot(list_timestamps, CT_mean_height_ts[i, :len(list_timestamps)], colour_cycle[5],
        #              label='H$C_sC_{\\theta_L}$')

    plt.tight_layout(pad=0.5)
    plt.gcf().set_size_inches(10, 5.5)
    plt.legend(fontsize=13, loc='upper left')


    bottom, top = plt.ylim()
    plt.ylim(bottom=0, top = 4000)

    time_label = []

    x_tick_loc = np.arange(32400-19800, 61200-19800, 3600)

    for i in range(len(x_tick_loc)):
        print(x_tick_loc[i])
        time_label.append(datetime.timedelta(seconds=(int(x_tick_loc[i]) + 19800)))

    og_xtic = plt.xticks()
    print(og_xtic)

    plt.xticks(x_tick_loc, time_label)
    plt.xlim(32400 - 19800, 61200 - 19800)

    plt.xlabel('Local time (hh:mm:ss)', fontsize=14)
    plt.ylabel('z (m)', fontsize=14)
    plt.title('Cloud top and base height for HCs (solid) vs SCs0.23 (star)', fontsize=14)

    plt.tight_layout()

    save_name = 'ARM_cloud_top_and_base_ts_D_200_Cs0_11_vs_HCsSA_cases'
        #'ARM_cloud_top_and_base_ts_D_200_all_Cs_cases'

    plt.savefig(plotdir+f'{save_name}.png', bbox_inches='tight') #_HCsSA
    plt.savefig(plotdir + f'{save_name}.pdf', bbox_inches='tight')
    plt.close()