
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



if case == 'ARM':

    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/MONC_profiles/'
    prof_file = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_ts_' #diagnostics_ts_18000.nc
    field_file = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_3d_ts_'

    zn_set = np.arange(0, 4410, 10)
    z_set = np.arange(-5, 4405, 10)

    set_time = ['18000', '25200', '32400', '39600']


elif case == 'BOMEX':

    plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/BOMEX/MONC_profiles/'
    todd_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
    prof_file = todd_dir + 'BOMEX_m0020_g0800_all_' #'BOMEX_m0020_g0800_all_14400.nc'
    field_file = todd_dir + 'BOMEX_m0020_g0800_all_'

    zn_set = np.arange(0, 3020, 20)
    z_set = np.arange(-10, 3010, 20)

    set_time = ['14400']

else:
    print('need to def case')

os.makedirs(plotdir, exist_ok = True)


#######################################################################################################################


def get_cloud_wth_profs(file_path, time_stamp=-1):

    prof_data = xr.open_dataset(file_path)

    wth_prof_raw = prof_data['wtheta_cn_mean'].data[...]
    th_prof_raw = prof_data['theta_mean'].data[...]
    cloud_prof_raw = prof_data['total_cloud_fraction'].data[...]
    zn_out = prof_data['zn'].data[...] # timeless parameter?

    if time_stamp == 'mean':
        wth_prof_out = np.mean(wth_prof_raw, 0)
        cloud_prof_out = np.mean(cloud_prof_raw, 0)
        th_prof_out = np.mean(th_prof_raw, 0)

    elif time_stamp == None:
        wth_prof_out = wth_prof_raw
        cloud_prof_out = cloud_prof_raw
        th_prof_out = th_prof_raw

    else:
        wth_prof_out = wth_prof_raw[time_stamp, ...]
        cloud_prof_out = cloud_prof_raw[time_stamp, ...]
        th_prof_out = th_prof_raw[time_stamp, ...]

    wth_prof_list = wth_prof_out.tolist()
    z_ML_ind = wth_prof_list.index(np.min(wth_prof_out))
    z_ML = zn_out[z_ML_ind]


    return wth_prof_out, th_prof_out, cloud_prof_out, zn_out, z_ML



def get_w_max_profs(field_path_in, time_stamp=-1):

    field_data = xr.open_dataset(field_path_in)

    w = field_data['w'].data[...]
    w_max_prof = np.zeros(len(w[0,0,0,:]))

    if time_stamp == None:
        for z in range(len(w_max_prof)):
            w_max_prof[z] = np.max(w[...,z])
        print('calced w max for time_stamp = None' )

    else:
        for z in range(len(w_max_prof)):
            w_max_prof[z] = np.max(w[..., z])
            print(f'calced w max for time_stamp = {time_stamp}')

    print('shape of the w_max_prof is ', np.shape(w_max_prof))

    return w_max_prof




def plot_MONC_profs(file_path, field_path, times, time_stamp_in='mean'):


    colours = ['tab:red', 'black', 'tab:green', 'tab:blue', 'tab:purple',
               'tab:cyan', 'tab:gray', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:orange']


    if len(times) == 1:
        fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(13, 4))
    else:
        fig, ax = plt.subplots(nrows=4, ncols=len(times), sharey=False, figsize=(14,19))

    fig.tight_layout(pad=0.5)

    setleft0 = 0
    setleft1 = 0
    setleft2 = 300
    setleft3 = 0
    setright0 = 0
    setright1 = 0
    setright2 = 0
    setright3 = 0


    for it, time_in in enumerate(times):

        file_in = file_path + f'{time_in}.nc'
        field_in = field_path + f'{time_in}.nc'

        wth_prof,  th_prof, cloud_prof, z, z_i = get_cloud_wth_profs(file_in, time_stamp=time_stamp_in)
        w_max_prof_in = get_w_max_profs(field_in, time_stamp=time_stamp_in)

        clock_time_int = 05.30 + int(time_in) / (60 * 60)
        clock_time = str(clock_time_int) + '0L'

        if len(times) == 1:
            ax[0].plot(wth_prof, z / z_i, color='black')
            ax[0].set_xlabel("$ \\overline{w' \\theta} (K m s^{-1})$", fontsize=16)
            ax[0].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)
            ax[0].set_xticks(ax[0].get_xticks()[::2])

            ax[1].plot(cloud_prof*100, z / z_i, color='black')
            ax[1].set_xlabel('cloud cover (%)', fontsize=16)

            ax[2].plot(th_prof, z / z_i, color='black')
            ax[2].set_xlabel("$ \\overline{\\theta}$ (K)", fontsize=16)

            ax[3].plot(w_max_prof_in, z / z_i, color='black')
            ax[3].set_xlabel("$ w'_{max}$ (m s^{-1})", fontsize=16)

        else:
            ax[0, it].plot(wth_prof, z / z_i, color='black')
            ax[0, it].set_xlabel("$ \\overline{w' \\theta'}$ at "  + clock_time, fontsize=16)
            ax[0, it].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)

            left0, right0 = ax[0, it].set_xlim()
            if left0 < setleft0:
                setleft0 = left0
            if right0 > setright0:
                setright0 = right0

            ax[1, it].plot(cloud_prof * 100, z / z_i, color='black')
            ax[1, it].set_xlabel('cloud cover (%) at '  + clock_time, fontsize=16)
            ax[1, it].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)

            left1, right1 = ax[1, it].set_xlim()
            if left1 < setleft1:
                setleft1 = left1
            if right1 > setright1:
                setright1 = right1

            ax[2, it].plot(th_prof, z / z_i, color='black')
            ax[2, it].set_xlabel("$ \\overline{\\theta'}$ at "  + clock_time, fontsize=16)
            ax[2, it].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)

            left2, right2 = ax[2, it].set_xlim()
            if left2 < setleft2:
                setleft2 = left2
            if right2 > setright2:
                setright2 = right2

            ax[3, it].plot(w_max_prof_in, z / z_i, color='black')
            ax[3, it].set_xlabel("$ w'_{max}$", fontsize=16)
            ax[3, it].set_ylabel("z/z$_{ML}$ (z$_{ML}$ = " + str(int(z_i)) + "m)", fontsize=16)

            left3, right3 = ax[3, it].set_xlim()
            if left3 < setleft3:
                setleft3 = left3
            if right3 > setright3:
                setright3 = right3

    if len(times) != 1:
        for itn in range(len(times)):
            ax[0, itn].set_xlim(right=setright0, left=setleft0)
            ax[1, itn].set_xlim(right=setright1, left=setleft1)
            ax[2, itn].set_xlim(right=setright2, left=setleft2)
            ax[3, itn].set_xlim(right=setright3, left=setleft3)
            ax[0, itn].set_xticks(ax[0, itn].get_xticks()[::2])


    plt.tight_layout()


    plt.savefig(plotdir + f'cloud_wth_th_wmax_profs.pdf', bbox_inches='tight')

    plt.close()



plot_MONC_profs(prof_file, field_file, set_time)