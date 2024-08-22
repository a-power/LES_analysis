import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import xarray as xr
import numpy.ma as ma

parser = argparse.ArgumentParser()
parser.add_argument('--case_in', type=str, default='ARM') #BOMEX
parser.add_argument('--times', type=str, default='32400')
args = parser.parse_args()
case = args.case_in
set_time = args.times



path_ARM25 = '/storage/silver/greybls/si818415/arm_2d_25m/diagnostics_ts_'
path_MONC_alt_HCs = '/storage/silver/scenario/si818415/altered_MONC/'
path_MONC_stand = '/storage/silver/scenario/si818415/og_monc/'

plotdir = f'/home/users/si818415/phd/plots/MONC_alt/cloud_fields/'
os.makedirs(plotdir, exist_ok = True)

list_timestamps = [18000, 25200, 32400, 39600]
times = list_timestamps[2]
model_setups = [path_MONC_stand+'400m/', path_MONC_alt_HCs+'400m/', path_MONC_alt_HCs+'400m/SA/',
                path_MONC_alt_HCs+'400m/HCth_L/', path_MONC_alt_HCs+'400m/HCth_L/SA/']

model_setup_names = ['Smag', '$C_s$ prof', '$C_s$ S-A prof', '$C_s$ & $Pr_{\\theta_L}$ prof',
                     '$C_s$ & $Pr_{\\theta_L}$ S-A prof']

model_setup_save = ['Smag', 'Cs', 'Cs_S-A', 'Cs_Pr', 'Cs_Pr']

for nf, file_n in enumerate(model_setups):

    ds_in = xr.open_dataset(file_n + f'arm_3d_{str(times)}.nc')


    print('successfully opened contour set')

    # print('length of time array for cloud field is ',
    #       len(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, 0, 0, 0]))


    cloud_top_field = ds_in['cltop'].data[0,...]

    mask_no_cloud = ma.masked_less_equal(cloud_top_field, 0)

    # elif x_or_y == 'y':
    #     cloud_field = contour_set['f(q_cloud_liquid_mass_on_p)_r'].data[t_set, :, axis_set, ...]
    #     w_field = contour_set['f(w_on_p)_r'].data[t_set, :, axis_set, ...]
    #     w2_field = contour_set['f(w_on_p.w_on_p)_r'].data[t_set, :, axis_set, ...]
    #     th_v_field = contour_set['f(th_v_on_p)_r'].data[t_set, :, axis_set, ...]

    ds_in.close()

    print('beginning plots')

    fig1, ax1 = plt.subplots()
    plt.title(f'{model_setup_names[nf]}' + ' with $\\widehat{\\bar{\\Delta}} = $' + '400m', fontsize=16)

    # mycmap = plt.get_cmap('YlOrRd').copy()
    # mycmap.set_extremes(under='white', over='maroon')

    cf = plt.contourf(np.transpose(mask_no_cloud), extend='max')


    cb = plt.colorbar(cf, format='%.1f')
    cb.set_label(f'Cloud top height', size=16)


    plt.xlabel(f'x (km)', fontsize=16)

    plt.ylabel("y (km)", fontsize=16)
    # plt.xlim(start_grid, end_grid)
    # og_xtic = plt.xticks()
    # plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0])), 1))

    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plt.ylim(0, z_top_in)
    # og_ytic = plt.yticks()
    # plt.yticks(z_tix_in, z_labels_in)  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

    plt.savefig(plotdir + f'Cloud_top_{model_setup_save[nf]}_32400.pdf',
                bbox_inches='tight')
    plt.clf()