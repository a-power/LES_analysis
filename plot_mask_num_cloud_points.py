import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn
import mask_cloud_vs_env as clo
import matplotlib.ticker as mtick


np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')


dir_data = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
dir_contour = dir_data + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots'
os.makedirs(plotdir, exist_ok = True)

cloud_thres = [0, 1e-5]


def count_mask(mask_in):

    counter = np.zeros((np.shape(mask_in)[0], np.shape(mask_in)[-1]))

    for nt in range(np.shape(mask_in)[0]):
        for i in range(np.shape(mask_in)[-1]):

            it = 0
            for j in range(np.shape(mask_in)[1]):
                for k in range(np.shape(mask_in)[2]):
                    if mask_in[nt, j, k, i] == False:
                        it += 1
            counter[nt, i] = it

    return counter


data_2D = dir_data + 'ga00.nc'
data_4D = dir_data + 'ga01.nc'
data_8D = dir_data + 'ga02.nc'
data_16D = dir_data + 'ga03.nc'
data_32D = dir_data + 'ga04.nc'
data_64D = dir_data + 'ga05.nc'

z = np.arange(0, 3020, 20)
z_i = 490

#index of 0 at the start is to get rid of the dummy time index thats required to save the files
for iters in range(len(cloud_thres)):

    Cth_cloud_2, env_mask = clo.cloud_vs_env_masks(data_2D, cloud_liquid_threshold=cloud_thres[0])

    cloud_count_2 = count_mask(Cth_cloud_2)
    print('finished cloud count 2')

    Cth_cloud_4, env_mask = clo.cloud_vs_env_masks(data_4D, cloud_liquid_threshold=cloud_thres[0])
    Cth_cloud_8, env_mask = clo.cloud_vs_env_masks(data_8D, cloud_liquid_threshold=cloud_thres[0])
    Cth_cloud_16, env_mask = clo.cloud_vs_env_masks(data_16D, cloud_liquid_threshold=cloud_thres[0])
    Cth_cloud_32, env_mask = clo.cloud_vs_env_masks(data_32D, cloud_liquid_threshold=cloud_thres[0])
    Cth_cloud_64, env_mask = clo.cloud_vs_env_masks(data_64D, cloud_liquid_threshold=cloud_thres[0])



    ################################################################



    cloud_count_4 = count_mask(Cth_cloud_4)
    print('finished cloud count 4')
    cloud_count_8 = count_mask(Cth_cloud_8)
    print('finished cloud count 8')
    cloud_count_16 = count_mask(Cth_cloud_16)
    print('finished cloud count 16')
    cloud_count_32 = count_mask(Cth_cloud_32)
    print('finished cloud count 32')
    cloud_count_64 = count_mask(Cth_cloud_64)
    print('finished cloud count 64')

    total_grid = 640000

    # for t_in in range(3):
    #
    #     plt.figure(figsize=(6,7))
    #     plt.plot(-26, -29)
    #     plt.plot(cloud_count_2[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 40}$m')
    #     plt.plot(cloud_count_4[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 80}$m')
    #     plt.plot(cloud_count_8[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 160}$m')
    #     plt.plot(cloud_count_16[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 320}$m')
    #     plt.plot(cloud_count_32[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 640}$m')
    #     plt.plot(cloud_count_64[t_in, :]/total_grid, z/z_i, label = '$\\Delta = 1280}$m')
    #     plt.xlabel(f"Ratio of 'cloudy' vs 'non-cloudy' grid points, time = {t_in}", fontsize=16)
    #     plt.ylabel("z/z$_{ML}$", fontsize=16)
    #     plt.legend(fontsize=12, loc='upper right')
    #     plt.xlim(0, 0.5)
    #     plt.ylim(bottom = 0)
    #     plt.savefig(plotdir+f'cloud_count_prof_scaled_t{t_in}.png', pad_inches=0)
    #     plt.close()
    #
    # print('finished cloud mask plot')

    plt.figure(figsize=(6,7))
    plt.plot(-26, -29)
    plt.plot(np.mean(cloud_count_2, axis=0)/total_grid, z/z_i, label = '$\\Delta = 40}$m')
    plt.plot(np.mean(cloud_count_4, axis=0)/total_grid, z/z_i, label = '$\\Delta = 80}$m')
    plt.plot(np.mean(cloud_count_8, axis=0)/total_grid, z/z_i, label = '$\\Delta = 160}$m')
    plt.plot(np.mean(cloud_count_16, axis=0)/total_grid, z/z_i, label = '$\\Delta = 320}$m')
    plt.plot(np.mean(cloud_count_32, axis=0)/total_grid, z/z_i, label = '$\\Delta = 640}$m')
    plt.plot(np.mean(cloud_count_64, axis=0)/total_grid, z/z_i, label = '$\\Delta = 1280}$m')

    plt.xlabel(f"Ratio of 'cloudy' vs 'non-cloudy' grid points", fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(0, 0.5)
    plt.ylim(0, 6)
    plt.savefig(plotdir+f'cloudy_vs_env_ratio_prof_cloud={cloud_thres}_t_av.png', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6,7))
    plt.plot(-26, -29)
    plt.plot(np.mean(cloud_count_2, axis=0), z/z_i, label = '$\\Delta = 40}$m')
    plt.plot(np.mean(cloud_count_4, axis=0), z/z_i, label = '$\\Delta = 80}$m')
    plt.plot(np.mean(cloud_count_8, axis=0), z/z_i, label = '$\\Delta = 160}$m')
    plt.plot(np.mean(cloud_count_16, axis=0), z/z_i, label = '$\\Delta = 320}$m')
    plt.plot(np.mean(cloud_count_32, axis=0), z/z_i, label = '$\\Delta = 640}$m')
    plt.plot(np.mean(cloud_count_64, axis=0), z/z_i, label = '$\\Delta = 1280}$m')

    plt.xlabel(f"number of 'cloudy' grid points", fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(0, 0.5)
    plt.ylim(0, 6)
    plt.savefig(plotdir+f'cloud_count_prof_cloud={cloud_thres}_t_av.png', pad_inches=0)
    plt.close()




# mean_mask_2 = np.ma.masked_where(cloud_count_2==0 , cloud_count_2)
# mean_mask_4 = np.ma.masked_where(cloud_count_4==0 , cloud_count_4)
# mean_mask_8 = np.ma.masked_where(cloud_count_8==0 , cloud_count_8)
# mean_mask_16 = np.ma.masked_where(cloud_count_16==0 , cloud_count_16)
# mean_mask_32 = np.ma.masked_where(cloud_count_32==0 , cloud_count_32)
# mean_mask_64 = np.ma.masked_where(cloud_count_64==0 , cloud_count_64)
#
#
# plt.figure(figsize=(6,7))
# plt.plot(-26, -29)
# plt.semilogx(np.mean(mean_mask_2, axis=0), z/z_i, label = '$\\Delta = 40}$m')
# plt.semilogx(np.mean(mean_mask_4, axis=0), z/z_i, label = '$\\Delta = 80}$m')
# plt.semilogx(np.mean(mean_mask_8, axis=0), z/z_i, label = '$\\Delta = 160}$m')
# plt.semilogx(np.mean(mean_mask_16, axis=0), z/z_i, label = '$\\Delta = 320}$m')
# plt.semilogx(np.mean(mean_mask_32, axis=0), z/z_i, label = '$\\Delta = 640}$m')
# plt.semilogx(np.mean(mean_mask_64, axis=0), z/z_i, label = '$\\Delta = 1280}$m')
# plt.xlabel(f"Number of Grid Points with 'Cloud', time averaged", fontsize=16)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# plt.ylim(bottom = 0)
# plt.savefig(plotdir+f'cloud_count_prof_logx_scaled_t_av.png', pad_inches=0)
# plt.close()
