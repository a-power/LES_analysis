import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn
import mask_cloud_vs_env as clo

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

dir_data = '/work/scratch-pw/apower/20m_gauss_dyn/q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/plots/coarse_data/cloud_v_env/'
os.makedirs(plotdir, exist_ok = True)

def count_mask(mask_in):

    counter = np.zeros((np.shape(mask_in)[0], np.shape(mask_in)[-1]))

    for nt in range(np.shape(mask_in)[0]):
        for i in range(np.shape(mask_in)[-1]):
            counter[nt, i] = mask_in[nt, :, :, i].count

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

Cth_cloud_2, env_mask = clo.cloud_vs_env_masks(data_2D)
Cth_cloud_4, env_mask = clo.cloud_vs_env_masks(data_4D)
Cth_cloud_8, env_mask = clo.cloud_vs_env_masks(data_8D)
Cth_cloud_16, env_mask = clo.cloud_vs_env_masks(data_16D)
Cth_cloud_32, env_mask = clo.cloud_vs_env_masks(data_32D)
Cth_cloud_64, env_mask = clo.cloud_vs_env_masks(data_64D)



################################################################


cloud_count_2 = count_mask(Cth_cloud_2)
print('finished cloud count 2')
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



plt.figure(figsize=(6,7))
plt.plot(-26, -29, label = '$\\Delta = 20$m')
plt.plot(cloud_count_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(cloud_count_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(cloud_count_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(cloud_count_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(cloud_count_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(cloud_count_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel("Number of Grid Points with 'Cloud'", fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.savefig(plotdir+'cloud_count_prof_scaled.png', pad_inches=0)
plt.close()

print('finished cloud mask plot')


