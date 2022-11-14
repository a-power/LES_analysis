import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

dir_data = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_cloud_v_env_'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/plots/coarse_data/cloud_v_env/'
os.makedirs(plotdir, exist_ok = True)

def count_mask(mask_in):

    counter = np.zeros(np.shape(mask_in)[0], np.shape(mask_in)[-1])

    for nt in range(np.shape(mask_in)[0]):
        for i in range(np.shape(mask_in)[-1]):
            counter[nt, i] = mask_in[nt, :, :, i]

    return counter


data_2D = xr.open_dataset(dir_data + '2D.nc')
data_4D = xr.open_dataset(dir_data + '4D.nc')
data_8D = xr.open_dataset(dir_data + '8D.nc')
data_16D = xr.open_dataset(dir_data + '16D.nc')
data_32D = xr.open_dataset(dir_data + '32D.nc')
data_64D = xr.open_dataset(dir_data + '64D.nc')


z = np.arange(0, 3020, 20)
z_i = 490

#index of 0 at the start is to get rid of the dummy time index thats required to save the files

Cs_cloud_2 = data_2D['Cs_cloud_prof'].data[0, ...]
Cs_cloud_4 = data_4D['Cs_cloud_prof'].data[0, ...]
Cs_cloud_8 = data_8D['Cs_cloud_prof'].data[0, ...]
Cs_cloud_16 = data_16D['Cs_cloud_prof'].data[0, ...]
Cs_cloud_32 = data_32D['Cs_cloud_prof'].data[0, ...]
Cs_cloud_64 = data_64D['Cs_cloud_prof'].data[0, ...]

Cth_cloud_2 = data_2D['Cth_cloud_prof'].data[0, ...]
Cth_cloud_4 = data_4D['Cth_cloud_prof'].data[0, ...]
Cth_cloud_8 = data_8D['Cth_cloud_prof'].data[0, ...]
Cth_cloud_16 = data_16D['Cth_cloud_prof'].data[0, ...]
Cth_cloud_32 = data_32D['Cth_cloud_prof'].data[0, ...]
Cth_cloud_64 = data_64D['Cth_cloud_prof'].data[0, ...]

Cqt_cloud_2 = data_2D['Cqt_cloud_prof'].data[0, ...]
Cqt_cloud_4 = data_4D['Cqt_cloud_prof'].data[0, ...]
Cqt_cloud_8 = data_8D['Cqt_cloud_prof'].data[0, ...]
Cqt_cloud_16 = data_16D['Cqt_cloud_prof'].data[0, ...]
Cqt_cloud_32 = data_32D['Cqt_cloud_prof'].data[0, ...]
Cqt_cloud_64 = data_64D['Cqt_cloud_prof'].data[0, ...]


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


###############################################################


Cs_env_2 = data_2D['Cs_env_prof'].data[0, ...]
Cs_env_4 = data_4D['Cs_env_prof'].data[0, ...]
Cs_env_8 = data_8D['Cs_env_prof'].data[0, ...]
Cs_env_16 = data_16D['Cs_env_prof'].data[0, ...]
Cs_env_32 = data_32D['Cs_env_prof'].data[0, ...]
Cs_env_64 = data_64D['Cs_env_prof'].data[0, ...]

Cth_env_2 = data_2D['Cth_env_prof'].data[0, ...]
Cth_env_4 = data_4D['Cth_env_prof'].data[0, ...]
Cth_env_8 = data_8D['Cth_env_prof'].data[0, ...]
Cth_env_16 = data_16D['Cth_env_prof'].data[0, ...]
Cth_env_32 = data_32D['Cth_env_prof'].data[0, ...]
Cth_env_64 = data_64D['Cth_env_prof'].data[0, ...]

Cqt_env_2 = data_2D['Cqt_env_prof'].data[0, ...]
Cqt_env_4 = data_4D['Cqt_env_prof'].data[0, ...]
Cqt_env_8 = data_8D['Cqt_env_prof'].data[0, ...]
Cqt_env_16 = data_16D['Cqt_env_prof'].data[0, ...]
Cqt_env_32 = data_32D['Cqt_env_prof'].data[0, ...]
Cqt_env_64 = data_64D['Cqt_env_prof'].data[0, ...]


########################################################################################################################


LijMij_cloud_2 = data_2D['LijMij_cloud_prof'].data[0, ...]
LijMij_cloud_4 = data_4D['LijMij_cloud_prof'].data[0, ...]
LijMij_cloud_8 = data_8D['LijMij_cloud_prof'].data[0, ...]
LijMij_cloud_16 = data_16D['LijMij_cloud_prof'].data[0, ...]
LijMij_cloud_32 = data_32D['LijMij_cloud_prof'].data[0, ...]
LijMij_cloud_64 = data_64D['LijMij_cloud_prof'].data[0, ...]

HjRj_th_cloud_2 = data_2D['HjRj_th_cloud_prof'].data[0, ...]
HjRj_th_cloud_4 = data_4D['HjRj_th_cloud_prof'].data[0, ...]
HjRj_th_cloud_8 = data_8D['HjRj_th_cloud_prof'].data[0, ...]
HjRj_th_cloud_16 = data_16D['HjRj_th_cloud_prof'].data[0, ...]
HjRj_th_cloud_32 = data_32D['HjRj_th_cloud_prof'].data[0, ...]
HjRj_th_cloud_64 = data_64D['HjRj_th_cloud_prof'].data[0, ...]

HjRj_qt_cloud_2 = data_2D['HjRj_qt_cloud_prof'].data[0, ...]
HjRj_qt_cloud_4 = data_4D['HjRj_qt_cloud_prof'].data[0, ...]
HjRj_qt_cloud_8 = data_8D['HjRj_qt_cloud_prof'].data[0, ...]
HjRj_qt_cloud_16 = data_16D['HjRj_qt_cloud_prof'].data[0, ...]
HjRj_qt_cloud_32 = data_32D['HjRj_qt_cloud_prof'].data[0, ...]
HjRj_qt_cloud_64 = data_64D['HjRj_qt_cloud_prof'].data[0, ...]

###############################################################


MijMij_cloud_2 = data_2D['MijMij_cloud_prof'].data[0, ...]
MijMij_cloud_4 = data_4D['MijMij_cloud_prof'].data[0, ...]
MijMij_cloud_8 = data_8D['MijMij_cloud_prof'].data[0, ...]
MijMij_cloud_16 = data_16D['MijMij_cloud_prof'].data[0, ...]
MijMij_cloud_32 = data_32D['MijMij_cloud_prof'].data[0, ...]
MijMij_cloud_64 = data_64D['MijMij_cloud_prof'].data[0, ...]

RjRj_th_cloud_2 = data_2D['RjRj_th_cloud_prof'].data[0, ...]
RjRj_th_cloud_4 = data_4D['RjRj_th_cloud_prof'].data[0, ...]
RjRj_th_cloud_8 = data_8D['RjRj_th_cloud_prof'].data[0, ...]
RjRj_th_cloud_16 = data_16D['RjRj_th_cloud_prof'].data[0, ...]
RjRj_th_cloud_32 = data_32D['RjRj_th_cloud_prof'].data[0, ...]
RjRj_th_cloud_64 = data_64D['RjRj_th_cloud_prof'].data[0, ...]

RjRj_qt_cloud_2 = data_2D['RjRj_qt_cloud_prof'].data[0, ...]
RjRj_qt_cloud_4 = data_4D['RjRj_qt_cloud_prof'].data[0, ...]
RjRj_qt_cloud_8 = data_8D['RjRj_qt_cloud_prof'].data[0, ...]
RjRj_qt_cloud_16 = data_16D['RjRj_qt_cloud_prof'].data[0, ...]
RjRj_qt_cloud_32 = data_32D['RjRj_qt_cloud_prof'].data[0, ...]
RjRj_qt_cloud_64 = data_64D['RjRj_qt_cloud_prof'].data[0, ...]

###############################################################


MijMij_env_2 = data_2D['MijMij_env_prof'].data[0, ...]
MijMij_env_4 = data_4D['MijMij_env_prof'].data[0, ...]
MijMij_env_8 = data_8D['MijMij_env_prof'].data[0, ...]
MijMij_env_16 = data_16D['MijMij_env_prof'].data[0, ...]
MijMij_env_32 = data_32D['MijMij_env_prof'].data[0, ...]
MijMij_env_64 = data_64D['MijMij_env_prof'].data[0, ...]

RjRj_th_env_2 = data_2D['RjRj_th_env_prof'].data[0, ...]
RjRj_th_env_4 = data_4D['RjRj_th_env_prof'].data[0, ...]
RjRj_th_env_8 = data_8D['RjRj_th_env_prof'].data[0, ...]
RjRj_th_env_16 = data_16D['RjRj_th_env_prof'].data[0, ...]
RjRj_th_env_32 = data_32D['RjRj_th_env_prof'].data[0, ...]
RjRj_th_env_64 = data_64D['RjRj_th_env_prof'].data[0, ...]

RjRj_qt_env_2 = data_2D['RjRj_qt_env_prof'].data[0, ...]
RjRj_qt_env_4 = data_4D['RjRj_qt_env_prof'].data[0, ...]
RjRj_qt_env_8 = data_8D['RjRj_qt_env_prof'].data[0, ...]
RjRj_qt_env_16 = data_16D['RjRj_qt_env_prof'].data[0, ...]
RjRj_qt_env_32 = data_32D['RjRj_qt_env_prof'].data[0, ...]
RjRj_qt_env_64 = data_64D['RjRj_qt_env_prof'].data[0, ...]



################################################################


LijMij_env_2 = data_2D['LijMij_env_prof'].data[0, ...]
LijMij_env_4 = data_4D['LijMij_env_prof'].data[0, ...]
LijMij_env_8 = data_8D['LijMij_env_prof'].data[0, ...]
LijMij_env_16 = data_16D['LijMij_env_prof'].data[0, ...]
LijMij_env_32 = data_32D['LijMij_env_prof'].data[0, ...]
LijMij_env_64 = data_64D['LijMij_env_prof'].data[0, ...]

HjRj_th_env_2 = data_2D['HjRj_th_env_prof'].data[0, ...]
HjRj_th_env_4 = data_4D['HjRj_th_env_prof'].data[0, ...]
HjRj_th_env_8 = data_8D['HjRj_th_env_prof'].data[0, ...]
HjRj_th_env_16 = data_16D['HjRj_th_env_prof'].data[0, ...]
HjRj_th_env_32 = data_32D['HjRj_th_env_prof'].data[0, ...]
HjRj_th_env_64 = data_64D['HjRj_th_env_prof'].data[0, ...]

HjRj_qt_env_2 = data_2D['HjRj_qt_env_prof'].data[0, ...]
HjRj_qt_env_4 = data_4D['HjRj_qt_env_prof'].data[0, ...]
HjRj_qt_env_8 = data_8D['HjRj_qt_env_prof'].data[0, ...]
HjRj_qt_env_16 = data_16D['HjRj_qt_env_prof'].data[0, ...]
HjRj_qt_env_32 = data_32D['HjRj_qt_env_prof'].data[0, ...]
HjRj_qt_env_64 = data_64D['HjRj_qt_env_prof'].data[0, ...]


########################################################################################################################



Cs_sq_cloud_2 = LijMij_cloud_2 / MijMij_cloud_2
Cs_sq_cloud_4 = LijMij_cloud_4 / MijMij_cloud_4
Cs_sq_cloud_8 = LijMij_cloud_8 / MijMij_cloud_8
Cs_sq_cloud_16 = LijMij_cloud_16 / MijMij_cloud_16
Cs_sq_cloud_32 = LijMij_cloud_32 / MijMij_cloud_32
Cs_sq_cloud_64 = LijMij_cloud_64 / MijMij_cloud_64

Cth_sq_cloud_2 = HjRj_th_cloud_2 / RjRj_th_cloud_2
Cth_sq_cloud_4 = HjRj_th_cloud_4 / RjRj_th_cloud_4
Cth_sq_cloud_8 = HjRj_th_cloud_8 / RjRj_th_cloud_8
Cth_sq_cloud_16 = HjRj_th_cloud_16 / RjRj_th_cloud_16
Cth_sq_cloud_32 = HjRj_th_cloud_32 / RjRj_th_cloud_32
Cth_sq_cloud_64 = HjRj_th_cloud_64 / RjRj_th_cloud_64

Cqt_sq_cloud_2 = HjRj_qt_cloud_2 / RjRj_qt_cloud_2
Cqt_sq_cloud_4 = HjRj_qt_cloud_4 / RjRj_qt_cloud_4
Cqt_sq_cloud_8 = HjRj_qt_cloud_8 / RjRj_qt_cloud_8
Cqt_sq_cloud_16 = HjRj_qt_cloud_16 / RjRj_qt_cloud_16
Cqt_sq_cloud_32 = HjRj_qt_cloud_32 / RjRj_qt_cloud_32
Cqt_sq_cloud_64 = HjRj_qt_cloud_64 / RjRj_qt_cloud_64
###############################################################


Cs_sq_env_2 = LijMij_env_2 / MijMij_env_2
Cs_sq_env_4 = LijMij_env_4 / MijMij_env_4
Cs_sq_env_8 = LijMij_env_8 / MijMij_env_8
Cs_sq_env_16 = LijMij_env_16 / MijMij_env_16
Cs_sq_env_32 = LijMij_env_32 / MijMij_env_32
Cs_sq_env_64 = LijMij_env_64 / MijMij_env_64

Cth_sq_env_2 = HjRj_th_env_2 / RjRj_th_env_2
Cth_sq_env_4 = HjRj_th_env_4 / RjRj_th_env_4
Cth_sq_env_8 = HjRj_th_env_8 / RjRj_th_env_8
Cth_sq_env_16 = HjRj_th_env_16 / RjRj_th_env_16
Cth_sq_env_32 = HjRj_th_env_32 / RjRj_th_env_32
Cth_sq_env_64 = HjRj_th_env_64 / RjRj_th_env_64

Cqt_sq_env_2 = HjRj_qt_env_2 / RjRj_qt_env_2
Cqt_sq_env_4 = HjRj_qt_env_4 / RjRj_qt_env_4
Cqt_sq_env_8 = HjRj_qt_env_8 / RjRj_qt_env_8
Cqt_sq_env_16 = HjRj_qt_env_16 / RjRj_qt_env_16
Cqt_sq_env_32 = HjRj_qt_env_32 / RjRj_qt_env_32
Cqt_sq_env_64 = HjRj_qt_env_64 / RjRj_qt_env_64


###################################################################################################

beta_s_cloud = dyn.beta_calc(Cs_sq_cloud_2, Cs_sq_cloud_4)
beta_th_cloud = dyn.beta_calc(Cth_sq_cloud_2, Cth_sq_cloud_4)
beta_q_cloud = dyn.beta_calc(Cqt_sq_cloud_2, Cqt_sq_cloud_4)

beta_s_env = dyn.beta_calc(Cs_sq_env_2, Cs_sq_env_4)
beta_th_env = dyn.beta_calc(Cth_sq_env_2, Cth_sq_env_4)
beta_q_env = dyn.beta_calc(Cqt_sq_env_2, Cqt_sq_env_4)



plt.figure(figsize=(6,7))
plt.plot(beta_s_cloud, z/z_i, label='$\\beta_s$')
plt.plot(beta_th_cloud, z/z_i, label='$\\beta_{\\theta}$')
plt.plot(beta_q_cloud, z/z_i, label='$\\beta_{qt}$')
plt.legend(fontsize=12, loc='upper right')
plt.xlabel('$\\beta$ in cloud', fontsize=14)
plt.ylabel("z/z$_{ML}$")
plt.xlim(0, 1)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'all_beta_profs_cloud_scaled.png', pad_inches=0)
plt.close()

plt.figure(figsize=(6,7))
plt.plot(beta_s_env, z/z_i, label='$\\beta_s$')
plt.plot(beta_th_env, z/z_i, label='$\\beta_{\\theta}$')
plt.plot(beta_q_env, z/z_i, label='$\\beta_{qt}$')
plt.legend(fontsize=12, loc='upper right')
plt.xlabel('$\\beta$ in env', fontsize=14)
plt.ylabel("z/z$_{ML}$")
plt.xlim(0, 1)
plt.savefig(plotdir+'all_beta_profs_env_scaled.png', pad_inches=0)
plt.close()


###########################################################################################################


Cs_cloud_beta_sq = Cs_sq_cloud_2/beta_s_cloud
Cth_cloud_beta_sq = Cth_sq_cloud_2/beta_th_cloud
Cqt_cloud_beta_sq = Cqt_sq_cloud_2/beta_q_cloud

Cs_env_beta_sq = Cs_sq_env_2/beta_s_env
Cth_env_beta_sq = Cth_sq_env_2/beta_th_env
Cqt_env_beta_sq = Cqt_sq_env_2/beta_q_env

Cs_cloud_beta = dyn.get_Cs(Cs_cloud_beta_sq)
Cth_cloud_beta = dyn.get_Cs(Cth_cloud_beta_sq)
Cqt_cloud_beta = dyn.get_Cs(Cqt_cloud_beta_sq)

Cs_env_beta = dyn.get_Cs(Cs_env_beta_sq)
Cth_env_beta = dyn.get_Cs(Cth_env_beta_sq)
Cqt_env_beta = dyn.get_Cs(Cqt_env_beta_sq)


# plt.figure(figsize=(6,7))
# plt.plot(Cs_beta, z, label = '$\\Delta = 20$m')
# plt.plot(Cs_2, z, label = '$\\Delta = 40}m$')
# plt.plot(Cs_4, z, label = '$\\Delta = 80}m$')
# plt.xlabel('$C_{s}$', fontsize=16)
# plt.ylabel("z (m)")
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'Cs_prof.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(Cs_cloud_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cs_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cs_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cs_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cs_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cs_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{s}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cs_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cs_env_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cs_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cs_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cs_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cs_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cs_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{s}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cs_env_prof_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cs_cloud_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_sq_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cs_sq_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cs_sq_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cs_sq_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cs_sq_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cs_sq_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{s}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cs_sq_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cs_env_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_sq_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cs_sq_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cs_sq_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cs_sq_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cs_sq_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cs_sq_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{s}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cs_sq_env_prof_scaled.png', pad_inches=0)
plt.close()




#
# plt.figure(figsize=(6,7))
# plt.plot(Cq_beta, z, label = '$\\Delta = 20$m')
# plt.plot(Cq_2, z, label = '$\\Delta = 40$m')
# plt.plot(Cq_4, z, label = '$\\Delta = 80$m')
# plt.xlabel('$C_{qt}$', fontsize=14)
# plt.ylabel("z (m)")
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'Cqt_prof.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(Cth_cloud_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cth_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cth_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cth_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cth_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cth_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{s}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cth_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cth_env_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cth_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cth_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cth_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cth_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cth_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{s}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cth_env_prof_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cth_cloud_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_sq_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cth_sq_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cth_sq_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cth_sq_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cth_sq_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cth_sq_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{\\theta}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cth_sq_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cth_env_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_sq_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cth_sq_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cth_sq_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cth_sq_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cth_sq_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cth_sq_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{\\theta}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cth_sq_env_prof_scaled.png', pad_inches=0)
plt.close()



plt.figure(figsize=(6,7))
plt.plot(Cqt_cloud_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cqt_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cqt_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cqt_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cqt_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cqt_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cqt_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{qt}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cqt_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cqt_env_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cqt_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cqt_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cqt_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cqt_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cqt_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cqt_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C_{qt}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cqt_env_prof_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cqt_cloud_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cqt_sq_cloud_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cqt_sq_cloud_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cqt_sq_cloud_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cqt_sq_cloud_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cqt_sq_cloud_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cqt_sq_cloud_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{qt}$ in Cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Cqt_sq_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Cqt_env_beta_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cqt_sq_env_2, z/z_i, label = '$\\Delta = 40}m$')
plt.plot(Cqt_sq_env_4, z/z_i, label = '$\\Delta = 80}m$')
plt.plot(Cqt_sq_env_8, z/z_i, label = '$\\Delta = 160}m$')
plt.plot(Cqt_sq_env_16, z/z_i, label = '$\\Delta = 320}m$')
plt.plot(Cqt_sq_env_32, z/z_i, label = '$\\Delta = 640}m$')
plt.plot(Cqt_sq_env_64, z/z_i, label = '$\\Delta = 1280}m$')
plt.xlabel('$C^2_{qt}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cqt_sq_env_prof_scaled.png', pad_inches=0)
plt.close()



######################################################################################################################







#########################################################################################################################

monc_l_20 = dyn.l_mix_MONC(0.23, 20, z, k=0.4)
monc_l_40 = dyn.l_mix_MONC(0.23, 40, z, k=0.4)
monc_l_80 = dyn.l_mix_MONC(0.23, 80, z, k=0.4)
monc_l_160 = dyn.l_mix_MONC(0.23, 160, z, k=0.4)
monc_l_320 = dyn.l_mix_MONC(0.23, 320, z, k=0.4)
monc_l_640 = dyn.l_mix_MONC(0.23, 640, z, k=0.4)
monc_l_1280 = dyn.l_mix_MONC(0.23, 1280, z, k=0.4)


# plt.plot(Cs_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cs_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cs_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(monc_l_20, z, color ='tab:blue')
# plt.plot(monc_l_40, z, color ='tab:orange')
# plt.plot(monc_l_80, z, color ='tab:green')
# plt.xlabel('$l_{mix}$', fontsize=16)
# plt.ylabel("z (m)", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_mix_w_MONC.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(Cs_cloud_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cs_cloud_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cs_cloud_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cs_cloud_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cs_cloud_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cs_cloud_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cs_cloud_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')
# plt.plot(monc_l_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_1280, z/z_i, color ='tab:pink')
plt.xlabel('$l_{mix}$ in cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'l_mix_cloud_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cs_env_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cs_env_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cs_env_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cs_env_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cs_env_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cs_env_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cs_env_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')
# plt.plot(monc_l_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_1280, z/z_i, color ='tab:pink')
plt.xlabel('$l_{mix}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'l_mix_env_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()



#########################################################################################################################

#scalars

#########################################################################################################################

C_scalar = np.sqrt((0.23*0.23)/0.7)

monc_l_scalar_20 = dyn.l_mix_MONC(C_scalar, 20, z, k=0.4)
monc_l_scalar_40 = dyn.l_mix_MONC(C_scalar, 40, z, k=0.4)
monc_l_scalar_80 = dyn.l_mix_MONC(C_scalar, 80, z, k=0.4)
monc_l_scalar_160 = dyn.l_mix_MONC(C_scalar, 160, z, k=0.4)
monc_l_scalar_320 = dyn.l_mix_MONC(C_scalar, 320, z, k=0.4)
monc_l_scalar_640 = dyn.l_mix_MONC(C_scalar, 640, z, k=0.4)
monc_l_scalar_1280 = dyn.l_mix_MONC(C_scalar, 1280, z, k=0.4)

# plt.figure(figsize=(6,7))
# plt.plot(Cth_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cth_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cth_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(monc_l_scalar_20, z, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z, color ='tab:green')
# plt.xlabel('$l_{\\theta}$', fontsize=16)
# plt.ylabel("z (m)", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_th_w_MONC.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(Cth_cloud_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cth_cloud_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cth_cloud_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cth_cloud_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cth_cloud_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cth_cloud_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cth_cloud_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')
# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')
plt.xlabel('$l_{\\theta}$ in cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'l_th_cloud_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cth_env_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cth_env_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cth_env_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cth_env_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cth_env_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cth_env_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cth_env_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')

plt.xlabel('$l_{\\theta}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'l_th_env_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()

print('plotted l_th')

#########################################################################################################################



#########################################################################################################################

#
# plt.figure(figsize=(6,7))
# plt.plot(Cq_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cq_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cq_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(monc_l_scalar_20, z, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z, color ='tab:green')
# plt.xlabel('$l_{qt}$', fontsize=16)
# plt.ylabel("z (m)", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_qt_w_MONC.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(Cqt_cloud_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cqt_cloud_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cqt_cloud_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cqt_cloud_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cqt_cloud_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cqt_cloud_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cqt_cloud_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')

plt.xlabel('$l_{qt}$ in cloud', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'l_qt_cloud_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()

print('plotted l_qt')





plt.figure(figsize=(6,7))
plt.plot(Cqt_env_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
plt.plot(Cqt_env_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
plt.plot(Cqt_env_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
plt.plot(Cqt_env_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
plt.plot(Cqt_env_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
plt.plot(Cqt_env_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
plt.plot(Cqt_env_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')

plt.xlabel('$l_{qt}$ in env', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'l_qt_env_no_stan_w_MONC_scaled.png', pad_inches=0)
plt.close()

print('plotted l_qt')

#########################################################################################################################


Pr_th_cloud_beta = dyn.Pr(Cs_cloud_beta_sq, Cth_cloud_beta_sq)
Pr_th_cloud_2D = dyn.Pr(Cs_sq_cloud_2, Cth_sq_cloud_2)
Pr_th_cloud_4D = dyn.Pr(Cs_sq_cloud_4, Cth_sq_cloud_4)
Pr_th_cloud_8D = dyn.Pr(Cs_sq_cloud_8, Cth_sq_cloud_8)
Pr_th_cloud_16D = dyn.Pr(Cs_sq_cloud_16, Cth_sq_cloud_16)
Pr_th_cloud_32D = dyn.Pr(Cs_sq_cloud_32, Cth_sq_cloud_32)
Pr_th_cloud_64D = dyn.Pr(Cs_sq_cloud_64, Cth_sq_cloud_64)


Pr_th_env_beta = dyn.Pr(Cs_env_beta_sq, Cth_env_beta_sq)
Pr_th_env_2D = dyn.Pr(Cs_sq_env_2, Cth_sq_env_2)
Pr_th_env_4D = dyn.Pr(Cs_sq_env_4, Cth_sq_env_4)
Pr_th_env_8D = dyn.Pr(Cs_sq_env_8, Cth_sq_env_8)
Pr_th_env_16D = dyn.Pr(Cs_sq_env_16, Cth_sq_env_16)
Pr_th_env_32D = dyn.Pr(Cs_sq_env_32, Cth_sq_env_32)
Pr_th_env_64D = dyn.Pr(Cs_sq_env_64, Cth_sq_env_64)

# plt.figure(figsize=(6,7))
# plt.plot(Pr_th_beta, z, label = '$\\Delta = 20$m')
# plt.plot(Pr_th_2D, z, label = '$\\Delta = 40$m')
# plt.plot(Pr_th_4D, z, label = '$\\Delta = 80$m')
# plt.plot(Pr_th_8D, z, label = '$\\Delta = 160$m')
# plt.plot(Pr_th_16D, z, label = '$\\Delta = 320$m')
# plt.plot(Pr_th_32D, z, label = '$\\Delta = 640$m')
# plt.plot(Pr_th_64D, z, label = '$\\Delta = 1280$m')
# plt.xlabel('$Pr_{\\theta}$', fontsize=14)
# plt.ylabel("z (m)")
# plt.legend(fontsize=12, loc='upper right')
# plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
# #plt.xlim(-3, 7)
# plt.savefig(plotdir+'Pr_th_prof.png', pad_inches=0)
# plt.close()

plt.figure(figsize=(6,7))
plt.plot(Pr_th_cloud_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Pr_th_cloud_2D, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Pr_th_cloud_4D, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Pr_th_cloud_8D, z/z_i, label = '$\\Delta = 160$m')
plt.plot(Pr_th_cloud_16D, z/z_i, label = '$\\Delta = 320$m')
plt.plot(Pr_th_cloud_32D, z/z_i, label = '$\\Delta = 640$m')
plt.plot(Pr_th_cloud_64D, z/z_i, label = '$\\Delta = 1280$m')
plt.xlabel('$Pr_{\\theta}$ in cloud', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
plt.xlim(-3, 7)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Pr_th_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Pr_th_env_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Pr_th_env_2D, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Pr_th_env_4D, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Pr_th_env_8D, z/z_i, label = '$\\Delta = 160$m')
plt.plot(Pr_th_env_16D, z/z_i, label = '$\\Delta = 320$m')
plt.plot(Pr_th_env_32D, z/z_i, label = '$\\Delta = 640$m')
plt.plot(Pr_th_env_64D, z/z_i, label = '$\\Delta = 1280$m')
plt.xlabel('$Pr_{\\theta}$ in env', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
plt.xlim(-3, 7)
plt.savefig(plotdir+'Pr_th_env_prof_scaled.png', pad_inches=0)
plt.close()



###############################################################################################

Pr_qt_cloud_beta = dyn.Pr(Cs_cloud_beta_sq, Cqt_cloud_beta_sq)
Pr_qt_cloud_2D = dyn.Pr(Cs_sq_cloud_2, Cqt_sq_cloud_2)
Pr_qt_cloud_4D = dyn.Pr(Cs_sq_cloud_4, Cqt_sq_cloud_4)
Pr_qt_cloud_8D = dyn.Pr(Cs_sq_cloud_8, Cqt_sq_cloud_8)
Pr_qt_cloud_16D = dyn.Pr(Cs_sq_cloud_16, Cqt_sq_cloud_16)
Pr_qt_cloud_32D = dyn.Pr(Cs_sq_cloud_32, Cqt_sq_cloud_32)
Pr_qt_cloud_64D = dyn.Pr(Cs_sq_cloud_64, Cqt_sq_cloud_64)

Pr_qt_env_beta = dyn.Pr(Cs_env_beta_sq, Cqt_env_beta_sq)
Pr_qt_env_2D = dyn.Pr(Cs_sq_env_2, Cqt_sq_env_2)
Pr_qt_env_4D = dyn.Pr(Cs_sq_env_4, Cqt_sq_env_4)
Pr_qt_env_8D = dyn.Pr(Cs_sq_env_8, Cqt_sq_env_8)
Pr_qt_env_16D = dyn.Pr(Cs_sq_env_16, Cqt_sq_env_16)
Pr_qt_env_32D = dyn.Pr(Cs_sq_env_32, Cqt_sq_env_32)
Pr_qt_env_64D = dyn.Pr(Cs_sq_env_64, Cqt_sq_env_64)



plt.figure(figsize=(6,7))
plt.plot(Pr_qt_cloud_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Pr_qt_cloud_2D, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Pr_qt_cloud_4D, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Pr_qt_cloud_8D, z/z_i, label = '$\\Delta = 160$m')
plt.plot(Pr_qt_cloud_16D, z/z_i, label = '$\\Delta = 320$m')
plt.plot(Pr_qt_cloud_32D, z/z_i, label = '$\\Delta = 640$m')
plt.plot(Pr_qt_cloud_64D, z/z_i, label = '$\\Delta = 1280$m')
plt.xlabel('$Pr_{qt}$ in cloud', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
plt.xlim(-3, 7)
plt.ylim(-0.25, 6.25)
plt.savefig(plotdir+'Pr_qt_cloud_prof_scaled.png', pad_inches=0)
plt.close()


plt.figure(figsize=(6,7))
plt.plot(Pr_qt_env_beta, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Pr_qt_env_2D, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Pr_qt_env_4D, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Pr_qt_env_8D, z/z_i, label = '$\\Delta = 160$m')
plt.plot(Pr_qt_env_16D, z/z_i, label = '$\\Delta = 320$m')
plt.plot(Pr_qt_env_32D, z/z_i, label = '$\\Delta = 640$m')
plt.plot(Pr_qt_env_64D, z/z_i, label = '$\\Delta = 1280$m')
plt.xlabel('$Pr_{qt}$ in env', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
plt.xlim(-3, 7)
plt.savefig(plotdir+'Pr_qt_env_prof_scaled.png', pad_inches=0)
plt.close()


