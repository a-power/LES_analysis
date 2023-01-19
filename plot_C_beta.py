import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn
import cmath as cm

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

dir_data_Cs = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_Cs_'
dir_data_C_th = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_C_th_'
dir_data_Cq_tot = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_C_qt_'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/plots/C_beta_profiles/'
os.makedirs(plotdir, exist_ok = True)

data_2D_s = xr.open_dataset(dir_data_Cs + '2D.nc')
data_4D_s = xr.open_dataset(dir_data_Cs + '4D.nc')
data_8D_s = xr.open_dataset(dir_data_Cs + '8D.nc')
data_16D_s = xr.open_dataset(dir_data_Cs + '16D.nc')
data_32D_s = xr.open_dataset(dir_data_Cs + '32D.nc')
data_64D_s = xr.open_dataset(dir_data_Cs + '64D.nc')

data_2D_th = xr.open_dataset(dir_data_C_th + '2D.nc')
data_4D_th = xr.open_dataset(dir_data_C_th + '4D.nc')
data_8D_th = xr.open_dataset(dir_data_C_th + '8D.nc')
data_16D_th = xr.open_dataset(dir_data_C_th + '16D.nc')
data_32D_th = xr.open_dataset(dir_data_C_th + '32D.nc')
data_64D_th = xr.open_dataset(dir_data_C_th + '64D.nc')

data_2D_qtot = xr.open_dataset(dir_data_Cq_tot + '2D.nc')
data_4D_qtot = xr.open_dataset(dir_data_Cq_tot + '4D.nc')
data_8D_qtot = xr.open_dataset(dir_data_Cq_tot + '8D.nc')
data_16D_qtot = xr.open_dataset(dir_data_Cq_tot + '16D.nc')
data_32D_qtot = xr.open_dataset(dir_data_Cq_tot + '32D.nc')
data_64D_qtot = xr.open_dataset(dir_data_Cq_tot + '64D.nc')


z = np.arange(0, 3020, 20)
z_i = 490

#index of 0 at the start is to get rid of the dummy time index thats required to save the files


Cs_sq_2 = data_2D_s['Cs_sq_prof'].data[0, ...]
Cs_sq_4 = data_4D_s['Cs_sq_prof'].data[0, ...]
Cs_sq_8 = data_8D_s['Cs_sq_prof'].data[0, ...]
Cs_sq_16 = data_16D_s['Cs_sq_prof'].data[0, ...]
Cs_sq_32 = data_32D_s['Cs_sq_prof'].data[0, ...]
Cs_sq_64 = data_64D_s['Cs_sq_prof'].data[0, ...]

Cth_sq_2 = data_2D_th['C_th_sq_prof'].data[0, ...]
Cth_sq_4 = data_4D_th['C_th_sq_prof'].data[0, ...]
Cth_sq_8 = data_8D_th['C_th_sq_prof'].data[0, ...]
Cth_sq_16 = data_16D_th['C_th_sq_prof'].data[0, ...]
Cth_sq_32 = data_32D_th['C_th_sq_prof'].data[0, ...]
Cth_sq_64 = data_64D_th['C_th_sq_prof'].data[0, ...]

Cq_sq_2 = data_2D_qtot['C_q_total_sq_prof'].data[0, ...]
Cq_sq_4 = data_4D_qtot['C_q_total_sq_prof'].data[0, ...]
Cq_sq_8 = data_8D_qtot['C_q_total_sq_prof'].data[0, ...]
Cq_sq_16 = data_16D_qtot['C_q_total_sq_prof'].data[0, ...]
Cq_sq_32 = data_32D_qtot['C_q_total_sq_prof'].data[0, ...]
Cq_sq_64 = data_64D_qtot['C_q_total_sq_prof'].data[0, ...]


########################################################################################################################


beta_s1 = dyn.beta_calc(Cs_sq_2, Cs_sq_4)
beta_th1 = dyn.beta_calc(Cth_sq_2, Cth_sq_4)
beta_q1 = dyn.beta_calc(Cq_sq_2, Cq_sq_4)

beta_s2 = dyn.beta_calc(Cs_sq_4, Cs_sq_8)
beta_th2 = dyn.beta_calc(Cth_sq_4, Cth_sq_8)
beta_q2 = dyn.beta_calc(Cq_sq_4, Cq_sq_8)

beta_s3 = dyn.beta_calc(Cs_sq_8, Cs_sq_16)
beta_th3 = dyn.beta_calc(Cth_sq_8, Cth_sq_16)
beta_q3 = dyn.beta_calc(Cq_sq_8, Cq_sq_16)

beta_s4 = dyn.beta_calc(Cs_sq_16, Cs_sq_32)
beta_th4 = dyn.beta_calc(Cth_sq_16, Cth_sq_32)
beta_q4 = dyn.beta_calc(Cq_sq_16, Cq_sq_32)

beta_s5 = dyn.beta_calc(Cs_sq_32, Cs_sq_64)
beta_th5 = dyn.beta_calc(Cth_sq_32, Cth_sq_64)
beta_q5 = dyn.beta_calc(Cq_sq_32, Cq_sq_64)


plt.figure(figsize=(6,7))
plt.plot(beta_s1, z/z_i, label='$2\\Delta \rightarrow 4\Delta$')
plt.plot(beta_s2, z/z_i, label='$4\\Delta \rightarrow 8\Delta$')
plt.plot(beta_s3, z/z_i, label='$8\\Delta \rightarrow 16\Delta$')
plt.plot(beta_s4, z/z_i, label='$16\\Delta \rightarrow 32\Delta$')
plt.plot(beta_s5, z/z_i, label='$32\\Delta \rightarrow 64\Delta$')
plt.xlabel('$\\beta_s$', fontsize=14)
plt.ylabel("z/z$_{ML}$")
#plt.xlim(0, 1)
plt.savefig(plotdir+'betas_mom_prof.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(beta_th1, z/z_i, label='$2\\Delta \rightarrow 4\Delta$')
plt.plot(beta_th2, z/z_i, label='$4\\Delta \rightarrow 8\Delta$')
plt.plot(beta_th3, z/z_i, label='$8\\Delta \rightarrow 16\Delta$')
plt.plot(beta_th4, z/z_i, label='$16\\Delta \rightarrow 32\Delta$')
plt.plot(beta_th5, z/z_i, label='$32\\Delta \rightarrow 64\Delta$')
plt.xlabel('$\\beta_{\\theta}$', fontsize=14)
plt.ylabel("z/z$_{ML}$")
#plt.xlim(0, 1)
plt.savefig(plotdir+'betas_th_prof.png', pad_inches=0)

plt.figure(figsize=(6,7))
plt.plot(beta_q1, z/z_i, label='$2\\Delta \rightarrow 4\Delta$')
plt.plot(beta_q2, z/z_i, label='$4\\Delta \rightarrow 8\Delta$')
plt.plot(beta_q3, z/z_i, label='$8\\Delta \rightarrow 16\Delta$')
plt.plot(beta_q4, z/z_i, label='$16\\Delta \rightarrow 32\Delta$')
plt.plot(beta_q5, z/z_i, label='$32\\Delta \rightarrow 64\Delta$')
plt.xlabel('$\\beta_{qt}$', fontsize=14)
plt.ylabel("z/z$_{ML}$")
#plt.xlim(0, 1)
plt.savefig(plotdir+'beta_qt_prof.png', pad_inches=0)




Delta = np.array([20, 40, 80, 160, 320])
z_cl_r = [50, 75]
z_ml_r = [6, 20]

beta_s_list = [beta_s1, beta_s2, beta_s3, beta_s4, beta_s5]
beta_th_list = [beta_th1, beta_th2, beta_th3, beta_th4, beta_th5]
beta_q_list = [beta_q1, beta_q2, beta_q3, beta_q4, beta_q5]

def cal_max_beta(beta_list, z_range):

    max_beta = np.zeros(len(beta_list))
    for i in range(len(beta_list)):
        max_beta[i] = np.max(beta_list[i][z_range[0]:z_range[1]])
    return max_beta

plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_beta(beta_s_list, z_ml_r), 'k', label = '$\\beta_{s}$')
plt.plot(Delta, cal_max_beta(beta_s_list, z_cl_r), 'k-.')
plt.plot(Delta, cal_max_beta(beta_th_list, z_ml_r), 'r', label = '$\\beta_{\theta}$')
plt.plot(Delta, cal_max_beta(beta_th_list, z_cl_r), 'r-.')
plt.plot(Delta, cal_max_beta(beta_q_list, z_ml_r), 'b', label = '$\\beta_{qt}$')
plt.plot(Delta, cal_max_beta(beta_q_list, z_cl_r), 'b-.')
plt.xlabel('Filter scale $\\Delta$ (m)', fontsize=16)
plt.ylabel('$\\beta$', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'beta_max_prof_ML_vs_CL.png', pad_inches=0)
plt.close()


# plt.figure(figsize=(6,7))
# plt.plot(beta_s, z/z_i)
# plt.xlabel('$\\beta_s$', fontsize=14)
# plt.ylabel("z/z$_{ML}$")
# plt.xlim(0, 1)
# plt.savefig(plotdir+'Cs_beta_prof_scaled.png', pad_inches=0)


###########################################################################################################


Cs_beta1_sq = Cs_sq_2/beta_s1
Cth_beta1_sq = Cth_sq_2/beta_th1
Cq_beta1_sq = Cq_sq_2/beta_q1

Cs_beta1 = dyn.get_Cs(Cs_beta1_sq)
Cth_beta1 = dyn.get_Cs(Cth_beta1_sq)
Cq_beta1 = dyn.get_Cs(Cq_beta1_sq)



Cs_beta2_sq = Cs_sq_4/beta_s2
Cth_beta2_sq = Cth_sq_4/beta_th2
Cq_beta2_sq = Cq_sq_4/beta_q2

Cs_beta2 = dyn.get_Cs(Cs_beta2_sq)
Cth_beta2 = dyn.get_Cs(Cth_beta2_sq)
Cq_beta2 = dyn.get_Cs(Cq_beta2_sq)



Cs_beta3_sq = Cs_sq_8/beta_s3
Cth_beta3_sq = Cth_sq_8/beta_th3
Cq_beta3_sq = Cq_sq_8/beta_q3

Cs_beta3 = dyn.get_Cs(Cs_beta3_sq)
Cth_beta3 = dyn.get_Cs(Cth_beta3_sq)
Cq_beta3 = dyn.get_Cs(Cq_beta3_sq)



Cs_beta4_sq = Cs_sq_16/beta_s4
Cth_beta4_sq = Cth_sq_16/beta_th4
Cq_beta4_sq = Cq_sq_16/beta_q4

Cs_beta4 = dyn.get_Cs(Cs_beta4_sq)
Cth_beta4 = dyn.get_Cs(Cth_beta4_sq)
Cq_beta4 = dyn.get_Cs(Cq_beta4_sq)



Cs_beta5_sq = Cs_sq_32/beta_s5
Cth_beta5_sq = Cth_sq_32/beta_th5
Cq_beta5_sq = Cq_sq_32/beta_q5

Cs_beta5 = dyn.get_Cs(Cs_beta5_sq)
Cth_beta5 = dyn.get_Cs(Cth_beta5_sq)
Cq_beta5 = dyn.get_Cs(Cq_beta5_sq)


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
plt.plot(Cs_beta1, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_beta2, z/z_i, label = '$\\Delta = 40}$m')
plt.plot(Cs_beta3, z/z_i, label = '$\\Delta = 80}$m')
plt.plot(Cs_beta4, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cs_beta5, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C_{s \\beta}$', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cs_beta_prof_scaled.png', pad_inches=0)
plt.close()

plt.figure(figsize=(6,7))
plt.plot(Cs_beta1_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cs_beta2_sq, z/z_i, label = '$\\Delta = 40}$m')
plt.plot(Cs_beta3_sq, z/z_i, label = '$\\Delta = 80}$m')
plt.plot(Cs_beta4_sq, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cs_beta5_sq, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C^2_{s \\beta}$', fontsize=16)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cs_beta_sq_prof_scaled.png', pad_inches=0)
plt.close()




plt.figure(figsize=(6,7))
plt.plot(Cq_beta1, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cq_beta2, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Cq_beta3, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Cq_beta4, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cq_beta5, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C_{qt \\beta}$', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cqt_beta_prof_scaled.png', pad_inches=0)
plt.close()
#
#
plt.figure(figsize=(6,7))
plt.plot(Cq_beta1_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cq_beta2_sq, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Cq_beta3_sq, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Cq_beta4_sq, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cq_beta5_sq, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C^2_{qt \\beta}$', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cqt_beta_sq_prof_scaled.png', pad_inches=0)
plt.close()
#
#
# #
# # plt.figure(figsize=(6,7))
# # plt.plot(Cth_beta, z, label = '$\\Delta = 20$m')
# # plt.plot(Cth_2, z, label = '$\\Delta = 40$m')
# # plt.plot(Cth_4, z, label = '$\\Delta = 80$m')
# # plt.xlabel('$C_{\\theta}$', fontsize=14)
# # plt.ylabel("z (m)")
# # plt.legend(fontsize=12, loc='upper right')
# # #plt.xlim(1, 3)
# # plt.savefig(plotdir+'Cth_prof.png', pad_inches=0)
#
plt.figure(figsize=(6,7))
plt.plot(Cth_beta1, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_beta2, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Cth_beta3, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Cth_beta4, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cth_beta5, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C_{\\theta \beta}$', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cth_beta_prof_scaled.png', pad_inches=0)
plt.close()
#
#
plt.figure(figsize=(6,7))
plt.plot(Cth_beta1_sq, z/z_i, label = '$\\Delta = 20$m')
plt.plot(Cth_beta2_sq, z/z_i, label = '$\\Delta = 40$m')
plt.plot(Cth_beta3_sq, z/z_i, label = '$\\Delta = 80$m')
plt.plot(Cth_beta4_sq, z/z_i, label = '$\\Delta = 160}$m')
plt.plot(Cth_beta5_sq, z/z_i, label = '$\\Delta = 320}$m')
plt.xlabel('$C^2_{\\theta \beta}$', fontsize=14)
plt.ylabel("z/z$_{ML}$", fontsize=16)
plt.legend(fontsize=12, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cth_beta_sq_prof_scaled.png', pad_inches=0)
plt.close()


#########################################################################################################################

monc_l_20 = dyn.l_mix_MONC(0.23, 20, z, k=0.4)
monc_l_40 = dyn.l_mix_MONC(0.23, 40, z, k=0.4)
monc_l_80 = dyn.l_mix_MONC(0.23, 80, z, k=0.4)
monc_l_160 = dyn.l_mix_MONC(0.23, 160, z, k=0.4)
monc_l_320 = dyn.l_mix_MONC(0.23, 320, z, k=0.4)
monc_l_640 = dyn.l_mix_MONC(0.23, 640, z, k=0.4)
monc_l_1280 = dyn.l_mix_MONC(0.23, 1280, z, k=0.4)

# plt.figure(figsize=(6,7))
# plt.plot(Cs_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cs_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cs_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(monc_l_20, z, color ='tab:blue')
# plt.plot(monc_l_40, z, color ='tab:orange')
# plt.plot(monc_l_80, z, color ='tab:green')
# plt.xlabel('$l_{mix}$', fontsize=16)
# plt.ylabel("z (m)", fontsize=16)
# plt.legend(fontsize=14, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_mix_w_MONC.png', pad_inches=0, bbox_inches = 'tight')

# plt.figure(figsize=(6,7))
#
# plt.plot(Cs_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cs_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cs_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(Cs_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
# plt.plot(Cs_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
# plt.plot(Cs_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
# plt.plot(Cs_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_20, z/z_i, linewidth=1.5, color ='tab:blue')
# plt.plot(monc_l_40, z/z_i, linewidth=1.5, color ='tab:orange')
# plt.plot(monc_l_80, z/z_i, linewidth=1.5, color ='tab:green')
# plt.plot(monc_l_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_1280, z/z_i, color ='tab:pink')

# plt.xlabel('$l_{mix}$', fontsize=16)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=16, loc='upper right')
# #plt.xlim(-1, 22)
# plt.savefig(plotdir+'l_mix_3D_w_MONC_scaled.png', pad_inches=0)
# plt.close()
#
# print('plotted l mix')

#########################################################################################################################

#scalars

#########################################################################################################################
#
# C_scalar = np.sqrt((0.23*0.23)/0.7)
#
# monc_l_scalar_20 = dyn.l_mix_MONC(C_scalar, 20, z, k=0.4)
# monc_l_scalar_40 = dyn.l_mix_MONC(C_scalar, 40, z, k=0.4)
# monc_l_scalar_80 = dyn.l_mix_MONC(C_scalar, 80, z, k=0.4)
# monc_l_scalar_160 = dyn.l_mix_MONC(C_scalar, 160, z, k=0.4)
# monc_l_scalar_320 = dyn.l_mix_MONC(C_scalar, 320, z, k=0.4)
# monc_l_scalar_640 = dyn.l_mix_MONC(C_scalar, 640, z, k=0.4)
# monc_l_scalar_1280 = dyn.l_mix_MONC(C_scalar, 1280, z, k=0.4)
#
# # plt.figure(figsize=(6,7))
# # plt.plot(Cth_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# # plt.plot(Cth_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# # plt.plot(Cth_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# # plt.plot(monc_l_scalar_20, z, color ='tab:blue')
# # plt.plot(monc_l_scalar_40, z, color ='tab:orange')
# # plt.plot(monc_l_scalar_80, z, color ='tab:green')
# # plt.xlabel('$l_{\\theta}$', fontsize=16)
# # plt.ylabel("z (m)", fontsize=16)
# # plt.legend(fontsize=12, loc='upper right')
# # #plt.xlim(1, 3)
# # plt.savefig(plotdir+'l_th_w_MONC.png', pad_inches=0)
#
# plt.figure(figsize=(6,7))
# plt.plot(Cth_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cth_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cth_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(Cth_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
# plt.plot(Cth_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
# plt.plot(Cth_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
# plt.plot(Cth_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')

# plt.xlabel('$l_{\\theta}$', fontsize=16)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_th_no_stan_w_MONC_scaled.png', pad_inches=0)
# plt.close()
#
# print('plotted l_th')
#
# #########################################################################################################################
#
#
#
# #########################################################################################################################
#
# #
# # plt.figure(figsize=(6,7))
# # plt.plot(Cq_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
# # plt.plot(Cq_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
# # plt.plot(Cq_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
# # plt.plot(monc_l_scalar_20, z, color ='tab:blue')
# # plt.plot(monc_l_scalar_40, z, color ='tab:orange')
# # plt.plot(monc_l_scalar_80, z, color ='tab:green')
# # plt.xlabel('$l_{qt}$', fontsize=16)
# # plt.ylabel("z (m)", fontsize=16)
# # plt.legend(fontsize=12, loc='upper right')
# # #plt.xlim(1, 3)
# # plt.savefig(plotdir+'l_qt_w_MONC.png', pad_inches=0)
#
# plt.figure(figsize=(6,7))

# plt.plot(Cq_beta*(20), z/z_i, color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
# plt.plot(Cq_2*(40), z/z_i, color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
# plt.plot(Cq_4*(80), z/z_i, color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
# plt.plot(Cq_8*(160), z/z_i, color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')
# plt.plot(Cq_16*(320), z/z_i, color ='tab:purple', markersize = 10, label = '$\\Delta = 320$m')
# plt.plot(Cq_32*(640), z/z_i, color ='tab:grey', markersize = 10, label = '$\\Delta = 640$m')
# plt.plot(Cq_64*(1280), z/z_i, color ='tab:pink', markersize = 10, label = '$\\Delta = 1280$m')

# plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
# plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
# plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
# plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')
# plt.plot(monc_l_scalar_320, z/z_i, color ='tab:purple')
# plt.plot(monc_l_scalar_640, z/z_i, color ='tab:grey')
# plt.plot(monc_l_scalar_1280, z/z_i, color ='tab:pink')

# plt.xlabel('$l_{qt}$', fontsize=16)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# #plt.xlim(1, 3)
# plt.savefig(plotdir+'l_qt_no_stan_w_MONC_scaled.png', pad_inches=0)
# plt.close()
#
# print('plotted l_qt')
# #
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
#
# # plt.figure(figsize=(6,7))
# # plt.plot(Pr_th_beta, z, label = '$\\Delta = 20$m')
# # plt.plot(Pr_th_2D, z, label = '$\\Delta = 40$m')
# # plt.plot(Pr_th_4D, z, label = '$\\Delta = 80$m')
# # plt.plot(Pr_th_8D, z, label = '$\\Delta = 160$m')
# # plt.plot(Pr_th_16D, z, label = '$\\Delta = 320$m')
# # plt.plot(Pr_th_32D, z, label = '$\\Delta = 640$m')
# # plt.plot(Pr_th_64D, z, label = '$\\Delta = 1280$m')
# # plt.xlabel('$Pr_{\\theta}$', fontsize=14)
# # plt.ylabel("z (m)")
# # plt.legend(fontsize=12, loc='upper right')
# # plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
# # #plt.xlim(-3, 7)
# # plt.savefig(plotdir+'Pr_th_prof.png', pad_inches=0)
# # plt.close()
#
# plt.figure(figsize=(6,7))
# plt.plot(Pr_th_beta, z/z_i, label = '$\\Delta = 20$m')
# plt.plot(Pr_th_2D, z/z_i, label = '$\\Delta = 40$m')
# plt.plot(Pr_th_4D, z/z_i, label = '$\\Delta = 80$m')
# plt.plot(Pr_th_8D, z/z_i, label = '$\\Delta = 160$m')
# plt.plot(Pr_th_16D, z/z_i, label = '$\\Delta = 320$m')
# plt.plot(Pr_th_32D, z/z_i, label = '$\\Delta = 640$m')
# plt.plot(Pr_th_64D, z/z_i, label = '$\\Delta = 1280$m')
# plt.xlabel('$Pr_{\\theta}$', fontsize=14)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
# plt.xlim(-3, 7)
# plt.savefig(plotdir+'Pr_th_prof_scaled1.png', pad_inches=0)
# plt.close()
#
#
# plt.figure(figsize=(6,7))
# plt.plot(Pr_th_beta, z/z_i, label = '$\\Delta = 20$m')
# plt.plot(Pr_th_2D, z/z_i, label = '$\\Delta = 40$m')
# plt.plot(Pr_th_4D, z/z_i, label = '$\\Delta = 80$m')
# plt.plot(Pr_th_8D, z/z_i, label = '$\\Delta = 160$m')
# plt.plot(Pr_th_16D, z/z_i, label = '$\\Delta = 320$m')
# plt.plot(Pr_th_32D, z/z_i, label = '$\\Delta = 640$m')
# plt.plot(Pr_th_64D, z/z_i, label = '$\\Delta = 1280$m')
# plt.xlabel('$Pr_{\\theta}$', fontsize=14)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
# plt.xlim(-3, 31)
# plt.savefig(plotdir+'Pr_th_prof_scaled2.png', pad_inches=0)
# plt.close()
#
# ###############################################################################################
#
# Pr_q_beta = dyn.Pr(Cs_beta_sq, Cq_beta_sq)
# Pr_q_2D = dyn.Pr(Cs_sq_2, Cq_sq_2)
# Pr_q_4D = dyn.Pr(Cs_sq_4, Cq_sq_4)
# Pr_q_8D = dyn.Pr(Cs_sq_8, Cq_sq_8)
# Pr_q_16D = dyn.Pr(Cs_sq_16, Cq_sq_16)
# Pr_q_32D = dyn.Pr(Cs_sq_32, Cq_sq_32)
# Pr_q_64D = dyn.Pr(Cs_sq_64, Cq_sq_64)
#
# # plt.figure(figsize=(6,7))
# # plt.plot(Pr_q_beta, z, label = '$\\Delta = 20$m')
# # plt.plot(Pr_q_2D, z, label = '$\\Delta = 40$m')
# # plt.plot(Pr_q_4D, z, label = '$\\Delta = 80$m')
# # plt.plot(Pr_q_8D, z, label = '$\\Delta = 160$m')
# # plt.plot(Pr_q_16D, z, label = '$\\Delta = 320$m')
# # plt.plot(Pr_q_32D, z, label = '$\\Delta = 640$m')
# # plt.plot(Pr_q_64D, z, label = '$\\Delta = 1280$m')
# # plt.xlabel('$Pr_{qt}$', fontsize=14)
# # plt.ylabel("z (m)")
# # plt.legend(fontsize=12, loc='upper right')
# # plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
# # #plt.xlim(-3, 7)
# # plt.savefig(plotdir+'Pr_qt_prof.png', pad_inches=0)
# # plt.close()
#
# plt.figure(figsize=(6,7))
# plt.plot(Pr_q_beta, z/z_i, label = '$\\Delta = 20$m')
# plt.plot(Pr_q_2D, z/z_i, label = '$\\Delta = 40$m')
# plt.plot(Pr_q_4D, z/z_i, label = '$\\Delta = 80$m')
# plt.plot(Pr_q_8D, z/z_i, label = '$\\Delta = 160$m')
# plt.plot(Pr_q_16D, z/z_i, label = '$\\Delta = 320$m')
# plt.plot(Pr_q_32D, z/z_i, label = '$\\Delta = 640$m')
# plt.plot(Pr_th_64D, z/z_i, label = '$\\Delta = 1280$m')
# plt.xlabel('$Pr_{qt}$', fontsize=14)
# plt.ylabel("z/z$_{ML}$", fontsize=16)
# plt.legend(fontsize=12, loc='upper right')
# plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
# plt.xlim(-3, 7)
# plt.savefig(plotdir+'Pr_qt_prof_scaled.png', pad_inches=0)
# plt.close()
#
#
#
# # plt.figure(figsize=(6,7))
# # plt.plot(Pr_th_beta, z, label = '$Pr_{\\theta}:$ $\\Delta = 20$m')
# # plt.plot(Pr_th_2D, z, label = '$Pr_{\\theta}:$ $\\Delta = 40$m')
# # plt.plot(Pr_th_4D, z, label = '$Pr_{\\theta}:$ $\\Delta = 80$m')
# # plt.plot(Pr_q_beta, z, label = '$Pr_{qt}:$ $\\Delta = 20$m', color ='tab:blue', linestyle='dashdot')
# # plt.plot(Pr_q_2D, z, label = '$Pr_{qt}:$ $\\Delta = 40$m', color ='tab:orange', linestyle='dashdot')
# # plt.plot(Pr_q_4D, z, label = '$Pr_{qt}:$ $\\Delta = 80$m', color ='tab:green', linestyle='dashdot')
# # plt.xlabel('$Pr$', fontsize=14)
# # plt.ylabel("z (m)")
# # plt.legend(fontsize=12, loc='upper right')
# # plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
# # plt.xlim(-3, 7)
# # plt.savefig(plotdir+'Pr_all_prof.png', pad_inches=0)
# #
# # plt.figure(figsize=(6,7))
# # plt.plot(Pr_th_beta, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 20$m')
# # plt.plot(Pr_th_2D, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 40$m')
# # plt.plot(Pr_th_4D, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 80$m')
# # plt.plot(Pr_q_beta, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 20$m', color ='tab:blue', linestyle='dashdot')
# # plt.plot(Pr_q_2D, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 40$m', color ='tab:orange', linestyle='dashdot')
# # plt.plot(Pr_q_4D, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 80$m', color ='tab:green', linestyle='dashdot')
# # plt.xlabel('$Pr$', fontsize=14)
# # plt.ylabel("z/$_{ML}$")
# # plt.legend(fontsize=12, loc='upper right')
# # plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
# # plt.xlim(-3, 7)
# # plt.savefig(plotdir+'Pr_all_prof_scaled.png', pad_inches=0)


