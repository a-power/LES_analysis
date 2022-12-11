import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')

dir_data_Cs = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_Cs_'
dir_data_C_th = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_C_th_'
dir_data_Cq_tot = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_C_qt_'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/plots/coarse_data/'
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

Cs_2 = data_2D_s['Cs_prof'].data[0, ...]
Cs_4 = data_4D_s['Cs_prof'].data[0, ...]
Cs_8 = data_8D_s['Cs_prof'].data[0, ...]
Cs_16 = data_16D_s['Cs_prof'].data[0, ...]
Cs_32 = data_32D_s['Cs_prof'].data[0, ...]
Cs_64 = data_64D_s['Cs_prof'].data[0, ...]

Cth_2 = data_2D_th['C_th_prof'].data[0, ...]
Cth_4 = data_4D_th['C_th_prof'].data[0, ...]
Cth_8 = data_8D_th['C_th_prof'].data[0, ...]
Cth_16 = data_16D_th['C_th_prof'].data[0, ...]
Cth_32 = data_32D_th['C_th_prof'].data[0, ...]
Cth_64 = data_64D_th['C_th_prof'].data[0, ...]

Cq_2 = data_2D_qtot['C_q_total_prof'].data[0, ...]
Cq_4 = data_4D_qtot['C_q_total_prof'].data[0, ...]
Cq_8 = data_8D_qtot['C_q_total_prof'].data[0, ...]
Cq_16 = data_16D_qtot['C_q_total_prof'].data[0, ...]
Cq_32 = data_32D_qtot['C_q_total_prof'].data[0, ...]
Cq_64 = data_64D_qtot['C_q_total_prof'].data[0, ...]

# Cql_2 = data_2D_th['C_ql_prof'].data[0, ...]
# Cql_4 = data_4D_th['C_ql_prof'].data[0, ...]
#
# Cqv_2 = data_2D_th['C_qv_prof'].data[0, ...]
# Cqv_4 = data_4D_th['C_qv_prof'].data[0, ...]



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


beta_s = dyn.beta_calc(Cs_sq_2, Cs_sq_4)
beta_th = dyn.beta_calc(Cth_sq_2, Cth_sq_4)
beta_q = dyn.beta_calc(Cq_sq_2, Cq_sq_4)

Cs_beta_sq = Cs_sq_2/beta_s
Cth_beta_sq = Cth_sq_2/beta_th
Cq_beta_sq = Cq_sq_2/beta_q

Cs_beta = dyn.get_Cs(Cs_beta_sq)
Cth_beta = dyn.get_Cs(Cth_beta_sq)
Cq_beta = dyn.get_Cs(Cq_beta_sq)

###########################################################################################################

Delta = np.array([20, 40, 80, 160, 320, 640, 1280])
Cs_list = [Cs_beta, Cs_2, Cs_4, Cs_8, Cs_16, Cs_32, Cs_64]
Cth_list = [Cth_beta, Cth_2, Cth_4, Cth_8, Cth_16, Cth_32, Cth_64]
Cq_list = [Cq_beta, Cq_2, Cq_4, Cq_8, Cq_16, Cq_32, Cq_64]

z_cl_r = [50, 75]
z_ml_r = [6, 20]

def cal_max_Cs(C_list, z_range):

    max_C = np.zeros(len(C_list))
    for i in range(len(C_list)):
        max_C[i] = np.max(C_list[i][z_range[0]:z_range[1]])
    return max_C



plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_list, z_ml_r), 'k', label = '$C_{s}$')
plt.plot(Delta, cal_max_Cs(Cs_list, z_cl_r), 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_list, z_ml_r), 'r', label = '$C_{\\theta}$')
plt.plot(Delta, cal_max_Cs(Cth_list, z_cl_r), 'r-.')
plt.plot(Delta, cal_max_Cs(Cq_list, z_ml_r), 'b', label = '$C_{qt}$')
plt.plot(Delta, cal_max_Cs(Cq_list, z_cl_r), 'b-.')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Smagorinsky Parameter', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'C_max_prof_ML_vs_CL.png', pad_inches=0)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_list, z_ml_r)*Delta, 'k', label = '$l_{mix}$')
plt.plot(Delta, cal_max_Cs(Cs_list, z_cl_r)*Delta, 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_list, z_ml_r)*Delta, 'r', label = '$l_{\\theta}$')
plt.plot(Delta, cal_max_Cs(Cth_list, z_cl_r)*Delta, 'r-.')
plt.plot(Delta, cal_max_Cs(Cq_list, z_ml_r)*Delta, 'b', label = '$l_{qt}$')
plt.plot(Delta, cal_max_Cs(Cq_list, z_cl_r)*Delta, 'b-.')
plt.plot(Delta, 0.23*Delta, 'g-', label = 'Standard $l_{mix}$')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Mixing Length (m)', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'l_mix_max_prof_ML_vs_CL_w_std.png', pad_inches=0)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_list, z_ml_r)*Delta, 'k', label = '$l_{mix}$')
plt.plot(Delta, cal_max_Cs(Cs_list, z_cl_r)*Delta, 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_list, z_ml_r)*Delta, 'r', label = '$l_{\\theta}$')
plt.plot(Delta, cal_max_Cs(Cth_list, z_cl_r)*Delta, 'r-.')
plt.plot(Delta, cal_max_Cs(Cq_list, z_ml_r)*Delta, 'b', label = '$l_{qt}$')
plt.plot(Delta, cal_max_Cs(Cq_list, z_cl_r)*Delta, 'b-.')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Mixing Length (m)', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'l_mix_max_prof_ML_vs_CL.png', pad_inches=0)
plt.close()





dir_data = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_cloud_v_env_'
os.makedirs(plotdir, exist_ok = True)


data_2D = xr.open_dataset(dir_data + '2D.nc')
data_4D = xr.open_dataset(dir_data + '4D.nc')
data_8D = xr.open_dataset(dir_data + '8D.nc')
data_16D = xr.open_dataset(dir_data + '16D.nc')
data_32D = xr.open_dataset(dir_data + '32D.nc')
data_64D = xr.open_dataset(dir_data + '64D.nc')


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



Cs_sq_cloud_2 = 0.5 * LijMij_cloud_2 / MijMij_cloud_2
Cs_sq_cloud_4 = 0.5 * LijMij_cloud_4 / MijMij_cloud_4
Cs_sq_cloud_8 = 0.5 * LijMij_cloud_8 / MijMij_cloud_8
Cs_sq_cloud_16 = 0.5 * LijMij_cloud_16 / MijMij_cloud_16
Cs_sq_cloud_32 = 0.5 * LijMij_cloud_32 / MijMij_cloud_32
Cs_sq_cloud_64 = 0.5 * LijMij_cloud_64 / MijMij_cloud_64

Cth_sq_cloud_2 = 0.5 * HjRj_th_cloud_2 / RjRj_th_cloud_2
Cth_sq_cloud_4 = 0.5 * HjRj_th_cloud_4 / RjRj_th_cloud_4
Cth_sq_cloud_8 = 0.5 * HjRj_th_cloud_8 / RjRj_th_cloud_8
Cth_sq_cloud_16 = 0.5 * HjRj_th_cloud_16 / RjRj_th_cloud_16
Cth_sq_cloud_32 = 0.5 * HjRj_th_cloud_32 / RjRj_th_cloud_32
Cth_sq_cloud_64 = 0.5 * HjRj_th_cloud_64 / RjRj_th_cloud_64

Cqt_sq_cloud_2 = 0.5 * HjRj_qt_cloud_2 / RjRj_qt_cloud_2
Cqt_sq_cloud_4 = 0.5 * HjRj_qt_cloud_4 / RjRj_qt_cloud_4
Cqt_sq_cloud_8 = 0.5 * HjRj_qt_cloud_8 / RjRj_qt_cloud_8
Cqt_sq_cloud_16 = 0.5 * HjRj_qt_cloud_16 / RjRj_qt_cloud_16
Cqt_sq_cloud_32 = 0.5 * HjRj_qt_cloud_32 / RjRj_qt_cloud_32
Cqt_sq_cloud_64 = 0.5 * HjRj_qt_cloud_64 / RjRj_qt_cloud_64
###############################################################


Cs_sq_env_2 = 0.5 * LijMij_env_2 / MijMij_env_2
Cs_sq_env_4 = 0.5 * LijMij_env_4 / MijMij_env_4
Cs_sq_env_8 = 0.5 * LijMij_env_8 / MijMij_env_8
Cs_sq_env_16 = 0.5 * LijMij_env_16 / MijMij_env_16
Cs_sq_env_32 = 0.5 * LijMij_env_32 / MijMij_env_32
Cs_sq_env_64 = 0.5 * LijMij_env_64 / MijMij_env_64

Cth_sq_env_2 = 0.5 * HjRj_th_env_2 / RjRj_th_env_2
Cth_sq_env_4 = 0.5 * HjRj_th_env_4 / RjRj_th_env_4
Cth_sq_env_8 = 0.5 * HjRj_th_env_8 / RjRj_th_env_8
Cth_sq_env_16 = 0.5 * HjRj_th_env_16 / RjRj_th_env_16
Cth_sq_env_32 = 0.5 * HjRj_th_env_32 / RjRj_th_env_32
Cth_sq_env_64 = 0.5 * HjRj_th_env_64 / RjRj_th_env_64

Cqt_sq_env_2 = 0.5 * HjRj_qt_env_2 / RjRj_qt_env_2
Cqt_sq_env_4 = 0.5 * HjRj_qt_env_4 / RjRj_qt_env_4
Cqt_sq_env_8 = 0.5 * HjRj_qt_env_8 / RjRj_qt_env_8
Cqt_sq_env_16 = 0.5 * HjRj_qt_env_16 / RjRj_qt_env_16
Cqt_sq_env_32 = 0.5 * HjRj_qt_env_32 / RjRj_qt_env_32
Cqt_sq_env_64 = 0.5 * HjRj_qt_env_64 / RjRj_qt_env_64


###################################################################################################

beta_s_cloud = dyn.beta_calc(Cs_sq_cloud_2, Cs_sq_cloud_4)
beta_th_cloud = dyn.beta_calc(Cth_sq_cloud_2, Cth_sq_cloud_4)
beta_q_cloud = dyn.beta_calc(Cqt_sq_cloud_2, Cqt_sq_cloud_4)

beta_s_env = dyn.beta_calc(Cs_sq_env_2, Cs_sq_env_4)
beta_th_env = dyn.beta_calc(Cth_sq_env_2, Cth_sq_env_4)
beta_q_env = dyn.beta_calc(Cqt_sq_env_2, Cqt_sq_env_4)



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



Cs_cloud_list = [Cs_cloud_beta, Cs_cloud_2, Cs_cloud_4, Cs_cloud_8, Cs_cloud_16, Cs_cloud_32, Cs_cloud_64]
Cth_cloud_list = [Cth_cloud_beta, Cth_cloud_2, Cth_cloud_4, Cth_cloud_8, Cth_cloud_16, Cth_cloud_32, Cth_cloud_64]
Cqt_cloud_list = [Cqt_cloud_beta, Cqt_cloud_2, Cqt_cloud_4, Cqt_cloud_8, Cqt_cloud_16, Cqt_cloud_32, Cqt_cloud_64]

Cs_env_list = [Cs_env_beta, Cs_env_2, Cs_env_4, Cs_env_8, Cs_env_16, Cs_env_32, Cs_env_64]
Cth_env_list = [Cth_env_beta, Cth_env_2, Cth_env_4, Cth_env_8, Cth_env_16, Cth_env_32, Cth_env_64]
Cqt_env_list = [Cqt_env_beta, Cqt_env_2, Cqt_env_4, Cqt_env_8, Cqt_env_16, Cqt_env_32, Cqt_env_64]



plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_cloud_list, z_ml_r), 'k', label = '$C_{s}$')
plt.plot(Delta, cal_max_Cs(Cs_env_list, z_cl_r), 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_cloud_list, z_ml_r), 'r', label = '$C_{\\theta}$')
plt.plot(Delta, cal_max_Cs(Cth_env_list, z_cl_r), 'r-.')
plt.plot(Delta, cal_max_Cs(Cqt_cloud_list, z_ml_r), 'b', label = '$C_{qt}$')
plt.plot(Delta, cal_max_Cs(Cqt_env_list, z_cl_r), 'b-.')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Smagorinsky Parameter in the cloud layer', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'C_max_prof_Env_vs_CL.png', pad_inches=0)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_cloud_list, z_ml_r)*Delta, 'k', label = '$l_{mix}$')
print(cal_max_Cs(Cs_cloud_list, z_ml_r)*Delta)
plt.plot(Delta, cal_max_Cs(Cs_env_list, z_cl_r)*Delta, 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_cloud_list, z_ml_r)*Delta, 'r', label = '$l_{\\theta}$')
print(cal_max_Cs(Cth_cloud_list, z_ml_r)*Delta)
plt.plot(Delta, cal_max_Cs(Cth_env_list, z_cl_r)*Delta, 'r-.')
plt.plot(Delta, cal_max_Cs(Cqt_cloud_list, z_ml_r)*Delta, 'b', label = '$l_{qt}$')
print(cal_max_Cs(Cqt_cloud_list, z_ml_r)*Delta)
plt.plot(Delta, cal_max_Cs(Cqt_env_list, z_cl_r)*Delta, 'b-.')
plt.plot(Delta, 0.23*Delta, 'g-', label = 'Standard $l_{mix}$')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Mixing Length (m) in the cloud layer', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'l_mix_max_prof_Env_vs_CL_w_std.png', pad_inches=0)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(Delta, cal_max_Cs(Cs_cloud_list, z_ml_r)*Delta, 'k', label = '$l_{mix}$')
plt.plot(Delta, cal_max_Cs(Cs_env_list, z_cl_r)*Delta, 'k-.')
plt.plot(Delta, cal_max_Cs(Cth_cloud_list, z_ml_r)*Delta, 'r', label = '$l_{\\theta}$')
plt.plot(Delta, cal_max_Cs(Cth_env_list, z_cl_r)*Delta, 'r-.')
plt.plot(Delta, cal_max_Cs(Cqt_cloud_list, z_ml_r)*Delta, 'b', label = '$l_{qt}$')
plt.plot(Delta, cal_max_Cs(Cqt_env_list, z_cl_r)*Delta, 'b-.')
plt.xlabel('Grid spacing $\\Delta$ (m)', fontsize=16)
plt.ylabel('Mixing Length (m) in the cloud layer', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.savefig(plotdir+'l_mix_max_prof_Env_vs_CL.png', pad_inches=0)
plt.close()



