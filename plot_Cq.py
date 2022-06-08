import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

dir_data = '/work/scratch-pw/apower/20m_gauss_dyn_update_subfilt/BOMEX_m0020_g0800_all_14400_gaussian_filter_Cq_'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_update_subfilt/plots/'
os.makedirs(plotdir, exist_ok = True)

data_2D = xr.open_dataset(dir_data + '2D_attempt.nc')
data_4D = xr.open_dataset(dir_data + '4D_attempt.nc')

Cs_2 = data_2D['C_q_total_prof'].data[0, ...]
Cs_4 = data_4D['C_q_total_prof'].data[0, ...]

z = np.arange(0, 3020, 20)

plt.figure(figsize=(6,7))
plt.plot(Cs_2, z, label = '$C_{q 2 \\Delta}$')
plt.plot(Cs_4, z, label = '$C_{q 4 \\Delta}$')
plt.xlabel('$C_{q}$', fontsize=16)
plt.ylabel("z (m)")
plt.legend(fontsize=16, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'C_q_prof.png')


Cs_sq_2 = data_2D['C_q_total_sq_prof'].data[0, ...]
Cs_sq_4 = data_4D['C_q_total_sq_prof'].data[0, ...]

Cs_sq_field_2 = data_2D['C_q_total_sq_field'].data[0, ...]
Cs_sq_field_4 = data_4D['C_q_total_sq_field'].data[0, ...]


def beta_calc(C_2D_sq_in, C_4D_sq_in):
    Cs_2D_sq_copy1 = C_2D_sq_in.copy()
    Cs_2D_sq_copy2 = C_2D_sq_in.copy()
    Cs_4D_sq_copy = C_4D_sq_in.copy()

    Cs_2D_sq_copy2[Cs_2D_sq_copy1 == 0.00000] = 1
    Cs_4D_sq_copy[Cs_2D_sq_copy1 == 0.00000] = 500  # remain as scale dependant

    beta = Cs_4D_sq_copy / Cs_2D_sq_copy2
    beta[beta < 0.125] = 0.125
    # beta[beta > 5] = 5

    return beta

beta = beta_calc(Cs_sq_2, Cs_sq_4)

plt.figure(figsize=(6,8))
plt.plot(beta, z)
plt.xlabel('$\\beta$', fontsize=14)
plt.ylabel("z (m)")
plt.xlim(0, 2)
plt.savefig(plotdir+'Cq_beta_prof_zoom.png')

Cs_beta_sq = Cs_sq_2/beta

plt.figure(figsize=(6,7))
plt.plot(Cs_beta_sq, z)
plt.xlabel('$C_{q \\beta} ^2$', fontsize=14)
plt.ylabel("z (m)")
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cq_beta_sq_prof.png')



def get_Cs(Cs_sq):
    """ calculates C_s from C_s^2 by setting neg values to zero
    and sq rooting"""

    Cs_sq_copy = Cs_sq.copy()
    Cs_sq_copy[Cs_sq < 0] = 0
    Cs = np.sqrt(Cs_sq_copy)

    return Cs

Cs_beta = get_Cs(Cs_beta_sq)
#
# Cs_2 = data_2D['C_q_total_prof'].data[0, ...]
# Cs_4 = data_4D['C_q_total_prof'].data[0, ...]

plt.figure(figsize=(6,7))
plt.plot(Cs_beta, z)
plt.xlabel('$C_{q \\beta}$', fontsize=14)
plt.ylabel("z (m)")
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cq_beta_prof.png')

plt.figure(figsize=(6,7))
plt.plot(Cs_2, z, label = '$C_{q 2 \\Delta}$')
plt.plot(Cs_4, z, label = '$C_{q 4 \\Delta}$')
plt.plot(Cs_beta, z, label = '$C_{q \\beta}$')
plt.xlabel('$C_{q}$', fontsize=14)
plt.ylabel("z (m)")
plt.legend(fontsize=16, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'Cq_prof_w_beta.png')


plt.figure(figsize=(6,7))
plt.plot(Cs_2*(40), z, label = '$l_{q 2 \\Delta}$')
plt.plot(Cs_4*(80), z, label = '$l_{q 4 \\Delta}$')
plt.plot(Cs_beta*(20), z, label = '$l_{q \\beta}$')
plt.xlabel('$l_{q}$', fontsize=14)
plt.ylabel("z (m)")
plt.legend(fontsize=16, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'l_q.png')



def l_mix_MONC(Cs, Delta, z_in, k=0.4, Pr=1):

    l_mix = np.sqrt( 1 / ( (1/((Cs*Cs)/(Pr*Pr) * Delta*Delta)) + (1/(k*k * z_in*z_in)) ) )

    return l_mix



monc_l_20 = l_mix_MONC(0.23, 20, z, k=0.4, Pr=0.7)
monc_l_40 = l_mix_MONC(0.23, 40, z, k=0.4, Pr=0.7)
monc_l_80 = l_mix_MONC(0.23, 80, z, k=0.4, Pr = 0.7)

plt.figure(figsize=(6,6))
plt.plot(Cs_2*(40), z, '.', markersize = 10, label = '$l_{mix 2 \\Delta}$')
plt.plot(Cs_4*(80), z, '.', markersize = 10, label = '$l_{mix 4 \\Delta}$')
plt.plot(Cs_beta*(20), z, '.', markersize = 10, label = '$l_{mix \\beta}$')

plt.plot(monc_l_40, z, 'tab:blue')
plt.plot(monc_l_80, z, 'tab:orange')
plt.plot(monc_l_20, z, 'tab:green')

plt.xlabel('$l_{\\theta}$', fontsize=16)
plt.ylabel("z (m)")
plt.legend(fontsize=16, loc='upper right')
#plt.xlim(1, 3)
plt.savefig(plotdir+'l_q_w_MONC.png')