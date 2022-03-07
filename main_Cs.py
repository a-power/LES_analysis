import time_av_dynamic as t_dy
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/'
path20f = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn_p/'
file20 = "BOMEX_m0020_g0800_all_14400_filter_"

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')


Cs_2D_av, Cs_2D_av_field = t_dy.time_av_Cs(data_2D, dx=20, dx_hat=40)
Cs_4D_av, Cs_4D_av_field = t_dy.time_av_Cs(data_4D, dx=20, dx_hat=80)


#########################plots#########################

fig = plt.figure(figsize=(12, 10))
# plt.plot(Cs_t_av_prof, z, '-', markersize=6, label='$C_{s \\beta}$')
# plt.plot(10, 10, 'o', markersize=6) #get correct colour
plt.plot(Cs_2D_av, z, '-', markersize=6, label='$ C_{s 2 \\Delta} $')
plt.plot(Cs_4D_av, z, '-', markersize=6, label='$ C_{s 4 \\Delta} $')
# plt.plot(Cs_8D_av, z, '-', markersize=6, label='$ C_{s 8 \\Delta} $')
# plt.plot(Cs_16D_av, z, '-', markersize=6, label='$ C_{s 16 \\Delta} $')
plt.xlim(-0.01, 0.25)
plt.ylabel('z (m)')
plt.xlabel('$ C_{s} $', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.savefig(plotdir + "Cs_profiles_t_av_line_" + str(av_type) + ".png", pad_inches=0)
plt.show()