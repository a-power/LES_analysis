import matplotlib.pyplot as plt
import time_av_dynamic as tdy #dot to get folder outside
import numpy as np
import os
import time_av_dynamic as t_dy
import xarray as xr

set_time = ['14400'] # ,'12600', '16200', '18000'
in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
model_res_list = ['0020_g0800']
outdir_og = '/work/scratch-pw/apower/'
outdir = outdir_og + '20m_gauss_dyn_update_subfilt' +'/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40])
opgrid = 'w'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
                tdy.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options, \
                            opgrid, domain_in=16, ref_file = None)



###############################################################################################################

# new_set_time = ['12600', '14400', '16200', '18000']
# av_type = 'all'
# mygrid = 'w'
#
#
#
# plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/trace/t_av_4_times/'
# filedir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/files/t_av_4_times/'
# count = 0
# for k, time_file in enumerate(new_set_time):
#     if k==1:
#         path20f = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn_w_trace/'
#     else:
#         path20f = '/work/scratch-pw/apower/20m_gauss_dyn_w_trace/'
#     file20 = f"BOMEX_m0020_g0800_all_{time_file}_filter_"
#
#     data_2D = path20f + file20 + str('ga00.nc')
#     data_4D = path20f + file20 + str('ga01.nc')
#
#     Cs_2D_prof_t0, times = t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=0, save_all=1)
#
#     Cs_2D_prof = np.zeros((len(new_set_time)*len(times), len(Cs_2D_prof_t0)))
#     Cs_4D_prof = np.zeros((len(new_set_time)*len(times), len(Cs_2D_prof_t0)))
#
#     Cs_2D_prof[0,:] = Cs_2D_prof_t0
#
#     for l in range(1,len(times)):
#         Cs_2D_prof[(k*len(times) + l), :] = t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=l, save_all=0)
#     for m in range(len(times)):
#         Cs_4D_prof[(k*len(times) + m), :] = t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, t_in=m, save_all=0)
#
# np.save(filedir + 'Cs_2D_prof', Cs_2D_prof)
# np.save(filedir + 'Cs_4D_prof', Cs_4D_prof)
#
# Cs_2D_prof_av = np.mean(Cs_2D_prof, axis=0)
# Cs_4D_prof_av = np.mean(Cs_4D_prof, axis=0)

#########################plots#########################
#
# z = np.arange(0,3020,20)
#
# fig = plt.figure(figsize=(10, 8))
# for n in range(len(Cs_2D_prof[:,0])):
#     plt.plot(Cs_2D_prof[n,:], z / 500, markersize=6, label=f't{n}')
# plt.plot(Cs_2D_prof_av, z / 500, 'k-*', markersize=6, label='av')
# plt.xlim(-0.01, 0.2)
# plt.title('$C_{s 2 \\Delta} $')
# plt.ylabel('z (m)')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='best')
# plt.savefig(plotdir + "Cs_profiles_2D_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig2 = plt.figure(figsize=(10, 8))
# for o in range(len(Cs_4D_prof[:,0])):
#     plt.plot(Cs_4D_prof[o,:], z/500, markersize=6, label=f't{o}')
# plt.plot(Cs_4D_prof_av, z / 500, 'k-*', markersize=6, label='av')
# plt.xlim(-0.01, 0.2)
# plt.title('$C_{s 4 \\Delta} $')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='best')
# plt.savefig(plotdir + "Cs_profiles_4D_scaled_" + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig3 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_av, z / 500, 'k-*', markersize=6, label='$Av C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_av, z / 500, 'k-*', markersize=6, label='$Av C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='best')
# plt.savefig(plotdir + "Cs_profiles_2D_4D_scaled_" + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
#
#
#
#
#
