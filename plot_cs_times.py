import matplotlib.pyplot as plt
import numpy as np
import os
import time_av_dynamic as t_dy

filedir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/files/t_av_4_times/'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/trace/t_av_4_times/'

new_set_time = ['12600', '14400', '16200', '18000']
times = 3

Cs_2D_prof = np.zeros((len(new_set_time)*times, len(Cs_2D_prof_t0)))
Cs_4D_prof = np.zeros((len(new_set_time)*times, len(Cs_2D_prof_t0)))

for k, time_file in enumerate(new_set_time):
    Cs_2D_prof[(k*times + l), :] = np.load(filedir + f'Cs_2D_prof_t{time_file}_{l}')
    Cs_4D_prof[(k*times + l), :] = np.load(filedir + f'Cs_4D_prof_t{time_file}_{l}')

Cs_2D_prof_av = np.mean(Cs_2D_prof, axis=0)
Cs_4D_prof_av = np.mean(Cs_4D_prof, axis=0)

#########################plots#########################

z = np.arange(0,3020,20)

fig = plt.figure(figsize=(10, 8))
for n in range(len(Cs_2D_prof[:,0])):
    plt.plot(Cs_2D_prof[n,:], z / 500, markersize=6, label=f't{n}')
plt.plot(Cs_2D_prof_av, z / 500, 'k-*', markersize=6, label='av')
plt.xlim(-0.01, 0.2)
plt.title('$C_{s 2 \\Delta} $')
plt.ylabel('z (m)')
plt.xlabel('$ C_{s} $', fontsize=14)
plt.legend(fontsize=16, loc='best')
plt.savefig(plotdir + "Cs_profiles_2D_scaled_" + str(av_type) + ".png", pad_inches=0)
plt.clf()
plt.cla()
plt.close()

fig2 = plt.figure(figsize=(10, 8))
for o in range(len(Cs_4D_prof[:,0])):
    plt.plot(Cs_4D_prof[o,:], z/500, markersize=6, label=f't{o}')
plt.plot(Cs_4D_prof_av, z / 500, 'k-*', markersize=6, label='av')
plt.xlim(-0.01, 0.2)
plt.title('$C_{s 4 \\Delta} $')
plt.ylabel('$z/z_i')
plt.xlabel('$ C_{s} $', fontsize=14)
plt.legend(fontsize=16, loc='best')
plt.savefig(plotdir + "Cs_profiles_4D_scaled_" + ".png", pad_inches=0)
plt.clf()
plt.cla()
plt.close()

fig3 = plt.figure(figsize=(10, 8))
plt.plot(Cs_2D_prof_av, z / 500, 'k-*', markersize=6, label='$Av C_{s 2 \\Delta} $')
plt.plot(Cs_4D_prof_av, z / 500, 'k-*', markersize=6, label='$Av C_{s 4 \\Delta} $')
plt.xlim(-0.01, 0.2)
plt.ylabel('$z/z_i')
plt.xlabel('$ C_{s} $', fontsize=14)
plt.legend(fontsize=16, loc='best')
plt.savefig(plotdir + "Cs_profiles_2D_4D_scaled_" + ".png", pad_inches=0)
plt.clf()
plt.cla()
plt.close()
