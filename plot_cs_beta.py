import matplotlib.pyplot as plt
import numpy as np
import dynamic_functions as dy

filedir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/files/t_av_4_times/'
plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/trace/t_av_4_times/'

time_file = '14400'
times = 3
z_len = 151

Cs_2D_prof = np.zeros((times, z_len))
Cs_4D_prof = np.zeros((times, z_len))

for l in range(times):
    Cs_2D_prof[(l), :] = np.load(filedir + f'Cs_2D_prof_t{time_file}_{l}.npy')
    Cs_4D_prof[(l), :] = np.load(filedir + f'Cs_4D_prof_t{time_file}_{l}.npy')

Cs_2D_prof_av = np.mean(Cs_2D_prof, axis=0)
Cs_4D_prof_av = np.mean(Cs_4D_prof, axis=0)

beta = dy.beta_calc((Cs_2D_prof_av**2), (Cs_4D_prof_av**2))
Cs_beta = dy.Cs_beta((Cs_2D_prof_av**2), beta)

np.save(filedir + f'Cs_beta_prof_t{time_file}_av', Cs_beta)

#########################plots#########################

z = np.arange(0,3020,20)

fig3 = plt.figure(figsize=(10, 8))
plt.plot(Cs_2D_prof_av, z / 500, 'b-', markersize=6, label='$Av C_{s 2 \\Delta} $')
plt.plot(Cs_4D_prof_av, z / 500, 'r-', markersize=6, label='$Av C_{s 4 \\Delta} $')
plt.plot(Cs_beta, z / 500, 'k-', markersize=6, label='$Av C_{s \\beta} $')
plt.xlim(-0.01, 0.22)
plt.ylabel('$z/z_i$')
plt.xlabel('$ C_{s} $', fontsize=14)
plt.legend(fontsize=16, loc='best')
plt.savefig(plotdir + f"Cs_profiles_beta_2D_4D_scaled_t{time_file}.png", pad_inches=0)
plt.clf()
plt.cla()
plt.close()
