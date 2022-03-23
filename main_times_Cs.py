import matplotlib.pyplot as plt
import numpy as np
import os
import time_av_dynamic as t_dy

new_set_time = ['12600', '14400', '16200', '18000']
av_type = 'all'
mygrid = 'w'


plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/trace/t_av_4_times/'
filedir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/files/t_av_4_times/'

for k, time_file in enumerate(new_set_time):
    if k==1:
        path20f = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn_w_trace/'
    else:
        path20f = '/work/scratch-pw/apower/20m_gauss_dyn_w_trace/'
    file20 = f"BOMEX_m0020_g0800_all_{time_file}_filter_"

    data_2D = path20f + file20 + str('ga00.nc')
    data_4D = path20f + file20 + str('ga01.nc')

    Cs_2D_prof_t0, times = t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=0, save_all=1)
    np.save(filedir + f'Cs_2D_prof_t{time_file}_0', Cs_2D_prof_t0)

    # Cs_2D_prof = np.zeros((len(new_set_time)*len(times), len(Cs_2D_prof_t0)))
    # Cs_4D_prof = np.zeros((len(new_set_time)*len(times), len(Cs_2D_prof_t0)))
    #
    # Cs_2D_prof[0,:] = Cs_2D_prof_t0

    for l in range(1,len(times)):
        # Cs_2D_prof[(k*len(times) + l), :] = t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=l, save_all=0)
        Cs_2D_prof = t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid=mygrid, t_in=l, save_all=0)
        np.save(filedir + f'Cs_2D_prof_t{time_file}_{l}', Cs_2D_prof)
    print('finished 2D')
    for m in range(len(times)):
        # Cs_4D_prof[(k*len(times) + m), :] = t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, t_in=m, save_all=0)
        Cs_4D_prof = t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid=mygrid, t_in=m, save_all=0)
        np.save(filedir + f'Cs_4D_prof_t{time_file}_{m}', Cs_4D_prof)
    print('Finished 4D')

# np.save(filedir + 'Cs_2D_prof', Cs_2D_prof)
# np.save(filedir + 'Cs_4D_prof', Cs_4D_prof)

# Cs_2D_prof_av = np.mean(Cs_2D_prof, axis=0)
# Cs_4D_prof_av = np.mean(Cs_4D_prof, axis=0)