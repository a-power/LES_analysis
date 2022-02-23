import numpy as np
import time_av_profiles as avp
import matplotlib.pyplot as plt

#set_time = '14400'
set_time='13200'

#mydir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
mydir='/storage/silver/scenario/si818415/phd/'

#model_res_list = ['0020_g0800', '0025_g0640', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
#model_res_list_int = np.array([20, 25, 40, 80, 160, 320])

model_res_list = ['20', '50', '100', '200', '400', '800']
model_res_list_int = np.array([20, 50, 100, 200, 400, 800])



########################### mean profiles #########################################


# myvars = ['u_wind_mean','v_wind_mean','w_wind_mean','ww_mean','theta_mean','vapour_mmr_mean', \
#           'liquid_mmr_mean','wtheta_ad_mean','wtheta_cn_mean','wql_ad_mean','wql_cn_mean',\
#           'wqv_ad_mean', 'wqv_cn_mean', 'total_cloud_fraction']

myvars = ['wtheta_ad_mean']

avp.time_av_prof(myvars, model_res_list, set_time, mydir, 'dry_cbl')

for i, var in enumerate(myvars):

    figs = plt.figure(figsize=(6, 6))
    plt.ylabel(r'z')
    plt.xlabel(f'{var}')

    for m, res in enumerate(model_res_list):
        var_plot = np.load(f'files/{res}_{var}_time_av.npy')
        z_plot = np.load(f'files/{res}_z.npy')
        plt.plot(var_plot, z_plot, label=f'$\\Delta x$ = {str(model_res_list_int[m])} m')
    plt.legend(fontsize=12)
    figs.savefig(f'plots/{var}_{set_time}_time_av_profile.png')
    print("Finished plotting profile for ", var, ",", \
          len(model_res_list)-(m+1), "variables remaining")

