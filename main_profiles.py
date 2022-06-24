import numpy as np
import time_av_profiles as avp
import matplotlib.pyplot as plt

 #'dry_cbl'

set_time = '14400'
#set_time='13200'

mydir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
mydir_filt = '/work/scratch-pw/apower/20m_gauss_dyn_update_subfilt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
#mydir='/storage/silver/scenario/si818415/phd/'

model_res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
model_res_list_int = np.array([20, 40, 80])

filter_res_list = ['0', '1']
filter_res_list_int = np.array([20, 40, 80])

# model_res_list = ['0020_g0800', '0025_g0640', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
# model_res_list_int = np.array([20, 25, 40, 80, 160, 320])

# model_res_list = ['20', '50', '100', '200', '400', '800']
# model_res_list_int = np.array([20, 50, 100, 200, 400, 800])



########################### mean profiles #########################################


# myvars = ['u_wind_mean','v_wind_mean','w_wind_mean','ww_mean','theta_mean','vapour_mmr_mean', \
#           'liquid_mmr_mean','wtheta_ad_mean','wtheta_cn_mean','wql_ad_mean','wql_cn_mean',\
#           'wqv_ad_mean', 'wqv_cn_mean', 'total_cloud_fraction']

myvars = ['w_wind_mean','ww_mean','theta_mean', 'wtheta_cn_mean', 'wql_cn_mean',\
          'wqv_cn_mean', 'total_cloud_fraction']

avp.time_av_prof(myvars, model_res_list, set_time, mydir, 'bomex_og')

my_vars_filt = ['f(w_on_w.w_on_w)_r', 'f(w_on_w.th_on_w)_r', 'f(w_on_w.q_total_on_w)_r']
avp.time_av_prof(my_vars_filt, filter_res_list, set_time, mydir_filt, 'bomex_filt')

plot_vars = []
plot_vars.append(myvars)
plot_vars.append('wqt')
plot_vars.append(my_vars_filt)

for i, var in enumerate(plot_vars):

    figs = plt.figure(figsize=(6, 6))
    plt.ylabel(r'z')

    if var == 'w_wind_mean':
        var_name = "$\overline{w'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'ww_mean':
        var_name = "$\overline{w'^2}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'theta_mean':
        var_name = "$\overline{\\theta'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'wtheta_cn_mean':
        var_name = "$\overline{w' \\theta'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'total_cloud_fraction':
        var_name = "Total cloud fraction"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'wqv_cn_mean':
        var_name = "$\overline{w' q_v'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'wql_cn_mean':
        var_name = "$\overline{w' q_l'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']
    elif var == 'wqt':
        var_name = "$\overline{w' q_t'}$"
        mydata = 'bomex_og'
        res_list = ['0020_g0800', '0040_g0400', '0080_g0200']

    elif var == 'f(w_on_w.w_on_w)_r':
        var_name = "$\overline{w'^2}$"
        mydata = 'bomex_filt'
        res_list = ['0020_g0800', '0', '1']
    elif var == 'f(w_on_w.th_on_w)_r':
        var_name = "$\overline{w' \\theta'}$"
        mydata = 'bomex_filt'
        res_list = ['0020_g0800', '0', '1']
    elif var == 'f(w_on_w.q_total_on_w)_r':
        var_name = "$\overline{w' q_t'}$"
        mydata = 'bomex_filt'
        res_list = ['0020_g0800', '0', '1']

    else:
        print('name not configured for this variable')

    plt.xlabel(f'{var_name}')

    for m, res in enumerate(res_list):
        if mydata=='bomex_filt':
            if m == 0:
                var_plot = np.load(f'files/bomex_og/{res}_{var}.npy')
        else:
            var_plot = np.load(f'files/{mydata}/{res}_{var}.npy')
        z_plot = np.load(f'files/{mydata}/{res}_z.npy')
        plt.plot(var_plot, z_plot, label=f'$\\Delta x$ = {str(model_res_list_int[m])} m')
    plt.legend(fontsize=12)
    figs.savefig(f'plots/{mydata}/{var}_{set_time}_profile.png')
    print("Finished plotting profile for ", var, ",", \
          len(model_res_list)-(m+1), "variables remaining")

