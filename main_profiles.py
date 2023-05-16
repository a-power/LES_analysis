import numpy as np
import MONC_out_profiles as avp
import matplotlib.pyplot as plt
import os

 #'dry_cbl'

set_time = '14400'
#set_time='13200'

mydir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
mydir_filt = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'
#mydir='/storage/silver/scenario/si818415/phd/'
plot_dir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/plots/coarse_data/profiles/'
os.makedirs(plot_dir, exist_ok = True)

model_res_list = ['0020_g0800', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
model_res_list_int = np.array([20, 40, 80, 160, 320])

filter_res_list = ['0', '1', '2', '3', '4', '5']
filter_res_list_int = np.array([40, 80, 160, 320, 640, 1280])




########################### mean profiles #########################################


# myvars = ['u_wind_mean','v_wind_mean','w_wind_mean','ww_mean','theta_mean','vapour_mmr_mean', \
#           'liquid_mmr_mean','wtheta_ad_mean','wtheta_cn_mean','wql_ad_mean','wql_cn_mean',\
#           'wqv_ad_mean', 'wqv_cn_mean', 'total_cloud_fraction']

# myvars = ['w_wind_mean','ww_mean','theta_mean', 'wtheta_cn_mean', 'wql_cn_mean', 'vapour_mmr_mean', \
#            'liquid_mmr_mean', 'wqv_cn_mean', 'total_cloud_fraction']

myvars = ['theta_mean', 'wtheta_cn_mean', 'total_cloud_fraction']

#avp.time_av_prof(myvars, model_res_list, set_time, mydir, 'bomex_og')




my_vars_filt = ['f(w_on_w.w_on_w)_r', 'f(w_on_w.th_on_w)_r', 'f(w_on_w.q_total_on_w)_r', \
                'f(w_on_w.q_cloud_liquid_mass_on_w)_r', 'f(q_cloud_liquid_mass_on_w)_r',\
                'f(q_total_on_w)_r', 'f(q_vapour_on_w)_r', 'f(w_on_w.q_vapour_on_w)_r', \
                'f(th_on_w)_r', 'f(th_v_on_w)_r', 'f(th_L_on_w)_r', 'f(th_L_on_w.th_L_on_w)_r',\
                'f(th_L_on_w.q_cloud_liquid_mass_on_w)_r', 'f(th_L_on_w.q_total_on_w)_r', \
                'f(th_L_on_w.q_vapour_on_w)_r']

#avp.time_av_prof(my_vars_filt, filter_res_list, set_time, mydir_filt, 'bomex_filt')

plot_vars = []
plot_vars.extend(myvars)
plot_vars.append('wqt')
plot_vars.extend(my_vars_filt)

res_list1 = ['0020_g0800', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
res_list2 = ['0', '1', '2', '3', '4', '5']

for i, var in enumerate(plot_vars):

    figs = plt.figure(figsize=(6, 7))
    plt.ylabel(r'z')

    if var == 'w_wind_mean':
        var_name = "$\overline{w'}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'ww_mean':
        var_name = "$\overline{w'^2}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'theta_mean':
        var_name = "$\overline{\\theta'}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'wtheta_cn_mean':
        var_name = "$\overline{w' \\theta'}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'total_cloud_fraction':
        var_name = "Total cloud fraction"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'wqv_cn_mean':
        var_name = "$\overline{w' q_v'}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'wql_cn_mean':
        var_name = "$\overline{w' q_l'}$"
        mydata = 'bomex_og'
        res_list = res_list1
    elif var == 'wqt':
        var_name = "$\overline{w' q_t'}$"
        mydata = 'bomex_og'
        res_list = res_list1

    elif var == 'f(w_on_w.w_on_w)_r':
        var_name = "$\overline{w'^2}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(w_on_w.th_on_w)_r':
        var_name = "$\overline{w' \\theta'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(w_on_w.q_total_on_w)_r':
        var_name = "$\overline{w' q_t'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(q_total_on_w.q_total_on_w)_r':
        var_name = "$\overline{q_t'^2}$"
        mydata = 'bomex_filt'
        res_list = res_list2

    elif var == 'f(w_on_w.q_cloud_liquid_mass_on_w)_r':
        var_name = "$\overline{w' q_L'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(q_cloud_liquid_mass_on_w)_r':
        var_name = "$\overline{q_L'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(q_total_on_w)_r':
        var_name = "$\overline{q_t'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(q_vapour_on_w)_r':
        var_name = "$\overline{q_v'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(w_on_w.q_vapour_on_w)_r':
        var_name = "$\overline{w' q_v'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_on_w)_r':
        var_name = "$\overline{\\theta'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_v_on_w)_r':
        var_name = "$\overline{\\theta_v'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_L_on_w)_r':
        var_name = "$\overline{\\theta_L'}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_on_w.th_on_w)_r':
        var_name = "$\overline{\\theta'^2}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_v_on_w.th_v_on_w)_r':
        var_name = "$\overline{\\theta_v'^2}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_L_on_w.th_L_on_w)_r':
        var_name = "$\overline{\\theta_L'^2}$"
        mydata = 'bomex_filt'
        res_list = res_list2

    elif var == 'f(th_L_on_w.q_cloud_liquid_mass_on_w)_r':
        var_name = "$\overline{\\theta_L' q_L}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_L_on_w.q_total_on_w)_r':
        var_name = "$\overline{\\theta_L' q_t}$"
        mydata = 'bomex_filt'
        res_list = res_list2
    elif var == 'f(th_L_on_w.q_vapour_on_w)_r':
        var_name = "$\overline{\\theta_L' q_v}$"
        mydata = 'bomex_filt'
        res_list = res_list2



    else:
        print(var, ' does not have a name configured for this variable')

    plt.xlabel(f'{var_name}')

    for m, res in enumerate(res_list):
        # if mydata=='bomex_filt':
        #     if m == 0:
        #         if var == 'f(w_on_w.w_on_w)_r':
        #             var_out = 'ww_mean'
        #         elif var == 'f(w_on_w.th_on_w)_r':
        #             var_out = 'wtheta_cn_mean'
        #         elif var == 'f(w_on_w.q_total_on_w)_r':
        #             var_out = 'wqt'
        #         else:
        #             print(var, 'not programed correctly')
        #         var_plot = np.load(f'files/bomex_og/{res}_{var_out}.npy')
        # else:
        var_plot = np.load(f'files/{mydata}/{res}_{var}.npy')

        # if mydata=='bomex_filt':
        #     if m == 0:
        #         z_plot = np.load(f'files/bomex_og/{res}_z.npy')
        # else:
        z_plot = np.load(f'files/{mydata}/{res}_z.npy')
        if len(np.shape(var_plot)) != 1:
            if mydata == 'bomex_filt':
                plt.plot(np.mean(var_plot, axis=0), z_plot, label=f'$\\Delta x$ = {str(filter_res_list_int[m])} m')
            else:
                plt.plot(np.mean(var_plot, axis=0), z_plot, label=f'$\\Delta x$ = {str(model_res_list_int[m])} m')

        else:
            if mydata == 'bomex_filt':
                plt.plot(var_plot, z_plot, label=f'$\\Delta x$ = {str(filter_res_list_int[m])} m')
            else:
                plt.plot(var_plot, z_plot, label=f'$\\Delta x$ = {str(model_res_list_int[m])} m')
    plt.legend(fontsize=12)
    figs.savefig(plot_dir+f'{var}_{set_time}_profile.png')
    print("Finished plotting profile for ", var, ",", \
          len(res_list)-(m+1), "variables remaining")
    plt.close()

