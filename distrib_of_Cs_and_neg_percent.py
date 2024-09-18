import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import mask_cloud_vs_env as clo
import matplotlib.ticker as ticker
import numpy.ma as ma


BOMEX_homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/LM/update/BOMEX_m0020_g0800_all_14400_'
BOMEX_dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'


ARM_homedir = f'/work/scratch-pw3/apower/ARM/second_filt/LM/update/diagnostics_3d_ts_{set_time}_'
ARM_dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_{set_time}_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/distribs/'
#'/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir, exist_ok = True)

cloud_field = f'f(q_cloud_liquid_mass_on_{mygrid})_r'

fields = [ ['LM_field', 'MM_field'], ['HR_th_field', 'RR_th_field'],
           ['HR_th_L_field', 'RR_th_L_field'], ['HR_q_field', 'RR_q_field'] ]


gen_options = {'deltas': None,
            'cloud_liquid_threshold_in': 10**(-7),
            'times': -1,
            'grid': 'p',
            'return_all_in': False,
            'set_bins':50
            }

field_names = ['Cs_field', 'Cth_field', 'Cth_L_field', 'Cqt_field']
field_latex = ['$C_{s}$', '$C_{\\theta}$', '$C_{\\theta_L}$', '$C_{q_t}$']

field_names_sq = ['Cs_sq_field', 'Cth_sq_field', 'Cth_L_sq_field', 'Cqt_sq_field']
field_latex_sq = ['$C_{s}^2$', '$C_{\\theta}^2$', '$C_{\\theta_L}^2$', '$C_{q_t}^2$']




def plot_hist(plotdir_in, delta, data1, data2, data3, data4, data5, what_plotting, bins_in):

    colours = ['tab:blue', 'tab:brown', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple',
               'tab:olive', 'tab:cyan', 'tab:gray', 'tab:pink']

    if what_plotting == 'C':
        fields_latex_in = field_latex
    else:
        fields_latex_in = field_latex_sq


    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 18), sharex='row', sharey='row')
    for i in range(4):
        for j in range(3):
            ax[i,j].hist(data1[i,j,:].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[0],
                     weights=np.ones(len(data1[i,j,:])) / len(data1[i,j,:]), label='BOMEX')
            ax[i,j].hist(data2[i,j,:].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[1],
                     weights=np.ones(len(data2[i,j,:])) / len(data2[i,j,:]), label='ARM 10:30L')
            ax[i,j].hist(data3[i,j,:].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[2],
                     weights=np.ones(len(data3[i,j,:])) / len(data3[i,j,:]), label='ARM 12:30L')
            ax[i,j].hist(data4[i,j,:].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[3],
                     weights=np.ones(len(data4[i,j,:])) / len(data4[i,j,:]), label='ARM 14:30L')
            ax[i,j].hist(data5[i,j,:].flatten(), bins=bins_in, histtype='step', stacked=False, color=colours[4],
                     weights=np.ones(len(data5[i,j,:])) / len(data5[i,j,:]), label='ARM 16:30L')
            ax[i,j].set_xlabel(f"{fields_latex_in[i]}", fontsize=16)
        ax[i,0].set_ylabel("Percentage of Occurrences", fontsize=16)

    # bottom_set, top_set = plt.ylim()
    # print('y_min = ', bottom_set, 'y_max = ', top_set)
    ax[4,0].legend(fontsize=12, loc='best')
    #plt.vlines(0, ymin=0, ymax=((1e9)), linestyles='dashed', colors='black', linewidths=0.5)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
    plt.savefig(plotdir_in + f'hist_of_{what_plotting}_values_{delta}.pdf',
                bbox_inches='tight')
    plt.clf()

    print(f'plotted {what_plotting}')








for i in range(len(data_field_list)):
    cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_cl_list[i])

    data_field = data_field_list[i][f'{field}'].data[...]
    print(np.shape(data_field[0, ...]))

    data_field_cloud = ma.masked_array(data_field[-1,...], mask=cloud_only_mask) #only look at one time stamp
    data_field_env = ma.masked_array(data_field[-1,...], mask=env_only_mask) #only look at one time stamp