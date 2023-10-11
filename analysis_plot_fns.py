import numpy as np
import matplotlib.pyplot as plt
import dynamic_functions as dyn
import mask_cloud_vs_env as clo
import numpy.ma as ma
import dynamic_functions as dyn
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import xarray as xr
import os
import matplotlib
from matplotlib.ticker import FormatStrFormatter

def negs_in_field(plotdir, field, data_field_list, data_cl_list):
    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']


    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_cl_list[i])

        data_field = data_field_list[i][f'{field}'].data[...]
        print(np.shape(data_field[0,...]))

        data_field_cloud = np.mean(ma.masked_array(data_field, mask=cloud_only_mask), axis=0)
        data_field_env = np.mean(ma.masked_array(data_field, mask=env_only_mask), axis=0)

        print(np.shape(data_field_env))

        counter_env = np.zeros(len(data_field_env[0, 0, :]))
        counter_cloud = np.zeros(len(data_field_cloud[0,0,:]))

        for j in range(len(data_field_cloud[0,0,:])):
            counter_cloud[j] = np.count_nonzero(data_field_cloud[:,:,j] < 0)
            counter_env[j] = np.count_nonzero(data_field_env[:, :, j] < 0)

        plt.figure(figsize=(7, 6))
        plt.hist([counter_env, counter_cloud], bins=12, histtype='bar', stacked=True, label=["environment", "in-cloud"])
        plt.legend()

        og_xtic = plt.xticks()
        plt.xticks(og_xtic[0],
                   np.round(np.linspace((0) * (20 / 480), (151) * (20 / 480), len(og_xtic[0])), 1))

        plt.xlabel("$z/z_{ML}$", fontsize=16)
        plt.ylabel("number of negative values", fontsize=16)
        plt.savefig(plotdir + f'neg_{field}_vs_z_{deltas[i]}.pdf', bbox_inches='tight')
        plt.clf()

        print(f'plotted neg vs z for {field}')

    plt.close('all')


def plot_hist(plotdir_in, field_in, time_set_in, delta, data1, data2, data3, data_names, bins_in):

    if field_in == 'Cs_field':
        name = field_in
        scalar = '$C_{s}$'
    elif field_in == 'Cth_field':
        name = field_in
        scalar = '$C_{\\theta}$'
    elif field_in == 'Cqt_field':
        name = field_in
        scalar = '$C_{qt}$'

    if field_in == 'Cs_sq_field':
        name = field_in
        scalar = '$C_{s}^2$'
    elif field_in == 'Cth_sq_field':
        name = field_in
        scalar = '$C_{\\theta}^2$'
    elif field_in == 'Cqt_sq_field':
        name = field_in
        scalar = '$C_{qt}^2$'

    elif field_in == 'f(LM_field_in_on_w)_r':
        scalar = '$L_{ij}M_{ij}$'
        name = 'LM'
    elif field_in == 'f(HR_th_field_in_on_w)_r':
        scalar = '$H_{j}R_{j \\theta}$'
        name = 'HR_th'
    elif field_in == 'f(HR_q_total_field_in_on_w)_r':
        scalar = '$H_{j}R_{j qt}$'
        name = 'HR_q_total'

    plt.figure(figsize=(5, 6))
    plt.hist(data1.flatten(), \
             bins=bins_in, histtype='step', stacked=False, label=data_names[0]) #, \
             #linewidth=1, linestyle='dotted')
    plt.hist(data2.flatten(), \
             bins=bins_in, histtype='step', stacked=False, label=data_names[1]) #, \
             #linewidth=1, linestyle='dotted')
    plt.hist(data3.flatten(), \
             bins=bins_in, histtype='step', stacked=False, label=data_names[2]) #, \
             #linewidth=1, linestyle='dotted')
    bottom_set, top_set = plt.ylim()
    print('y_min = ', bottom_set, 'y_max = ', top_set)
    plt.legend(fontsize=12, loc='best')
    plt.vlines(0, ymin=0, ymax=((1e9)), linestyles='dashed', colors='black', linewidths=0.5)
    plt.yscale('log', nonposy='clip')
    plt.ylim(bottom_set, (top_set+(top_set/10)))
    plt.xlabel(f"{scalar} at time {time_set_in}", fontsize=16)
    plt.ylabel("number of value occurrences", fontsize=16)
    plt.savefig(plotdir_in + f'hist_of_{name}_values_{delta}_time_{time_set_in}_vars_{data_names[0]}.pdf',
                bbox_inches='tight')
    plt.clf()

    print(f'plotted for time {time_set_in} {field_in} {delta}')



def plot_C_Delta_hist_comp(dir_in, field, condits = None, deltas=None):

    data_dir = dir_in + 'data/'
    if deltas == None:
        deltas = ['2', '4', '8', '16', '32', '64']
    if condits == None:
        condits = ['domain', 'ML', 'clear_sky', 'cloud', 'cloud_up', 'cloud_core']

    if field == 'Cs_sq':
        scalar = '$C_{s}^2$'
    elif field == 'Cth_sq':
        scalar = '$C_{\\theta}^2$'
    elif field == 'Cqt_sq':
        scalar = '$C_{qt}^2$'
    else:
        print('field must be Cs_sq, Cth_sq, or Cqt_sq')

    for j in range(len(condits)):

        plt.figure(figsize=(10, 6))

        for i in range(len(deltas)):
            C = np.load(data_dir + f'{deltas[i]}D_{field}_field_flat_{condits[j]}.npy')
            plt.hist(C, bins=50, histtype='step', stacked=False, label=deltas[i]+'$\\Delta$')
        bottom_set, top_set = plt.ylim()
        print('y_min = ', bottom_set, 'y_max = ', top_set)
        plt.legend(fontsize=12, loc='best')
        plt.vlines(0, ymin=0, ymax=((1e9)), linestyles='dashed', colors='black', linewidths=0.5)
        plt.yscale('log', nonposy='clip')
        plt.ylim(bottom_set, (top_set + (top_set / 10)))
        plt.xlabel(f"{scalar}", fontsize=16)
        plt.ylabel("number of value occurrences", fontsize=16)
        plt.savefig(data_dir + f'delta_hist_of_{field}_{condits[j]}_values.pdf', bbox_inches='tight')
        plt.clf()

        print(f'plotted for {field} {condits[j]}')


def C_values_dist(plotdir, field, data_field_list, data_contour, set_bins, deltas=None, times='av', grid='p', other_vars=None,
                  other_var_thres=None, less_greater_in=['less'], and_or_in = ['and'], cloud_liquid_threshold_in=10**(-5),
                  res_counter_in=None, return_all_in = False, grid_in='p', **kwargs):

    data_dir = plotdir + f'data/'
    os.makedirs(data_dir, exist_ok=True)

    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(deltas)):
        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_contour +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                                cloud_liquid_threshold=cloud_liquid_threshold_in)

        data_field = data_field_list+f'{i}_0_running_mean_filter_rm00.nc'
        data_field_in = xr.open_dataset(data_field)

        if other_vars!=None:
            if return_all_in == False:
                if len(other_vars) == 1:
                    combo2_out_mask = clo.cloudy_and_or(data_contour +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask = clo.cloudy_and_or(data_contour  +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
            else:
                if len(other_vars) == 1:
                    combo2_out_mask, cloud_mask, var_mask = clo.cloudy_and_or(data_contour  +
                                                                f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask, cloud_mask, var_mask, extra_var_mask = \
                        clo.cloudy_and_or(data_contour + f'{i}_gaussian_filter_ga00_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)

        if field == 'Cs_field' or field == 'Cs_sq_field':
            print('length of time array for LM is ', len(data_field_in[f'f(LM_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(LM_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(MM_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cs_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None

        elif field == 'Cth_field' or field == 'Cth_sq_field':
            print('length of time array for HR_th is ', len(data_field_in[f'f(HR_th_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(RR_th_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cth_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None

        elif field == 'Cqt_field' or field == 'Cqt_sq_field':
            print('length of time array for HR_qt is ',
                  len(data_field_in[f'f(HR_q_total_f_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(HR_q_total_f_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(RR_q_total_f_field_on_{grid})_r'].data[...]

            data_field_sq = 0.5 * num_field / den_field
            data_field_C = dyn.get_Cs(data_field_sq)
            if field == 'Cqt_field':
                data_field = data_field_C
            else:
                data_field = data_field_sq
            data_field_sq = None
            data_field_C = None



        else:
            data_field = data_field_in[f'{field}'].data[...]
            print(np.shape(data_field[...]))

        data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
        data_field_env = ma.masked_array(data_field, mask=env_only_mask)

        if len(other_vars) == 2:
            data_field_cloud_up = ma.masked_array(data_field, mask=combo2_out_mask)
            data_field_cloud_core = ma.masked_array(data_field, mask=combo3_out_mask)

        print(np.shape(data_field_env))


        if times != 'av':
            for time_set in times:
                plot_hist(plotdir, field, time_set, deltas[i],
                         data_field_env[time_set,...,0:24],
                          data_field_env[time_set,...,24:151],
                          data_field_cloud[time_set,...],
                          data_names=["ML", "CL: cloud-free", "CL: cloudy"],
                          bins_in=set_bins)


                if len(other_vars) == 2:
                    plot_hist(plotdir, field, time_set, deltas[i],
                              data_field_cloud[time_set,...],
                              data_field_cloud_up[time_set,...],
                              data_field_cloud_core[time_set,...],
                              data_names=["Cloud", "Cloud updraft", "Cloud core"],
                              bins_in=set_bins)


        else:

            plot_hist(plotdir, field, 'av', deltas[i],
                      data_field_env[..., 0:24],
                      data_field_env[..., 24:151],
                      data_field_cloud[...],
                      data_names=["ML", "CL: cloud-free", "CL: cloudy"],
                      bins_in=set_bins)

            if len(other_vars) == 2:
                plot_hist(plotdir, field, 'av', deltas[i],
                          data_field_cloud[...],
                          data_field_cloud_up[...],
                          data_field_cloud_core[...],
                          data_names = ["Cloud", "Cloud updraft", "Cloud core"],
                          bins_in=set_bins)

        np.save( data_dir + f'{deltas[i]}_{field}_flat_domain.npy', data_field.flatten() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_ML.npy', data_field_env[..., 0:24].compressed() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_clear_sky.npy', data_field_env[..., 24:151].compressed() )
        np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud.npy', data_field_cloud.compressed() )

        if len(other_vars) == 2:
            np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud_up', data_field_cloud_up.compressed() )
            np.save( data_dir + f'{deltas[i]}_{field}_flat_cloud_core', data_field_cloud_core.compressed() )

    plt.close('all')


def plotfield(plot_dir, field, x_or_y, axis_set, data_field_in, set_percentile, contour_field_in, t_av_or_not,
              start_end, z_top_in, z_tix_in, z_labels_in, set_percentile_C_sq=None, deltas=None,
              set_cb=[[None, None], [None, None]], delta_grid=25):

    print('starting to plot field: ', field)

    myvmin_C = set_cb[0][0]
    myvmax_C = set_cb[0][1]

    myvmin_C_sq = set_cb[1][0]
    myvmax_C_sq = set_cb[1][1]

    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    start = start_end[0]
    start_grid = int(start/(0.001*delta_grid)) # going from km to grid spacing co-ords (20m or 25m grid)
    end = start_end[1]
    end_grid = int(end/(0.001*delta_grid)) # going from km to grid spacing co-ords (20m or 25m grid)


    if field == 'Cs_field':
        field_name = '$C_s$'
        field_name_sq = '$C_s^2$'
    if field == 'Cth_field':
        field_name = '$C_{\\theta}$'
        field_name_sq = '$C_{\\theta}^2$'
    if field == 'Cqt_field':
        field_name = '$C_{qt}$'
        field_name_sq = '$C_{qt}^2$'

    if field == 'LM_field':
        field_name = '$LM$'
    if field == 'HR_th_field':
        field_name = '$HR_{\\theta}$'
    if field == 'HR_qt_field':
        field_name = '$HR_{qt}$'
    if field == 'MM_field':
        field_name = '$MM$'
    if field == 'RR_th_field':
        field_name = '$RR_{\\theta}$'
    if field == 'RR_qt_field':
        field_name = '$RR_{qt}$'


    for i in range(len(deltas)):
        if deltas[i] == '0_0':
            CL_itr = '0'
            beta_CL_itr = '0'
            delta_label = '2$\\Delta$'

        else:
            print('need to code the delta for ', deltas[i])

        for t_set in t_av_or_not:
            if field == 'Cs_field':
                print('opening dataset ', data_field_in, f'{deltas[i]}_running_mean_filter_rm00.nc')
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')
                print('successfully opened dataset')

                print('length of time array for LM is ', len(data_set['f(LM_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    else:
                        print("x_or_y must be set to either 'x' or 'y', not ", x_or_y)
                else:
                    if x_or_y == 'x':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, :, axis_set, ...]
                    else:
                        print("x_or_y must be set to either 'x' or 'y', not ", x_or_y)

                data_field_sq = 0.5 * LM_field / MM_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated Cs^2')

            elif field == 'Cth_field':
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

                print('length of time array for HR_th is ', len(data_set['f(HR_th_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        HR_field = np.mean(data_set['f(HR_th_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_th_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        HR_field = np.mean(data_set['f(HR_th_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_th_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        HR_field = data_set['f(HR_th_field_on_p)_r'].data[t_set, axis_set, ...]
                        RR_field = data_set['f(RR_th_field_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        HR_field = data_set['f(HR_th_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        RR_field = data_set['f(RR_th_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * HR_field / RR_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated C_th^2')

            elif field == 'Cqt_field':
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

                print('length of time array for HR_qt is ',
                      len(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        HR_field = np.mean(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_f_field_on_p)_r'].data[:, axis_set, ...], axis=0)

                    elif x_or_y == 'y':
                        HR_field = np.mean(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_f_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        HR_field = data_set['f(HR_q_total_f_field_on_p)_r'].data[t_set, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_f_field_on_p)_r'].data[t_set, axis_set, ...]

                    elif x_or_y == 'y':
                        HR_field = data_set['f(HR_q_total_f_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_f_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * HR_field / RR_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated C_qt^2')

            else:
                print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        data_field = np.mean(data_set[f'f({field}_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        data_field = np.mean(data_set[f'f({field}_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        data_field = data_set[f'f({field}_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        data_field = data_set[f'f({field}_on_p)_r'].data[t_set, :, axis_set, ...]

            data_set.close()

            print('opening the contour dataset')

            contour_set = xr.open_dataset(data_field_in +
                                          f'_ga0{CL_itr}_gaussian_filter_ga0{beta_CL_itr}_running_mean_filter_rm00.nc')

            print('successfully opened contour set')

            print('length of time array for cloud field is ',
                  len(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                if x_or_y == 'x':
                    cloud_field = np.mean(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                                          axis=0)
                    w_field = np.mean(contour_set['f(f(w_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)
                    w2_field = np.mean(contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)
                    th_v_field = np.mean(contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)

                elif x_or_y == 'y':
                    cloud_field = np.mean(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    w_field = np.mean(contour_set['f(f(w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    w2_field = np.mean(contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    th_v_field = np.mean(contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)

                mytime = 't_av'
            else:
                if x_or_y == 'x':
                    cloud_field = contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    w_field = contour_set['f(f(w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    w2_field = contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    th_v_field = contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]

                elif x_or_y == 'y':
                    cloud_field = contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    w_field = contour_set['f(f(w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    w2_field = contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    th_v_field = contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]

                mytime = f't{t_set}'

            contour_set.close()

            print('beginning plots')

            fig1, ax1 = plt.subplots(figsize=(20, 5))
            plt.title(f'{field_name}' + ' with $\\widehat{\\bar{\\Delta}} = $' + f'{delta_label}', fontsize=16)

            mycmap = plt.get_cmap('YlOrRd').copy()
            mycmap.set_extremes(under='white', over='maroon')

            if myvmin_C != None:
                 myvmin = myvmin_C
                 myvmax = myvmax_C
                 mylevels = np.linspace(myvmin, myvmax, 8)
                 cf = plt.contourf(np.transpose(data_field), cmap=mycmap, levels=mylevels,
                                   extend='both')
            else:
                if set_percentile != None:
                    myvmin = np.percentile(data_field[start_grid:end_grid, 5:z_top_in], set_percentile[0])
                    myvmax = np.percentile(data_field[start_grid:end_grid, 5:z_top_in], set_percentile[1])
                    mylevels = np.linspace(myvmin, myvmax, 8)
                    cf = plt.contourf(np.transpose(data_field), cmap=mycmap, levels=mylevels,
                                  extend='both')

                if set_percentile == None:
                    cf = plt.contourf(np.transpose(data_field), cmap=mycmap, extend='both')

            cb = plt.colorbar(cf, format='%.2f')
            cb.set_label(f'{field_name}', size=16)

            cl_c = plt.contour(np.transpose(cloud_field), colors='black', linewidths=2,
                               levels=[1e-7])
            th_v_c = plt.contour(np.transpose(th_v_field[start_grid:end_grid, :]), colors='black', linestyles='dashed',
                        linewidths=1)  # , levels=[0.1, 1, 2])
            ax1.clabel(th_v_c, inline=True, fontsize=10)
            w_c = plt.contour(np.transpose(w_field), colors='darkslategrey', linewidths=1,
                        levels=[0.1, 0.5]) #start_grid:end_grid
            ax1.clabel(w_c, inline=True, fontsize=8)
            # plt.contour(np.transpose(w2_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=1, levels=[0.1])
            plt.xlabel(f'x (km) (cross section with {x_or_y} = {round(axis_set*delta_grid/1000, 1)}km) (km)', fontsize=16)

            plt.ylabel("z (km)", fontsize=16)
            plt.xlim(start_grid, end_grid)
            og_xtic = plt.xticks()
            plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0])), 1))

            # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plt.ylim(0, z_top_in)
            og_ytic = plt.yticks()
            plt.yticks(z_tix_in, z_labels_in)  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

            plt.savefig(plot_dir + f'{field}_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.pdf',
                        bbox_inches='tight')
            plt.clf()

            if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
                fig2, ax2 = plt.subplots(figsize=(20, 5))
                plt.title(f'{field_name}' + ' with $\\widehat{\\bar{\\Delta}} = $' + f'{delta_label}', fontsize=16)

                if myvmin_C_sq != None:

                    if set_percentile_C_sq[0] == 'min':
                        myvmin_temp = np.min(data_field_sq[start_grid:end_grid, 5:z_top_in])
                        myvmin = myvmin_temp + abs(0.6*myvmin_temp)
                        myvmax = np.percentile(data_field_sq[start_grid:end_grid, 5:z_top_in], set_percentile_C_sq[1])

                        mylevels = np.linspace(myvmin, myvmax, 8)
                        cf = plt.contourf(np.transpose(data_field_sq), cmap=cm.bwr,
                                  norm=TwoSlopeNorm(vmin=myvmin, vcenter=0, vmax=myvmax),
                                  levels=mylevels, extend='both')
                    else:
                        myvmin = myvmin_C_sq
                        myvmax = myvmax_C_sq

                        mylevels = np.linspace(myvmin, myvmax, 8)
                        cf = plt.contourf(np.transpose(data_field_sq), cmap=cm.bwr,
                                          norm=TwoSlopeNorm(vmin=myvmin, vcenter=0, vmax=myvmax),
                                          levels=mylevels, extend='both')

                else:
                   cf = plt.contourf(np.transpose(data_field_sq), cmap=cm.bwr, vcenter=0, extend='both')

                cb = plt.colorbar(cf, format='%.2f')
                # cb.set_under('k')
                cb.set_label(f'{field_name_sq}', size=16)

                cl_c = plt.contour(np.transpose(cloud_field), colors='black', linewidths=2,
                                   levels=[1e-7])
                th_v_c = plt.contour(np.transpose(th_v_field[start_grid:end_grid, :]), colors='black', linestyles='dashed',
                                     linewidths=1)  # , levels=[0.1, 1, 2])
                ax2.clabel(th_v_c, inline=True, fontsize=10)
                w_c = plt.contour(np.transpose(w_field), colors='darkslategrey',
                                  linewidths=1, levels=[0.1, 0.5])
                ax2.clabel(w_c, inline=True, fontsize=8)
                # plt.contour(np.transpose(w2_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=1, levels=[0.1])
                plt.xlabel(f'x (km) (cross section with {x_or_y} = {round(axis_set*delta_grid/1000, 1)}km)', fontsize=16)

                plt.xlim(start_grid, end_grid)
                og_xtic = plt.xticks()
                plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0])), 1))

                # ax2.set_xticks(np.linspace(start, end, len(og_xtic[0])))
                # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # og_ytic = plt.yticks()
                plt.ylim(0, z_top_in)
                plt.yticks(z_tix_in, z_labels_in)#plt.yticks(np.linspace(0, 151, 7), np.linspace(0, 3, 7))
                plt.ylabel("z (km)", fontsize=16)
                plt.savefig(plot_dir + f'{field}_sq_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.pdf',
                            bbox_inches='tight')
                plt.clf()

            print(f'plotted fields for {field} {mytime}')

    plt.close('all')





def shiftedColorMap(cmap, vmax, vmin, start=0, midpoint='calc', stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    if midpoint == 'calc':
        midpoint = 1 - ( vmax / (vmax + abs(vmin)) )


    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap




def plot_C_contours(plot_dir, field, x_or_y, axis_set, data_field_in, set_percentile, var_field, var_path, t_av_or_not,
              start_end, z_top_in, z_tix_in, z_labels_in, C_perc_1st, C_perc_2nd, deltas=None,
                    set_cb=[None, None], delta_grid=25, set_percentile_C_sq = None):

    print('starting to plot field: ', field)

    myvmin_var = set_cb[0]
    myvmax_var = set_cb[1]

    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    start = start_end[0]
    start_grid = int(start/(0.001*delta_grid)) # going from km to grid spacing co-ords (20m or 25m grid)
    end = start_end[1]
    end_grid = int(end/(0.001*delta_grid)) # going from km to grid spacing co-ords (20m or 25m grid)


    if field == 'Cs_field':
        field_name = '$C_s$'
        field_name_sq = '$C_s^2$'
    if field == 'Cth_field':
        field_name = '$C_{\\theta}$'
        field_name_sq = '$C_{\\theta}^2$'
    if field == 'Cqt_field':
        field_name = '$C_{qt}$'
        field_name_sq = '$C_{qt}^2$'

    if field == 'LM_field':
        field_name = '$LM$'
    if field == 'HR_th_field':
        field_name = '$HR_{\\theta}$'
    if field == 'HR_qt_field':
        field_name = '$HR_{qt}$'
    if field == 'MM_field':
        field_name = '$MM$'
    if field == 'RR_th_field':
        field_name = '$RR_{\\theta}$'
    if field == 'RR_qt_field':
        field_name = '$RR_{qt}$'


    if var_field == 'w':
        var_name = "$w'$"
        var_units = '$m s^{-1}$'
    if var_field == 'w_th_v':
        var_name = "$w' \\theta_v'$"
        var_units = '$K m s^{-1}$'
    if var_field == 'TKE':
        var_name = "TKE"
        var_units = '$m^2 s^{-2}$'



    for i in range(len(deltas)):
        if deltas[i] == '0_0':
            CL_itr = '0'
            beta_CL_itr = '0'
            delta_label = '2$\\Delta$'

        else:
            print('need to code the delta for ', deltas[i])

        for t_set in t_av_or_not:
            if field == 'Cs_field':
                print('opening dataset ', data_field_in, f'{deltas[i]}_running_mean_filter_rm00.nc')
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')
                print('successfully opened dataset')

                print('length of time array for LM is ', len(data_set['f(LM_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    else:
                        print("x_or_y must be set to either 'x' or 'y', not ", x_or_y)
                else:
                    if x_or_y == 'x':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, :, axis_set, ...]
                    else:
                        print("x_or_y must be set to either 'x' or 'y', not ", x_or_y)

                data_field_sq = 0.5 * LM_field / MM_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated Cs^2')

            elif field == 'Cth_field':
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

                print('length of time array for HR_th is ', len(data_set['f(HR_th_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        HR_field = np.mean(data_set['f(HR_th_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_th_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        HR_field = np.mean(data_set['f(HR_th_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_th_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        HR_field = data_set['f(HR_th_field_on_p)_r'].data[t_set, axis_set, ...]
                        RR_field = data_set['f(RR_th_field_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        HR_field = data_set['f(HR_th_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        RR_field = data_set['f(RR_th_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * HR_field / RR_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated C_th^2')

            elif field == 'Cqt_field':
                data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

                print('length of time array for HR_qt is ',
                      len(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        HR_field = np.mean(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_f_field_on_p)_r'].data[:, axis_set, ...], axis=0)

                    elif x_or_y == 'y':
                        HR_field = np.mean(data_set['f(HR_q_total_f_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_f_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        HR_field = data_set['f(HR_q_total_f_field_on_p)_r'].data[t_set, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_f_field_on_p)_r'].data[t_set, axis_set, ...]

                    elif x_or_y == 'y':
                        HR_field = data_set['f(HR_q_total_f_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_f_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * HR_field / RR_field
                data_field = dyn.get_Cs(data_field_sq)

                print('successfully calculated C_qt^2')

            else:
                print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        data_field = np.mean(data_set[f'f({field}_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        data_field = np.mean(data_set[f'f({field}_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        data_field = data_set[f'f({field}_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        data_field = data_set[f'f({field}_on_p)_r'].data[t_set, :, axis_set, ...]

            data_set.close()

            print('opening the contour dataset')

            var_field_data = xr.open_dataset(var_path +
                                          f'{CL_itr}_gaussian_filter_ga0{beta_CL_itr}_running_mean_filter_rm00.nc')

            print('successfully opened contour set')

            print('length of time array for cloud field is ',
                  len(var_field_data['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                if x_or_y == 'x':
                    cloud_field = np.mean(var_field_data['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                                          axis=0)
                    if var_field == 'w':
                        var_field_plot = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif var_field == 'TKE':
                        u_mean = np.zeros(len(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        v_mean = np.zeros(len(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        #w_mean = np.zeros(len(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(v_mean)):
                            u_mean[nz] = np.mean(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[..., nz])
                            v_mean[nz] = np.mean(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[..., nz])
                            #w_mean[nz] = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[..., nz])
                        u_prime_field = var_field_data['f(f(u_on_p)_r_on_p)_r'].data[:, axis_set, ...] - u_mean
                        v_prime_field = var_field_data['f(f(v_on_p)_r_on_p)_r'].data[:, axis_set, ...] - v_mean
                        #w_prime_field = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[:, axis_set, ...] - w_mean
                        ww_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, axis_set, ...]
                        var_field_plot = np.mean( 0.5*(u_prime_field*u_prime_field + \
                                                       v_prime_field*v_prime_field + \
                                                       ww_field), axis=0)
                        u_prime_field = None
                        v_prime_field = None
                        ww_field = None
                    elif var_field == 'w_th_v':
                        th_v_mean = np.zeros(len(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[0,0,0,:]))
                        for nz in range(len(th_v_mean)):
                            th_v_mean[nz] = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[..., nz])
                        th_v_f = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)

                        # var_field_plot = np.mean(var_field_data['f(f(w_on_p.th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                        #                          axis=0)

                        var_field_plot = th_v_f - th_v_mean
                        th_v_mean = None
                        th_v_f = None

                    #w2_field = np.mean(var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)
                    th_v_field = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)

                elif x_or_y == 'y':
                    cloud_field = np.mean(var_field_data['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    if var_field == 'w':
                        var_field_plot = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    elif var_field == 'TKE':
                        u_mean = np.zeros(len(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        v_mean = np.zeros(len(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        #w_mean = np.zeros(len(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(v_mean)):
                            u_mean[nz] = np.mean(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[..., nz])
                            v_mean[nz] = np.mean(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[..., nz])
                            #w_mean[nz] = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[..., nz])
                        u_prime_field = var_field_data['f(f(u_on_p)_r_on_p)_r'].data[:, :, axis_set, ...] - u_mean
                        v_prime_field = var_field_data['f(f(v_on_p)_r_on_p)_r'].data[:, :, axis_set, ...] - v_mean
                        #w_prime_field = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...] - w_mean
                        ww_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...]
                        var_field_plot = np.mean(0.5 * (u_prime_field * u_prime_field + \
                                                        v_prime_field * v_prime_field + \
                                                        ww_field), axis=0)
                        u_prime_field = None
                        v_prime_field = None
                        ww_field = None
                    elif var_field == 'w_th_v':
                        th_v_mean = np.zeros(len(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(th_v_mean)):
                            th_v_mean[nz] = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[..., nz])
                        th_v_f = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)

                        # var_field_plot = np.mean(var_field_data['f(f(w_on_p.th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                        #                          axis=0)

                        var_field_plot = th_v_f - th_v_mean
                        th_v_mean = None
                        th_v_f = None

                    #w2_field = np.mean(var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                    th_v_field = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[:, :, axis_set, ...], axis=0)

                mytime = 't_av'
            else:
                if x_or_y == 'x':
                    cloud_field = var_field_data['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    if var_field == 'w':
                        var_field_plot = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    elif var_field == 'TKE':
                        u_mean = np.zeros(len(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        v_mean = np.zeros(len(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        #w_mean = np.zeros(len(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(v_mean)):
                            u_mean[nz] = np.mean(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[..., nz])
                            v_mean[nz] = np.mean(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[..., nz])
                            #w_mean[nz] = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[..., nz])
                        u_prime_field = var_field_data['f(f(u_on_p)_r_on_p)_r'].data[t_set, axis_set, ...] - u_mean
                        v_prime_field = var_field_data['f(f(v_on_p)_r_on_p)_r'].data[t_set, axis_set, ...] - v_mean
                        #w_prime_field = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...] - w_mean
                        ww_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                        var_field_plot = 0.5 * (u_prime_field * u_prime_field + \
                                                        v_prime_field * v_prime_field + \
                                                        ww_field)
                        u_prime_field = None
                        v_prime_field = None
                        ww_field = None
                    elif var_field == 'w_th_v':
                        th_v_mean = np.zeros(len(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(th_v_mean)):
                            th_v_mean[nz] = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set,..., nz])
                        th_v_f = var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]

                        # var_field_plot = np.mean(var_field_data['f(f(w_on_p.th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                        #                          axis=0)

                        var_field_plot = th_v_f - th_v_mean
                        th_v_mean = None
                        th_v_f = None


                    #w2_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]
                    th_v_field = var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, axis_set, ...]

                elif x_or_y == 'y':
                    cloud_field = var_field_data['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    if var_field == 'w':
                        var_field_plot = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    elif var_field == 'TKE':
                        u_mean = np.zeros(len(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        v_mean = np.zeros(len(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        #w_mean = np.zeros(len(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(v_mean)):
                            u_mean[nz] = np.mean(var_field_data['f(f(u_on_p)_r_on_p)_r'].data[..., nz])
                            v_mean[nz] = np.mean(var_field_data['f(f(v_on_p)_r_on_p)_r'].data[..., nz])
                            #w_mean[nz] = np.mean(var_field_data['f(f(w_on_p)_r_on_p)_r'].data[..., nz])
                        u_prime_field = var_field_data['f(f(u_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...] - u_mean
                        v_prime_field = var_field_data['f(f(v_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...] - v_mean
                        # w_prime_field = var_field_data['f(f(w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...] - w_mean
                        ww_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                        var_field_plot = 0.5 * (u_prime_field * u_prime_field + \
                                                        v_prime_field * v_prime_field + \
                                                        ww_field)
                        u_prime_field = None
                        v_prime_field = None
                        w_prime_field = None
                    elif var_field == 'w_th_v':
                        th_v_mean = np.zeros(len(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[0, 0, 0, :]))
                        for nz in range(len(th_v_mean)):
                            th_v_mean[nz] = np.mean(var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, ..., nz])
                        th_v_f = var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]

                        # var_field_plot = np.mean(var_field_data['f(f(w_on_p.th_v_on_p)_r_on_p)_r'].data[:, axis_set, ...],
                        #                          axis=0)

                        var_field_plot = th_v_f - th_v_mean
                        th_v_mean = None
                        th_v_f = None

                    #w2_field = var_field_data['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]
                    th_v_field = var_field_data['f(f(th_v_on_p)_r_on_p)_r'].data[t_set, :, axis_set, ...]

                mytime = f't{t_set}'

            var_field_data.close()

            print('beginning plots')

            fig1, ax1 = plt.subplots(figsize=(20, 5))
            plt.title(f'{var_name} and {field_name} contours' + ' with $\\widehat{\\bar{\\Delta}} = $' + f'{delta_label}', fontsize=16)

            if myvmin_var != None:
                 myvmin = myvmin_var
                 myvmax = myvmax_var

                 # orig_cmap = matplotlib.cm.coolwarm
                 # shifted_cmap = shiftedColorMap(orig_cmap, myvmin, myvmax)

                 print('vmax and vmin values are: ', myvmax, myvmin)
                 mylevels = np.linspace(myvmin, myvmax, 9)
                 if var_field == 'w' or var_field == 'w_th_v':
                     cf = plt.contourf(np.transpose(var_field_plot), cmap=cm.coolwarm,
                                       norm=TwoSlopeNorm(vmin=myvmin, vcenter=0, vmax=myvmax),
                                       levels=mylevels, extend='both')
                 else:
                     cf = plt.contourf(np.transpose(var_field_plot), cmap=cm.coolwarm,
                                       levels=mylevels, extend='both')
            else:
                if set_percentile != None:
                    myvmin = np.percentile(var_field_plot[start_grid:end_grid, 5:z_top_in], set_percentile[0])
                    myvmax = np.percentile(var_field_plot[start_grid:end_grid, 5:z_top_in], set_percentile[1])

                    # orig_cmap = matplotlib.cm.coolwarm
                    # shifted_cmap = shiftedColorMap(orig_cmap, myvmin, myvmax)

                    mylevels = np.linspace(myvmin, myvmax, 9)
                    if var_field == 'w' or var_field == 'w_th_v':
                        cf = plt.contourf(np.transpose(var_field_plot), cmap=cm.coolwarm,
                                      norm=TwoSlopeNorm(vmin=myvmin, vcenter=0, vmax=myvmax),
                                      levels=mylevels, extend='both')
                    else:
                        cf = plt.contourf(np.transpose(var_field_plot), cmap=cm.coolwarm,
                                          levels=mylevels, extend='both')

                if set_percentile == None:
                    cf = plt.contourf(np.transpose(var_field_plot), cmap=cm.coolwarm, extend='both')

            cb = plt.colorbar(cf, format='%.2f')
            cb.set_label(f'{var_name} ({var_units})', size=16)

            cl_c = plt.contour(np.transpose(cloud_field), colors='black', linewidths=2, levels=[1e-7])
            th_v_c = plt.contour(np.transpose(th_v_field[start_grid:end_grid, :]), colors='black', linestyles='dashed',
                        linewidths=1)  # , levels=[0.1, 1, 2])
            ax1.clabel(th_v_c, inline=True, fontsize=10)

            C_1st = np.percentile(data_field[start_grid:end_grid, 5:z_top_in], C_perc_1st)
            my_C_levels = [C_1st]
            if C_perc_2nd != None:
                C_2nd = np.percentile(data_field[start_grid:end_grid, 5:z_top_in], C_perc_2nd)
                my_C_levels = [C_1st, C_2nd]
            C_contour = plt.contour(np.transpose(data_field), colors='darkslategrey', linewidths=2,
                        levels=my_C_levels) #darkslategrey
            ax1.clabel(C_contour, inline=True, fontsize=10, fmt='%1.2f')
            # plt.contour(np.transpose(w2_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=2, levels=[0.1])
            plt.xlabel(f'x (km) (cross section with {x_or_y} = {round(axis_set*delta_grid/1000, 1)}km) (km)', fontsize=16)

            plt.ylabel("z (km)", fontsize=16)
            plt.xlim(start_grid, end_grid)
            og_xtic = plt.xticks()
            plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0])), 1))

            # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plt.ylim(0, z_top_in)
            og_ytic = plt.yticks()
            plt.yticks(z_tix_in, z_labels_in)  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

            plt.savefig(plot_dir + f'{var_field}_{field}_{C_perc_1st}_{C_perc_2nd}_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.pdf',
                        bbox_inches='tight')
            plt.clf()


            print(f'plotted fields for {field} {mytime}')

    plt.close('all')




def get_conditional_profiles(dataset_in, contour_field_in, field, deltas,
                      cloud_thres, other_vars, other_var_thres,
                             less_greater_in, and_or_in, grid, beta):
    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    if field == 'Cs_field':
        field_name = '$C_s$'
        save_name = 'Cs'
    elif field == 'Cs_sq_field':
        field_name_sq = '$C_s^2$'
        save_name = 'Cs_sq'
    elif field == 'Cth_field':
        field_name = '$C_{\\theta}$'
        save_name = 'Cth'
    elif field == 'Cth_sq_field':
        field_name_sq = '$C_{\\theta}^2$'
        save_name = 'Cth_sq'
    elif field == 'Cqt_field':
        field_name = '$C_{qt}$'
        save_name = 'Cqt'
    elif field == 'Cqt_sq_field':
        field_name_sq = '$C_{qt}^2$'
        save_name = 'Cqt_sq'


    elif field == 'LM_field' or field == f'f(LM_field_on_{grid})_r':
        field_name = '$LM$'
        save_name = 'LM'
    elif field == 'HR_th_field' or field == f'f(HR_th_field_on_{grid})_r':
        field_name = '$HR_{\\theta}$'
        save_name = 'HR_th'
    elif field == 'HR_q_total_field' or field == 'HR_q_total_f_field' or field == f'f(HR_q_total_f_field_on_{grid})_r':
        field_name = '$HR_{qt}$'
        save_name = 'HR_qt'

    elif field == 'MM_field' or field == f'f(MM_field_on_{grid})_r':
        field_name = '$MM$'
        save_name = 'MM'
    elif field == 'RR_th_field' or field == f'f(RR_th_field_on_{grid})_r':
        field_name = '$RR_{\\theta}$'
        save_name = 'RR_th'
    elif field == 'RR_q_total_field'  or field == 'RR_q_total_f_field' or field == f'f(RR_q_total_f_field_on_{grid})_r':
        field_name = '$RR_{qt}$'
        save_name = 'RR_qt'
    else:
        print('field not found')


    cloud_only_mask, env_only_mask = \
        clo.cloud_vs_env_masks(contour_field_in, \
                               cloud_liquid_threshold=cloud_thres)

    if other_vars != None:
        if len(other_vars) == 1:
            combo2_out_mask = clo.cloudy_and_or(contour_field_in,
                                                other_var=other_vars, var_thres=other_var_thres,
                                                less_greater=less_greater_in, and_or=and_or_in,
                                                cloud_liquid_threshold=cloud_thres, grid=grid)
        else:
            combo2_out_mask, combo3_out_mask = clo.cloudy_and_or(contour_field_in,
                                                                 other_var=other_vars, var_thres=other_var_thres,
                                                                 less_greater=less_greater_in, and_or=and_or_in,
                                                                 cloud_liquid_threshold=cloud_thres, grid=grid)


    data_set = xr.open_dataset(dataset_in)

    time_data = data_set['time']
    times = time_data.data
    nt = len(times)
    x_data = data_set['x_p']
    x_s = x_data.data
    y_data = data_set['y_p']
    y_s = y_data.data
    z_data = data_set['z']
    z_s = z_data.data
    zn_data = data_set['zn']
    zn_s = zn_data.data

    print('current field = ', field)

    if field == 'Cs_field' or field == 'Cs_sq_field':
        #print('length of time array for LM is ', len(data_set[f'f(LM_field_on_{grid})_r'].data[:, 0, 0, 0]))

        if f'f(LM_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(LM_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(MM_field_on_{grid})_r'].data[...]
        elif f'LM_field' in data_set:
            num_field = data_set[f'LM_field'].data[...]
            den_field = data_set[f'MM_field'].data[...]
        else:
            print('LM_field_on_{grid} not in file')
        C=True

    elif field == 'Cth_field' or field == 'Cth_sq_field':
        #print('length of time array for HR_th is ', len(data_set[f'f(HR_th_field_on_{grid})_r'].data[:, 0, 0, 0]))

        if f'f(HR_th_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_th_field_on_{grid})_r'].data[...]
        elif f'HR_th_field' in data_set:
            num_field = data_set[f'HR_th_field'].data[...]
            den_field = data_set[f'RR_th_field'].data[...]
        else:
            print('HR_field_on_{grid} not in file')

        C=True

    elif field == 'Cqt_field' or field == 'Cqt_sq_field':
        if beta==True:
            # print('length of time array for HR_qt is ', len(data_set[f'f(HR_q_total_f_field_on_{grid})_r'].data[:, 0, 0, 0]))
            # num_field = data_set[f'f(HR_q_total_f_field_on_{grid})_r'].data[...]
            # den_field = data_set[f'f(RR_q_total_f_field_on_{grid})_r'].data[...]

            if f'f(HR_q_total_f_field_on_{grid})_r' in data_set:
                num_field = data_set[f'f(HR_q_total_f_field_on_{grid})_r'].data[...]
                den_field = data_set[f'f(RR_q_total_f_field_on_{grid})_r'].data[...]
            elif f'HR_q_total_f_field' in data_set:
                num_field = data_set[f'HR_q_total_f_field'].data[...]
                den_field = data_set[f'RR_q_total_f_field'].data[...]
            else:
                print('HR_field_on_{grid} not in file')

        else:
            print('length of time array for HR_qt is ', len(data_set[f'f(HR_q_total_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_set[f'f(HR_q_total_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_q_total_field_on_{grid})_r'].data[...]
        C=True

    else:

        if beta==True:

            if f'f({field}_on_{grid})_r' in data_set:
                data_field = data_set[f'f({field}_on_{grid})_r'].data[...]
            elif f'{field}' in data_set:
                data_field = data_set[f'{field}'].data[...]

            #print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_{grid})_r'].data[:, 0, 0, 0]))

        else:
            #print('length of time array for HR_qt is ', len(data_set[f'f({field}_on_{grid})_r'].data[:, 0, 0, 0]))
            if f'f({field}_on_{grid})_r' in data_set:
                data_field = data_set[f'f({field}_on_{grid})_r'].data[...]
            elif f'{field}' in data_set:
                data_field = data_set[f'{field}'].data[...]

        C=False

    data_set.close()

    z_num = len(z_s)


    if C==True:
        num_field_cloud = ma.masked_array(num_field, mask=cloud_only_mask)
        num_field_env = ma.masked_array(num_field, mask=env_only_mask)
        den_field_cloud = ma.masked_array(den_field, mask=cloud_only_mask)
        den_field_env = ma.masked_array(den_field, mask=env_only_mask)

        if other_vars != None:
            num_field_combo2 = ma.masked_array(num_field, mask=combo2_out_mask)
            den_field_combo2 = ma.masked_array(den_field, mask=combo2_out_mask)
            if other_vars[0] == f'f(f(w_on_{grid})_r_on_{grid})_r' or other_vars[0] == f'f(w_on_{grid})_r':
                othervar1 = 'w'
            elif other_vars[0] == f'f(f(th_v_on_{grid})_r_on_{grid})_r' or other_vars[0] == f'f(th_v_on_{grid})_r':
                othervar1 = 'th_v'

            if len(other_vars) > 1:
                num_field_combo3 = ma.masked_array(num_field, mask=combo3_out_mask)
                den_field_combo3 = ma.masked_array(den_field, mask=combo3_out_mask)
                if other_vars[1] == f'f(f(w_on_{grid})_r_on_{grid})_r' or other_vars[1] == f'f(w_on_{grid})_r':
                    othervar2 = 'w'
                elif other_vars[1] == f'f(f(th_v_on_{grid})_r_on_{grid})_r' or other_vars[1] == f'f(th_v_on_{grid})_r':
                    othervar2 = 'th'

        num_prof = np.zeros(z_num)
        num_cloud_prof = np.zeros(z_num)
        num_env_prof = np.zeros(z_num)
        num_combo2_prof = np.zeros(z_num)
        num_combo3_prof = np.zeros(z_num)

        den_prof = np.zeros(z_num)
        den_cloud_prof = np.zeros(z_num)
        den_env_prof = np.zeros(z_num)
        den_combo2_prof = np.zeros(z_num)
        den_combo3_prof = np.zeros(z_num)

        for k in range(z_num):
            num_prof[k] = np.mean(num_field[..., k])
            num_cloud_prof[k] = np.mean(num_field_cloud[..., k])
            num_env_prof[k] = np.mean(num_field_env[..., k])

            den_prof[k] = np.mean(den_field[..., k])
            den_cloud_prof[k] = np.mean(den_field_cloud[..., k])
            den_env_prof[k] = np.mean(den_field_env[..., k])

            if other_vars != None:
                num_combo2_prof[k] = np.mean(num_field_combo2[..., k])
                den_combo2_prof[k] = np.mean(den_field_combo2[..., k])
                if len(other_vars) > 1:
                    num_combo3_prof[k] = np.mean(num_field_combo3[..., k])
                    den_combo3_prof[k] = np.mean(den_field_combo3[..., k])

        C_sq_prof = (0.5 * (num_prof / den_prof))
        C_sq_cloud_prof = (0.5 * (num_cloud_prof / den_cloud_prof))
        C_sq_env_prof = (0.5 * (num_env_prof / den_env_prof))

        # C_cloud_prof = dyn.get_Cs(C_sq_cloud_prof)
        # C_env_prof = dyn.get_Cs(C_sq_env_prof)

        C_sq_prof_nc = xr.DataArray(C_sq_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                dims=['time', "zn"], name=f'{save_name}_prof')

        C_sq_cloud_prof_nc = xr.DataArray(C_sq_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                dims=['time', "zn"], name=f'{save_name}_cloud_prof')

        C_sq_env_prof_nc = xr.DataArray(C_sq_env_prof[np.newaxis,...], coords={'time': [nt], 'zn': zn_s},
                             dims=['time', "zn"], name=f'{save_name}_env_prof')



        if other_vars != None:
            C_sq_combo2_prof = (0.5 * (num_combo2_prof / den_combo2_prof))
            #C_combo2_prof = dyn.get_Cs(C_sq_combo2_prof)

            C_sq_combo2_prof_nc = xr.DataArray(C_sq_combo2_prof[np.newaxis,...], coords={'time': [nt], 'zn': zn_s},
                                              dims=['time', "zn"], name=f'{save_name}_{othervar1}_prof')


            if len(other_vars) > 1:
                C_sq_combo3_prof = (0.5 * (num_combo3_prof / den_combo3_prof))
                #C_combo3_prof = dyn.get_Cs(C_sq_combo3_prof)

                C_sq_combo3_prof_nc = xr.DataArray(C_sq_combo3_prof[np.newaxis,...], coords={'time': [nt], 'zn': zn_s},
                                                   dims=['time', "zn"],
                                                   name=f'{save_name}_{othervar1}_{othervar2}_prof')



        if other_vars == None:
            return C_sq_prof_nc, C_sq_env_prof_nc, C_sq_cloud_prof_nc
        else:
            if len(other_vars) == 1:
                return C_sq_prof_nc, C_sq_env_prof_nc, C_sq_cloud_prof_nc, C_sq_combo2_prof_nc
            else:
                return C_sq_prof_nc, C_sq_env_prof_nc, C_sq_cloud_prof_nc, C_sq_combo2_prof_nc, C_sq_combo3_prof_nc


    else:

        data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
        data_field_env = ma.masked_array(data_field, mask=env_only_mask)


        if other_vars != None:
            data_field_combo2 = ma.masked_array(data_field, mask=combo2_out_mask)

            if other_vars[0] == f'f(f(w_on_{grid})_r_on_{grid})_r' or other_vars[0] == f'f(w_on_{grid})_r':
                othervar1 = 'w'
            elif other_vars[0] == f'f(f(th_v_on_{grid})_r_on_{grid})_r' or other_vars[0] == f'f(th_v_on_{grid})_r':
                othervar1 = 'th_v'

            if len(other_vars) > 1:
                data_field_combo3 = ma.masked_array(data_field, mask=combo3_out_mask)

                if other_vars[1] == f'f(f(w_on_{grid})_r_on_{grid})_r' or other_vars[1] == f'f(w_on_{grid})_r':
                    othervar2 = 'w'
                elif other_vars[1] == f'f(f(th_v_on_{grid})_r_on_{grid})_r' or other_vars[1] == f'f(th_v_on_{grid})_r':
                    othervar2 = 'th_v'

        data_prof = np.zeros(z_num)
        data_cloud_prof = np.zeros(z_num)
        data_env_prof = np.zeros(z_num)
        data_combo2_prof = np.zeros(z_num)
        data_combo3_prof = np.zeros(z_num)

        for k in range(z_num):
            data_prof[k] = np.mean(data_field[..., k])
            data_cloud_prof[k] = np.mean(data_field_cloud[..., k])
            data_env_prof[k] = np.mean(data_field_env[..., k])


            if other_vars != None:
                data_combo2_prof[k] = np.mean(data_field_combo2[..., k])

                if len(other_vars) > 1:
                    data_combo3_prof[k] = np.mean(data_field_combo3[..., k])


        data_prof_nc = xr.DataArray(data_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                    dims=['time', "zn"], name=f'{save_name}_prof')

        data_cloud_prof_nc = xr.DataArray(data_cloud_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                          dims=['time', "zn"], name=f'{save_name}_cloud_prof')

        data_env_prof_nc = xr.DataArray(data_env_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                        dims=['time', "zn"], name=f'{save_name}_env_prof')

        if other_vars != None:

            data_combo2_prof_nc = xr.DataArray(data_combo2_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                               dims=['time', "zn"], name=f'{save_name}_{othervar1}_prof')

            if len(other_vars) > 1:

                data_combo3_prof_nc = xr.DataArray(data_combo3_prof[np.newaxis, ...], coords={'time': [nt], 'zn': zn_s},
                                                   dims=['time', "zn"],
                                                   name=f'{save_name}_{othervar1}_{othervar2}_prof')

        if other_vars == None:
            return data_prof_nc, data_env_prof_nc, data_cloud_prof_nc
        else:
            if len(other_vars) == 1:
                return data_prof_nc, data_env_prof_nc, data_cloud_prof_nc, data_combo2_prof_nc
            else:
                return data_prof_nc, data_env_prof_nc, data_cloud_prof_nc, data_combo2_prof_nc, data_combo3_prof_nc





