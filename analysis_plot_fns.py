import numpy as np
import matplotlib.pyplot as plt
import dynamic_functions as dyn
import mask_cloud_vs_env as clo
import numpy.ma as ma
import dynamic_functions as dyn
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import xarray as xr
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
        plt.savefig(plotdir + f'neg_{field}_vs_z_{deltas[i]}.png', bbox_inches='tight')
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
    plt.savefig(plotdir_in + f'hist_of_{name}_values_{delta}_time_{time_set_in}_vars_{data_names[0]}.png',
                bbox_inches='tight')
    plt.clf()

    print(f'plotted for time {time_set_in} {field_in} {delta}')


def C_values_dist(plotdir, field, data_field_list, data_contour, set_bins, deltas=None, times='av', grid='p', other_vars=None,
                  other_var_thres=None, less_greater_in=['less'], and_or_in = ['and'], cloud_liquid_threshold_in=10**(-5),
                  res_counter_in=None, return_all_in = False, grid_in='p', **kwargs):

    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(deltas)):
        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_contour+f'{i}_running_mean_filter_rm00.nc')

        data_field = data_field_list+f'{deltas[i]}_running_mean_filter_rm00.nc'
        data_field_in = xr.open_dataset(data_field)

        if other_vars!=None:
            if return_all_in == False:
                if len(other_vars) == 1:
                    combo2_out_mask = clo.cloudy_and_or(data_contour + f'{i}_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask = clo.cloudy_and_or(data_contour + f'{i}_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
            else:
                if len(other_vars) == 1:
                    combo2_out_mask, cloud_mask, var_mask = clo.cloudy_and_or(data_contour + f'{i}_running_mean_filter_rm00.nc',
                                                               other_var=other_vars, var_thres=other_var_thres,
                                                               less_greater=less_greater_in, and_or = and_or_in,
                                                               cloud_liquid_threshold=cloud_liquid_threshold_in,
                                                               res_counter=res_counter_in, return_all = return_all_in,
                                                               grid=grid_in)
                else:
                    combo2_out_mask, combo3_out_mask, cloud_mask, var_mask, extra_var_mask = \
                        clo.cloudy_and_or(data_contour + f'{i}_running_mean_filter_rm00.nc',
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
                  len(data_field_in[f'f(HR_q_total_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_field_in[f'f(HR_q_total_field_on_{grid})_r'].data[...]
            den_field = data_field_in[f'f(RR_q_total_field_on_{grid})_r'].data[...]

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


    plt.close('all')


def plotfield(plot_dir, field, x_or_y, axis_set, data_field_in, set_percentile, contour_field_in, t_av_or_not,
              start_end, set_percentile_C2=None, deltas=None):

    if deltas==None:
        deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    start = start_end[0]
    start_grid = int(start/0.02) # going from km to grid spacing co-ords (20m grid)
    end = start_end[1]
    end_grid = int(end/0.02) # going from km to grid spacing co-ords (20m grid)


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

        data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

        for t_set in range(len(t_av_or_not)):
            if field == 'Cs_field':
                print('length of time array for LM is ', len(data_set['f(LM_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                    elif x_or_y == 'y':
                        LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, axis_set, ...]
                    elif x_or_y == 'y':
                        LM_field = data_set['f(LM_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        MM_field = data_set['f(MM_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * LM_field / MM_field
                data_field = dyn.get_Cs(data_field_sq)

            elif field == 'Cth_field':

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

            elif field == 'Cqt_field':
                print('length of time array for HR_qt is ',
                      len(data_set['f(HR_q_total_field_on_p)_r'].data[:, 0, 0, 0]))
                if t_av_or_not == 'yes':
                    if x_or_y == 'x':
                        HR_field = np.mean(data_set['f(HR_q_total_field_on_p)_r'].data[:, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_field_on_p)_r'].data[:, axis_set, ...], axis=0)

                    elif x_or_y == 'y':
                        HR_field = np.mean(data_set['f(HR_q_total_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                        RR_field = np.mean(data_set['f(RR_q_total_field_on_p)_r'].data[:, :, axis_set, ...], axis=0)
                else:
                    if x_or_y == 'x':
                        HR_field = data_set['f(HR_q_total_field_on_p)_r'].data[t_set, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_field_on_p)_r'].data[t_set, axis_set, ...]

                    elif x_or_y == 'y':
                        HR_field = data_set['f(HR_q_total_field_on_p)_r'].data[t_set, :, axis_set, ...]
                        RR_field = data_set['f(RR_q_total_field_on_p)_r'].data[t_set, :, axis_set, ...]

                data_field_sq = 0.5 * HR_field / RR_field
                data_field = dyn.get_Cs(data_field_sq)

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

            contour_set = xr.open_dataset(contour_field_in + f'{i}_running_mean_filter_rm00.nc')

            print('length of time array for cloud field is ',
                  len(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                if x_or_y == 'x':
                    cloud_field = np.mean(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[:, axis_set, ...], axis=0)
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

            fig1, ax1 = plt.subplots(figsize=(20, 5))
            plt.title(f'{field_name} with $\\Delta = $ {deltas[i]}', fontsize=16)

            # if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
            #     myvmin = 0
            # else:
            myvmin = np.percentile(data_field[start_grid:end_grid, 5:120], set_percentile[0])
            myvmax = np.percentile(data_field[start_grid:end_grid, 5:120], set_percentile[1])

            mylevels = np.linspace(myvmin, myvmax, 8)

            mycmap = plt.get_cmap('YlOrRd').copy()
            mycmap.set_extremes(under='white', over='maroon')

            cf = plt.contourf(np.transpose(data_field[start_grid:end_grid, 0:101]), cmap=mycmap, levels=mylevels,
                              extend='both')
            cb = plt.colorbar(cf, format='%.2f')
            cb.set_label(f'{field_name}', size=16)

            cl_c = plt.contour(np.transpose(cloud_field[start_grid:end_grid, 0:101]), colors='black', linewidths=2, levels=[1e-5])
            th_v_c = plt.contour(np.transpose(th_v_field[start_grid:end_grid, 0:101]), colors='black', linestyles='dashed',
                        linewidths=1)  # , levels=[0.1, 1, 2])
            ax1.clabel(th_v_c, inline=True, fontsize=10)
            w_c = plt.contour(np.transpose(w_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=1,
                        levels=[0.1, 0.5])
            ax1.clabel(w_c, inline=True, fontsize=8)
            # plt.contour(np.transpose(w2_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=1, levels=[0.1])
            plt.xlabel(f'x (cross section with {x_or_y} = {axis_set*20/1000}) (km)', fontsize=16)

            plt.ylabel("z (km)", fontsize=16)
            og_xtic = plt.xticks()
            plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0]))))
            # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            og_ytic = plt.yticks()
            plt.yticks(np.linspace(0, 101, 5), np.linspace(0, 2, 5))  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

            plt.savefig(plot_dir + f'{field}_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.png',
                        bbox_inches='tight')
            plt.clf()

            if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
                fig2, ax2 = plt.subplots(figsize=(20, 5))
                plt.title(f'{field_name_sq} with $\\Delta = $ {deltas[i]}', fontsize=16)

                if set_percentile_C2[0] == 'min':
                    myvmin_temp = np.min(data_field_sq[start_grid:end_grid, 5:120])
                    myvmin = myvmin_temp + abs(0.6*myvmin_temp)
                else:
                    myvmin = np.percentile(data_field_sq[start_grid:end_grid, 5:120], set_percentile_C2[0])
                myvmax = np.percentile(data_field_sq[start_grid:end_grid, 5:120], set_percentile_C2[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field_sq[start_grid:end_grid, 0:101]), cmap=cm.bwr,
                                  norm=TwoSlopeNorm(vmin=myvmin, vcenter=0, vmax=myvmax),
                                  levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                # cb.set_under('k')
                cb.set_label(f'{field_name_sq}', size=16)

                cl_c = plt.contour(np.transpose(cloud_field[start_grid:end_grid, 0:101]), colors='black', linewidths=2,
                                   levels=[1e-5])
                th_v_c = plt.contour(np.transpose(th_v_field[start_grid:end_grid, 0:101]), colors='black', linestyles='dashed',
                                     linewidths=1)  # , levels=[0.1, 1, 2])
                ax2.clabel(th_v_c, inline=True, fontsize=10)
                w_c = plt.contour(np.transpose(w_field[start_grid:end_grid, 0:101]), colors='darkslategrey',
                                  linewidths=1, levels=[0.1, 0.5])
                ax2.clabel(w_c, inline=True, fontsize=8)
                # plt.contour(np.transpose(w2_field[start_grid:end_grid, 0:101]), colors='darkslategrey', linewidths=1, levels=[0.1])
                plt.xlabel(f'y (cross section with {x_or_y} = {axis_set*20/1000}) (km)', fontsize=16)

                og_xtic = plt.xticks()
                plt.xticks(og_xtic[0], np.round(np.linspace(start, end, len(og_xtic[0]))))
                # ax2.set_xticks(np.linspace(start, end, len(og_xtic[0])))
                # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # og_ytic = plt.yticks()
                plt.yticks(np.linspace(0, 101, 5), np.linspace(0, 2, 5))#plt.yticks(np.linspace(0, 151, 7), np.linspace(0, 3, 7))
                plt.ylabel("z (km)", fontsize=16)
                plt.savefig(plot_dir + f'{field}_sq_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.png',
                            bbox_inches='tight')
                plt.clf()

            print(f'plotted fields for {field} {mytime}')

    plt.close('all')





def get_conditional_profiles(dataset_in, contour_field_in, field, deltas=None,
                      cloud_thres = 10**(-5), other_vars = False, other_var_thres=False,
                             less_greater_in='less', and_or_in = 'and', grid='p'):
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

    elif field == 'LM_field':
        field_name = '$LM$'
    elif field == 'HR_th_field':
        field_name = '$HR_{\\theta}$'
    elif field == 'HR_qt_field':
        field_name = '$HR_{qt}$'
    elif field == 'MM_field':
        field_name = '$MM$'
    elif field == 'RR_th_field':
        field_name = '$RR_{\\theta}$'
    elif field == 'RR_qt_field':
        field_name = '$RR_{qt}$'
    else:
        print('field not found')

    for i in range(len(deltas)):
        cloud_only_mask, env_only_mask = \
            clo.cloud_vs_env_masks(contour_field_in + f'{i}_running_mean_filter_rm00.nc', \
                                   cloud_liquid_threshold=cloud_thres)

        if other_vars != None:
            if len(other_vars) == 1:
                combo2_out_mask = clo.cloudy_and_or(contour_field_in + f'{i}_running_mean_filter_rm00.nc',
                                                    other_var=other_vars, var_thres=other_var_thres,
                                                    less_greater=less_greater_in, and_or=and_or_in,
                                                    cloud_liquid_threshold=cloud_thres, grid=grid)
            else:
                combo2_out_mask, combo3_out_mask = clo.cloudy_and_or(contour_field_in + f'{i}_running_mean_filter_rm00.nc',
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
            print('length of time array for LM is ', len(data_set[f'f(LM_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_set[f'f(LM_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(MM_field_on_{grid})_r'].data[...]
            C=True

        elif field == 'Cth_field' or field == 'Cth_sq_field':
            print('length of time array for HR_th is ', len(data_set[f'f(HR_th_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_set[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_th_field_on_{grid})_r'].data[...]
            C=True

        elif field == 'Cqt_field' or field == 'Cqt_sq_field':
            print('length of time array for HR_qt is ', len(data_set[f'f(HR_q_total_field_on_{grid})_r'].data[:, 0, 0, 0]))
            num_field = data_set[f'f(HR_q_total_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_q_total_field_on_{grid})_r'].data[...]
            C=True

        else:
            print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_{grid})_r'].data[:, 0, 0, 0]))
            data_field = data_set[f'f({field}_on_{grid})_r'].data[...]
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
                if other_vars[0] == 'f(f(w_on_p)_r_on_p)_r':
                    othervar1 = 'w'
                elif other_vars[0] == 'f(f(th_v_on_p)_r_on_p)_r':
                    othervar1 = 'th'

                if len(other_vars) > 1:
                    num_field_combo3 = ma.masked_array(num_field, mask=combo3_out_mask)
                    den_field_combo3 = ma.masked_array(den_field, mask=combo3_out_mask)
                    if other_vars[1] == 'f(f(w_on_p)_r_on_p)_r':
                        othervar2 = 'w'
                    elif other_vars[1] == 'f(f(th_v_on_p)_r_on_p)_r':
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

            print('need to finish coding for vars other than C')
            # data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
            # data_field_env = ma.masked_array(data_field, mask=env_only_mask)

            # if other_vars != None:
            #     data_field_combo2 = ma.masked_array(data_field, mask=combo2_out_mask)
            #     if len(other_vars) > 1:
            #         data_field_combo3 = ma.masked_array(data_field, mask=combo3_out_mask)







