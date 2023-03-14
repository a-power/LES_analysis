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
        plt.savefig(plotdir + f'neg_{field}_vs_z_{deltas[i]}.png', pad_inches=0)
        plt.clf()

        print(f'plotted neg vs z for {field}')

    plt.close('all')


def C_values(plotdir, field, data_field_list, data_cl_list, **kwargs):
    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']


    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_cl_list[i])

        data_field = data_field_list[i][f'{field}'].data[...]
        print(np.shape(data_field[...]))

        data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
        data_field_env = ma.masked_array(data_field, mask=env_only_mask)

        print(np.shape(data_field_env))

        if field=='Cs':
            name=field
            scalar='$C_{s}$'
        elif field == 'C_theta':
            name=field
            scalar = '$C_{\\theta}$'
        elif field == 'C_q':
            name=field
            scalar = '$C_{qt}$'

        elif field == 'f(LM_field_on_w)_r':
            scalar = '$L_{ij}M_{ij}$'
            name = 'LM'
        elif field == 'f(HR_th_field_on_w)_r':
            scalar = '$H_{j}R_{j \\theta}$'
            name = 'HR_th'
        elif field == 'f(HR_q_total_field_on_w)_r':
            scalar = '$H_{j}R_{j qt}$'
            name = 'HR_q_total'

        #print('mean')

        plt.figure(figsize=(7, 6))
        plt.hist(data_field_env[...,0:24].flatten(), \
                 bins=500, histtype='step', stacked=False, label="ML", \
                 linewidth = 2, linestyle='solid')
        plt.hist(data_field_env[...,24:151].flatten(), \
                 bins=500, histtype='step', stacked=False, label="CL: clear sky", \
                 linewidth = 2, linestyle='dotted')
        plt.hist(data_field_cloud[...].flatten(), \
                 bins=500, histtype='step', stacked=False, label="CL: cloudy", \
                 linewidth = 2, linestyle='dotted')
        plt.legend()

        # og_xtic = plt.xticks()
        #plt.xlim(-1,1)
        plt.xlabel(f"{scalar}", fontsize=16)
        plt.yscale('log', nonposy='clip')
        plt.ylabel("number of value occurrences", fontsize=16)
        plt.savefig(plotdir + f'dist_of_{name}_values_{deltas[i]}.png', pad_inches=0)
        plt.clf()

        print(f'plotted for {field} {deltas[i]}')

    plt.close('all')


def plotfield(plot_dir, field, x_or_y, axis_set, data_field_in, set_percentile, contour_field_in, t_av_or_not,
              start_end, set_percentile_C2=None)\
        :
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

            fig1, ax1 = plt.subplots(figsize=(16, 5))
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
            plt.xticks(og_xtic[0], np.linspace(start, end, len(og_xtic[0])))
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            og_ytic = plt.yticks()
            plt.yticks(np.linspace(0, 101, 5), np.linspace(0, 2, 5))  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

            plt.savefig(plot_dir + f'{field}_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.png', pad_inches=0)
            plt.clf()

            if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
                fig2, ax2 = plt.subplots(figsize=(16, 5))
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
                plt.xticks(og_xtic[0], np.linspace(start, end, len(og_xtic[0])))
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # og_ytic = plt.yticks()
                plt.yticks(np.linspace(0, 101, 5), np.linspace(0, 2, 5))#plt.yticks(np.linspace(0, 151, 7), np.linspace(0, 3, 7))
                plt.ylabel("z (km)", fontsize=16)
                plt.savefig(plot_dir + f'{field}_sq_{deltas[i]}_{mytime}_{x_or_y}={axis_set}_start_{start}_end_{end}.png', pad_inches=0)
                plt.clf()

            print(f'plotted fields for {field} {mytime}')

    plt.close('all')
