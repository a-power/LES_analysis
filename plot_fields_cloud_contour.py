import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm


homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
contour_data = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/fields_contour/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'Cs_'
dir_th = mydir + 'C_th_'
dir_qt = mydir + 'C_qt_'

in_set_percentile = [25,99]
in_set_percentile_C = [70,99]

time_av_or_not = 'yes' #'yes' #if not then give the time stamp index (integer) you want to look at (eg 0, 1, ..)

my_axis = 299
my_x_y = 'y'



LijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
                   'field': 'LM_field',
                   'data_field_in': dir_s,
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not,
                  'contour_field_in': contour_data
           }

MijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
                   'field': 'MM_field',
                   'data_field_in': dir_s,
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not,
                  'contour_field_in': contour_data
           }

Cs_options = {'axis_set': my_axis,
                'x_or_y': my_x_y,
               'field': 'Cs_field',
               'data_field_in': dir_s,
              'set_percentile': in_set_percentile_C,
              't_av_or_not': time_av_or_not,
              'contour_field_in': contour_data
           }


HjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
                   'field': 'HR_th_field',
                   'data_field_in': dir_th,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not,
                   'contour_field_in': contour_data
           }

RjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
                   'field': 'RR_th_field',
                   'data_field_in': dir_th,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not,
                   'contour_field_in': contour_data
           }

Cth_options = {'axis_set': my_axis,
                'x_or_y': my_x_y,
               'field': 'Cth_field',
               'data_field_in': dir_th,
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not,
               'contour_field_in': contour_data
           }


HjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
                   'field': 'HR_q_total_field',
                   'data_field_in': dir_qt,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not,
                   'contour_field_in': contour_data
           }

RjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
                   'field': 'RR_q_total_field',
                   'data_field_in': dir_qt,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not,
                   'contour_field_in': contour_data
           }

Cqt_options = {'axis_set': my_axis,
               'x_or_y': my_x_y,
               'field': 'Cqt_field',
               'data_field_in': dir_qt,
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not,
               'contour_field_in': contour_data
           }


def plotfield(field, x_or_y, axis_set, data_field_in, set_percentile, contour_field_in, t_av_or_not):

    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(data_field_in)):

        data_set = xr.open_dataset(data_field_in + f'{deltas[i]}_running_mean_filter_rm00.nc')

        if field == 'Cs_field':
            print('length of time array for LM is ', len(data_set['f(LM_field_on_p)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                LM_field = np.mean(data_set['f(LM_field_on_p)_r'].data[...], axis=0)
                MM_field = np.mean(data_set['f(MM_field_on_p)_r'].data[...], axis=0)
            else:
                LM_field = data_set['f(LM_field_on_p)_r'].data[t_av_or_not,...]
                MM_field = data_set['f(MM_field_on_p)_r'].data[t_av_or_not, ...]

            data_field_sq = 0.5 * LM_field/MM_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cth_field':

            print('length of time array for HR_th is ', len(data_set['f(HR_th_field_on_p)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_set['f(HR_th_field_on_p)_r'].data[...], axis=0)
                RR_field = np.mean(data_set['f(RR_th_field_on_p)_r'].data[...], axis=0)
            else:
                HR_field = data_set['f(HR_th_field_on_p)_r'].data[t_av_or_not, ...]
                RR_field = data_set['f(RR_th_field_on_p)_r'].data[t_av_or_not, ...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cqt_field':
            print('length of time array for HR_qt is ', len(data_set['f(HR_q_total_field_on_p)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_set['f(HR_q_total_field_on_p)_r'].data[...], axis=0)
                RR_field = np.mean(data_set['f(RR_q_total_field_on_p)_r'].data[...], axis=0)
            else:
                HR_field = data_set['f(HR_q_total_field_on_p)_r'].data[t_av_or_not, ...]
                RR_field = data_set['f(RR_q_total_field_on_p)_r'].data[t_av_or_not, ...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        else:
            print(f'length of time array for {field} is ', len(data_set[f'f({field}_on_p)_r'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                data_field = np.mean(data_set[f'f({field}_on_p)_r'].data[...], axis = 0)
            else:
                data_field = data_set[f'f({field}_on_p)_r'].data[t_av_or_not,...]

        data_set.close()

        contour_set = xr.open_dataset(contour_field_in + f'{i}_running_mean_filter_rm00.nc')

        print('length of time array for cloud field is ', len(contour_set['f(f(q_cloud_liquid_mass_on_w)_r_on_p)_r'].data[:, 0, 0, 0]))
        if t_av_or_not == 'yes':
            cloud_field = np.mean(contour_set['f(f(q_cloud_liquid_mass_on_p)_r_on_p)_r'].data[...], axis = 0)
            w_field = np.mean(contour_set['f(f(w_on_p)_r_on_p)_r'].data[...], axis=0)
            w2_field = np.mean(contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[...], axis=0)
            th_v_field = np.mean(contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[...], axis=0)
            mytime = 't_av'
        else:
            cloud_field = contour_set['f(f(q_cloud_liquid_mass_on_w)_r_on_p)_r'].data[t_av_or_not,...]
            w_field = contour_set['f(f(w_on_p)_r_on_p)_r'].data[t_av_or_not,...]
            w2_field = contour_set['f(f(w_on_p.w_on_p)_r_on_p)_r'].data[t_av_or_not,...]
            th_v_field = contour_set['f(f(th_v_on_p)_r_on_p)_r'].data[t_av_or_not,...]
            mytime = f't{t_av_or_not}'

        contour_set.close()


        plt.figure(figsize=(16,5))
        plt.title(f'{field}', fontsize=16)

        if x_or_y == 'x':

            if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                myvmin = 0
            else:
                myvmin = np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
            myvmax = np.percentile(data_field[axis_set, :, 5:120], set_percentile[1])

            mylevels = np.linspace(myvmin, myvmax, 8)

            cf = plt.contourf(np.transpose(data_field[axis_set, :, :]), cmap=cm.hot_r, levels=mylevels, extend='both')
            cb = plt.colorbar(cf, format='%.2f')
            #cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2, levels=[1e-5])
            plt.contour(np.transpose(th_v_field[axis_set, :, :]), colors='black', linestyle='dashed', linewidths=1,
                        levels=[0, 1, 2])
            plt.contour(np.transpose(w_field[axis_set, :, :]), colors='blue', linestyle='dashed', linewidths=1,
                        levels=[0, 0.5, 1])
            # plt.contour(np.transpose(w2_field[axis_set, :, :]), colors='blue', linewidths=1, levels=[0])
            plt.xlabel(f'y (cross section with x = {axis_set}) (km)', fontsize=16)

        elif x_or_y == 'y':

            if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                myvmin = 0
            else:
                myvmin = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[0])
            myvmax = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[1])

            mylevels = np.linspace(myvmin, myvmax, 8)

            cf = plt.contourf(np.transpose(data_field[50:351, axis_set, 0:101]), cmap=cm.hot_r, levels=mylevels, extend='both')
            cb = plt.colorbar(cf, format='%.2f')
            cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2, levels=[1e-5])
            plt.contour(np.transpose(th_v_field[axis_set, :, :]), colors='black', linestyle='dashed', linewidths=1,
                        levels=[0, 1, 2])
            plt.contour(np.transpose(w_field[axis_set, :, :]), colors='blue', linestyle='dashed', linewidths=1,
                        levels=[0, 0.5, 1])
            # plt.contour(np.transpose(w2_field[axis_set, :, :]), colors='blue', linewidths=1, levels=[0])
            plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
        else:
            print("axis_set must be 'x' or'y'.")
        plt.ylabel("z (km)", fontsize=16)
        og_xtic = plt.xticks()
        plt.xticks(og_xtic[0], np.linspace(1, 7, len(og_xtic[0])))
        og_ytic = plt.yticks()
        plt.yticks(np.linspace(0, 101, 5) , np.linspace(0, 2, 5)) # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

        plt.savefig(plotdir+f'zoomed_{field}_{deltas[i]}_{mytime}_{x_or_y}={axis_set}.png', pad_inches=0)
        plt.clf()

        if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
            plt.figure(figsize=(20, 5))
            plt.title(f'{field}$^2$', fontsize=16)
            if x_or_y == 'x':

                myvmin = 0 #np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
                myvmax = np.percentile(data_field_sq[axis_set, :, 5:120], set_percentile[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field_sq[axis_set, :, :]), cmap=cm.bwr, norm=TwoSlopeNorm(0),
                                  levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                #cb.set_under('k')
                cb.set_label(f'{field}$^2$', size=16)

                plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2, levels=[1e-5])
                plt.contour(np.transpose(th_v_field[axis_set, :, :]), colors='black', linestyle='dashed', linewidths=1,
                            levels=[0, 1, 2])
                plt.contour(np.transpose(w_field[axis_set, :, :]), colors='gray', linestyle='dashed', linewidths=1,
                            levels=[0, 0.5, 1])
                # plt.contour(np.transpose(w2_field[axis_set, :, :]), colors='blue', linewidths=1, levels=[0])
                plt.xlabel(f'y (cross section with x = {axis_set}) (km)')

            elif x_or_y == 'y':

                myvmin = 0 #np.percentile(data_field[:, axis_set, 5:120], set_percentile[0])
                myvmax = np.percentile(data_field_sq[:, axis_set, 5:120], set_percentile[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field_sq[:, axis_set, :]), cmap=cm.bwr, norm=TwoSlopeNorm(0),
                                  levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                #cb.set_under('k')
                cb.set_label(f'{field}$^2$', size=16)

                plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2, levels=[1e-5])
                plt.contour(np.transpose(th_v_field[axis_set, :, :]), colors='black', linestyle='dashed', linewidths=1, levels=[0, 1, 2])
                plt.contour(np.transpose(w_field[axis_set, :, :]), colors='gray', linestyle='dashed', linewidths=1, levels=[0, 0.5, 1])
                # plt.contour(np.transpose(w2_field[axis_set, :, :]), colors='blue', linewidths=1, levels=[0])
                plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
            else:
                print("axis_set must be 'x' or 'y'.")

            og_xtic = plt.xticks()
            plt.xticks(og_xtic[0],np.linspace(0, 16, len(og_xtic[0])))
            og_ytic = plt.yticks()
            plt.yticks(np.linspace(0, 151, 7) ,np.linspace(0, 3, 7))
            plt.ylabel("z (km)")
            plt.savefig(plotdir + f'{field}_sq_{deltas[i]}_{mytime}_{x_or_y}={axis_set}.png', pad_inches=0)
            plt.clf()

        print(f'plotted fields for {field}')

    plt.close('all')


# plotfield(**LijMij_options)
#
# plotfield(**MijMij_options)
#
plotfield(**Cs_options)
#
#
# plotfield(**HjRj_th_options)

# plotfield(**RjRj_th_options)
#
plotfield(**Cth_options)
#
#
# plotfield(**HjRj_qt_options)

# plotfield(**RjRj_qt_options)

plotfield(**Cqt_options)