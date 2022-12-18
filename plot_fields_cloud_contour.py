import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn


homedir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/'
mydir = homedir + 'LijMij_HjRj/BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_cloud = homedir + 'q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/plots/fields_cloud_contour/new_percentiles/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'LijMij_'
dir_th = mydir + 'HjRj_th_'
dir_qt = mydir + 'HjRj_qt_'

in_set_percentile = [25,99]
in_set_percentile_C = [70,99]

time_av_or_not = 'yes' #'yes' #if not then give the time stamp you want to look at

my_axis = 299
my_x_y = 'y'

data_s2 = xr.open_dataset(dir_s+'2D.nc')
data_s4 = xr.open_dataset(dir_s+'4D.nc')
data_s8 = xr.open_dataset(dir_s+'8D.nc')
data_s16 = xr.open_dataset(dir_s+'16D.nc')
data_s32 = xr.open_dataset(dir_s+'32D.nc')
data_s64 = xr.open_dataset(dir_s+'64D.nc')

data_th2 = xr.open_dataset(dir_th+'2D.nc')
data_th4 = xr.open_dataset(dir_th+'4D.nc')
data_th8 = xr.open_dataset(dir_th+'8D.nc')
data_th16 = xr.open_dataset(dir_th+'16D.nc')
data_th32 = xr.open_dataset(dir_th+'32D.nc')
data_th64 = xr.open_dataset(dir_th+'64D.nc')

data_qt2 = xr.open_dataset(dir_qt+'2D.nc')
data_qt4 = xr.open_dataset(dir_qt+'4D.nc')
data_qt8 = xr.open_dataset(dir_qt+'8D.nc')
data_qt16 = xr.open_dataset(dir_qt+'16D.nc')
data_qt32 = xr.open_dataset(dir_qt+'32D.nc')
data_qt64 = xr.open_dataset(dir_qt+'64D.nc')

data_cl2 = xr.open_dataset(dir_cloud+'0.nc')
data_cl4 = xr.open_dataset(dir_cloud+'1.nc')
data_cl8 = xr.open_dataset(dir_cloud+'2.nc')
data_cl16 = xr.open_dataset(dir_cloud+'3.nc')
data_cl32 = xr.open_dataset(dir_cloud+'4.nc')
data_cl64 = xr.open_dataset(dir_cloud+'5.nc')


#%%
#print(data_s2.variables)
#%%
#print(data_th2.variables)
#%%
#print(data_qt2.variables)
#%%

data_s_list = [data_s2, data_s4, data_s8, data_s16, data_s32, data_s64]
data_th_list = [data_th2, data_th4, data_th8, data_th16, data_th32, data_th64]
data_qt_list = [data_qt2, data_qt4, data_qt8, data_qt16, data_qt32, data_qt64]

data_cl_list = [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]



LijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'LM_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not
           }

MijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'MM_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not
           }

Cs_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cs_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
              'set_percentile': in_set_percentile_C,
              't_av_or_not': time_av_or_not
           }


HjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_th_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

RjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'RR_th_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

Cth_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cth_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not
           }


HjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_q_total_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

RjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'RR_q_total_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

Cqt_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cqt_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64],
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not
           }


def plotfield(field, x_or_y, axis_set, data_field_list, set_percentile, data_cl_list, t_av_or_not):

    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(data_field_list)):

        if field == 'Cs_field':
            print('length of time array for LM is ', len(data_field_list[i]['LM_field'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                LM_field = np.mean(data_field_list[i]['LM_field'].data[...], axis=0)
                MM_field = np.mean(data_field_list[i]['MM_field'].data[...], axis=0)
            else:
                LM_field = data_field_list[i]['LM_field'].data[t_av_or_not,...]
                MM_field = data_field_list[i]['MM_field'].data[t_av_or_not, ...]

            data_field_sq = 0.5 * LM_field/MM_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cth_field':
            print('length of time array for HR_th is ', len(data_field_list[i]['HR_th_field'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_field_list[i]['HR_th_field'].data[...], axis=0)
                RR_field = np.mean(data_field_list[i]['RR_th_field'].data[...], axis=0)
            else:
                HR_field = data_field_list[i]['HR_th_field'].data[t_av_or_not, ...]
                RR_field = data_field_list[i]['RR_th_field'].data[t_av_or_not, ...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cqt_field':
            print('length of time array for HR_qt is ', len(data_field_list[i]['HR_q_total_field'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_field_list[i]['HR_q_total_field'].data[...], axis=0)
                RR_field = np.mean(data_field_list[i]['RR_q_total_field'].data[...], axis=0)
            else:
                HR_field = data_field_list[i]['HR_q_total_field'].data[t_av_or_not, ...]
                RR_field = data_field_list[i]['RR_q_total_field'].data[t_av_or_not, ...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        else:
            print(f'length of time array for {field} is ', len(data_field_list[i][f'{field}'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                data_field = np.mean(data_field_list[i][f'{field}'].data[...], axis = 0)
            else:
                data_field = data_field_list[i][f'{field}'].data[t_av_or_not,...]


        print('length of time array for cloud field is ', len(data_cl_list[i]['f(q_cloud_liquid_mass_on_w)_r'].data[:, 0, 0, 0]))
        if t_av_or_not == 'yes':
            cloud_field = np.mean(data_cl_list[i]['f(q_cloud_liquid_mass_on_w)_r'].data[...], axis = 0)
            mytime = 't_av'
        else:
            cloud_field = data_cl_list[i]['f(q_cloud_liquid_mass_on_w)_r'].data[t_av_or_not,...]
            mytime = f't{t_av_or_not}'



        plt.figure(figsize=(16,5))
        plt.title(f'{field}', fontsize=16)
        if x_or_y == 'x':

            if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                myvmin = 0
            else:
                myvmin = np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
            myvmax = np.percentile(data_field[axis_set, :, 5:120], set_percentile[1])

            mylevels = np.linspace(myvmin, myvmax, 8)

            cf = plt.contourf(np.transpose(data_field[axis_set, :, :]), levels=mylevels, extend='both')
            cb = plt.colorbar(cf, format='%.2f')
            #cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='red', linewidths=2, levels=[1e-5])
            plt.xlabel(f'y (cross section with x = {axis_set}) (km)', fontsize=16)

        elif x_or_y == 'y':

            if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                myvmin = 0
            else:
                myvmin = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[0])
            myvmax = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[1])

            mylevels = np.linspace(myvmin, myvmax, 8)

            cf = plt.contourf(np.transpose(data_field[50:351, axis_set, 0:101]), levels=mylevels, extend='both')
            cb = plt.colorbar(cf, format='%.2f')
            cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[50:351, axis_set, 0:101]), colors='red', linewidths=2, levels=[1e-5])
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

                cf = plt.contourf(np.transpose(data_field_sq[axis_set, :, :]), levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                #cb.set_under('k')
                cb.set_label(f'{field}$^2$', size=16)

                plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='red', linewidths=2, levels=[1e-5])
                plt.xlabel(f'y (cross section with x = {axis_set}) (km)')

            elif x_or_y == 'y':

                myvmin = 0 #np.percentile(data_field[:, axis_set, 5:120], set_percentile[0])
                myvmax = np.percentile(data_field_sq[:, axis_set, 5:120], set_percentile[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field_sq[:, axis_set, :]), levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                #cb.set_under('k')
                cb.set_label(f'{field}$^2$', size=16)

                plt.contour(np.transpose(cloud_field[:, axis_set, :]), colors='red', linewidths=2, levels=[1e-5])
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
# plotfield(**Cs_options)
#
#
# plotfield(**HjRj_th_options)

# plotfield(**RjRj_th_options)
#
# plotfield(**Cth_options)
#
#
# plotfield(**HjRj_qt_options)

plotfield(**RjRj_qt_options)

plotfield(**Cqt_options)