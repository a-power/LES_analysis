import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn


homedir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/'
mydir = homedir + 'LijMij_HjRj/BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_cloud = homedir + 'q_l/BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/plots/fields_cloud_contour/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'LijMij_'
dir_th = mydir + 'HjRj_th_'
dir_qt = mydir + 'HjRj_qt_'


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

# data_cl64 = Dataset(dir_cloud+'5.nc', mode='r')
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
#%%
# LM_field = data_s_list['LM_field'].data[...]
# MM_field = data_s_list['MM_field'].data[...]
#
# cloud_field = data_cl_list['f(q_cloud_liquid_mass_on_w)_r'].data[...]
#%%

my_axis = 299
my_x_y = 'y'

LijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'LM_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

MijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'MM_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

Cs_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cs_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }


HjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_th_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

RjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'RR_th_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

Cth_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cth_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }


HjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_q_total_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

RjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'RR_q_total_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

Cqt_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cqt_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }


def plotfield(field, x_or_y, axis_set, data_field_list, data_cl_list):

    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(data_field_list)):

        if field == 'Cs_field':
            LM_field = np.mean(data_field_list[i]['LM_field'].data[...], axis=0)
            MM_field = np.mean(data_field_list[i]['MM_field'].data[...], axis=0)
            data_field_sq = LM_field/MM_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cth_field':
            HR_field = np.mean(data_field_list[i]['HR_th_field'].data[...], axis=0)
            RR_field = np.mean(data_field_list[i]['RR_th_field'].data[...], axis=0)
            data_field_sq = HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cqt_field':
            HR_field = np.mean(data_field_list[i]['HR_q_total_field'].data[...], axis=0)
            RR_field = np.mean(data_field_list[i]['RR_q_total_field'].data[...], axis=0)
            data_field_sq = HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        else:
            data_field = np.mean(data_field_list[i][f'{field}'].data[...], axis = 0)

        cloud_field = np.mean(data_cl_list[i]['f(q_cloud_liquid_mass_on_w)_r'].data[...], axis = 0)

        plt.figure(figsize=(20,7))
        if x_or_y == 'x':

            myvmin = np.percentile(data_field[axis_set, :, 5:120], 5)
            myvmax = np.percentile(data_field[axis_set, :, 5:120], 95)

            mylevels = np.linspace(myvmin, myvmax, 9)

            plt.contourf(np.transpose(data_field[axis_set, :, :]), levels=mylevels, extend='both')
            cb = plt.colorbar()
            cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2.5, levels=[1e-5])
            plt.xlabel(f'y (x = {axis_set})')

        elif x_or_y == 'y':

            myvmin = np.percentile(data_field[:, axis_set, 5:120], 5)
            myvmax = np.percentile(data_field[:, axis_set, 5:120], 95)

            mylevels = np.linspace(myvmin, myvmax, 9)

            plt.contourf(np.transpose(data_field[:, axis_set, :]), levels=mylevels, extend='both')
            cb = plt.colorbar()
            cb.set_label(f'{field}', size=16)

            plt.contour(np.transpose(cloud_field[:, axis_set, :]), colors='black', linewidths=2.5, levels=[1e-5])
            plt.xlabel(f'x (y = {axis_set})')
        else:
            print("axis_set must be 'x' or'y'.")
        plt.ylabel("z")
        plt.savefig(plotdir+f'{field}_{deltas[i]}_field_{x_or_y}={axis_set}.png', pad_inches=0)
        plt.clf()

        if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
            plt.figure(figsize=(20, 7))
            if x_or_y == 'x':

                myvmin = np.percentile(data_field[axis_set, :, 5:120], 5)
                myvmax = np.percentile(data_field[axis_set, :, 5:120], 95)

                mylevels = np.linspace(myvmin, myvmax, 9)

                plt.contourf(np.transpose(data_field[axis_set, :, :]), levels=mylevels, extend='both')
                cb = plt.colorbar()
                cb.set_label(f'{field}', size=16)

                plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='black', linewidths=2.5, levels=[1e-5])
                plt.xlabel(f'y (x = {axis_set})')

            elif x_or_y == 'y':

                myvmin = np.percentile(data_field[:, axis_set, 5:120], 5)
                myvmax = np.percentile(data_field[:, axis_set, 5:120], 95)

                mylevels = np.linspace(myvmin, myvmax, 9)

                plt.contourf(np.transpose(data_field[:, axis_set, :]), levels=mylevels, extend='both')
                cb = plt.colorbar()
                cb.set_label(f'{field}', size=16)

                plt.contour(np.transpose(cloud_field[:, axis_set, :]), colors='black', linewidths=2.5, levels=[1e-5])
                plt.xlabel(f'x (y = {axis_set})')
            else:
                print("axis_set must be 'x' or 'y'.")
            plt.ylabel("z")
            # plt.xlim(0, 1)
            plt.savefig(plotdir + f'{field}_sq_{deltas[i]}_field_{x_or_y}={axis_set}.png', pad_inches=0)
            plt.clf()

        print(f'plotted fields for {field}')

    plt.close('all')


plotfield(**LijMij_options)

plotfield(**MijMij_options)

plotfield(**Cs_options)


plotfield(**HjRj_th_options)

plotfield(**RjRj_th_options)

plotfield(**Cth_options)


plotfield(**HjRj_qt_options)

plotfield(**RjRj_qt_options)

plotfield(**Cqt_options)