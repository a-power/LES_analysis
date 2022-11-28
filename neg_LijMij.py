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


data_s_list = [data_s2, data_s4, data_s8, data_s16, data_s32, data_s64]
data_th_list = [data_th2, data_th4, data_th8, data_th16, data_th32, data_th64]
data_qt_list = [data_qt2, data_qt4, data_qt8, data_qt16, data_qt32, data_qt64]

data_cl_list = [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
#%%
# LM_field = data_s_list['LM_field'].data[...]
# MM_field = data_s_list['MM_field'].data[...]
#


LijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'LM_field',
           'data_field_list': data_s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

HjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_th_field',
           'data_field_list': data_th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

HjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_qt_field',
           'data_field_list': data_qt_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8, data_cl16, data_cl32, data_cl64]
           }

##Need to time average