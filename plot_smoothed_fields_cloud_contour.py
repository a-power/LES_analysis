import xarray as xr
import os
import analysis_plot_fns as apf


mydir = '/work/scratch-pw2/apower/20m_gauss_dyn/filtered_LM_HR_fields/'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/20m_gauss_dyn/plots/smoothed_fields_cloud_contour/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_LijMij_2D_running_mean_filter_rm0'
dir_th = mydir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_HjRj_th_2D_running_mean_filter_rm0'
dir_qt = mydir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_HjRj_qt_2D_running_mean_filter_rm0'
dir_cloud = mydir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga00_running_mean_filter_rm0'

in_set_percentile = [25,99]
in_set_percentile_C = [70,99]

time_av_or_not = 0 #'yes' #'yes' #if not then give the time stamp index (integer) you want to look at (eg 0, 1, ..)

my_axis = 299
my_x_y = 'y'

data_s2 = xr.open_dataset(dir_s+'0.nc')
data_s4 = xr.open_dataset(dir_s+'1.nc')
data_s8 = xr.open_dataset(dir_s+'2.nc')

data_th2 = xr.open_dataset(dir_th+'0.nc')
data_th4 = xr.open_dataset(dir_th+'1.nc')
data_th8 = xr.open_dataset(dir_th+'2.nc')

data_qt2 = xr.open_dataset(dir_qt+'0.nc')
data_qt4 = xr.open_dataset(dir_qt+'1.nc')
data_qt8 = xr.open_dataset(dir_qt+'2.nc')

data_cl2 = xr.open_dataset(dir_cloud+'0.nc')
data_cl4 = xr.open_dataset(dir_cloud+'1.nc')
data_cl8 = xr.open_dataset(dir_cloud+'2.nc')



s_list = [data_s2, data_s4, data_s8]#, data_s16, data_s32, data_s64]
th_list = [data_th2, data_th4, data_th8]#, data_th16, data_th32, data_th64]
qt_list = [data_qt2, data_qt4, data_qt8]#, data_qt16, data_qt32, data_qt64]

cl_list = [data_cl2, data_cl4, data_cl8]#, data_cl16, data_cl32, data_cl64]



LijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'LM_field',
           'data_field_list': s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not
           }

MijMij_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'MM_field',
           'data_field_list': s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
                  'set_percentile': in_set_percentile,
                  't_av_or_not': time_av_or_not
           }

Cs_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cs_field',
           'data_field_list': s_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
              'set_percentile': in_set_percentile_C,
              't_av_or_not': time_av_or_not
           }


HjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'HR_th_field',
           'data_field_list': th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

RjRj_th_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'RR_th_field',
           'data_field_list': th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

Cth_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cth_field',
           'data_field_list': th_list,
           'data_cl_list': [data_cl2, data_cl4, data_cl8],# data_cl16, data_cl32, data_cl64],
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not
           }


HjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'f(HR_q_total_field_on_w)_r',
           'data_field_list': qt_list,
           'data_cl_list': cl_list,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

RjRj_qt_options = {'axis_set': my_axis,
                   'x_or_y': my_x_y,
           'field': 'f(RR_q_total_field_on_w)_r',
           'data_field_list': qt_list,
           'data_cl_list': cl_list,
                   'set_percentile': in_set_percentile,
                   't_av_or_not': time_av_or_not
           }

Cqt_options = {'axis_set': my_axis,
                  'x_or_y': my_x_y,
           'field': 'Cqt_field',
           'data_field_list': qt_list,
           'data_cl_list': cl_list,
               'set_percentile': in_set_percentile_C,
               't_av_or_not': time_av_or_not
           }


# plotfield(**LijMij_options)
#
# plotfield(**MijMij_options)

apf.plotfield(**Cs_options)


# apf.plotfield(**HjRj_th_options)
#
# apf.plotfield(**RjRj_th_options)

apf.plotfield(**Cth_options)


# apf.plotfield(**HjRj_qt_options)
#
# apf.plotfield(**RjRj_qt_options)

apf.plotfield(**Cqt_options)