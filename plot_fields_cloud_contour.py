import numpy as np
import os
import analysis_plot_fns as apf

homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
contour_data = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/fields_contour/standard_cb/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'Cs_'
dir_th = mydir + 'C_th_'
dir_qt = mydir + 'C_qt_'

in_set_percentile = [25,99]
in_set_percentile_C = [70,99]
in_set_percentile_C2 = ['min',99] #note that the first entry of this can be 'min' or a number representing the percentile

x_axis_start_end = [0, 4] #start and end points in km

set_cb_in = [[0.15, 0.30], [-0.1, 0.1]] #C_min, C_max, C^2_min, C^2_max = [[None, None], [None, None]]

time_av_or_not = np.array([2]) #, 1, 2 #'yes' (in the array) #if not then give the time stamp index/indices (integer) you want to look at (eg 0, 1, ..)

my_axis = 300
my_x_y = 'y' # must be y due to direction of wind

Deltas = ['2D', '4D', '8D', '16D', '32D', '64D']


general_options = {'set_cb': set_cb_in,
                    'axis_set': my_axis,
                    'x_or_y': my_x_y,
                    'set_percentile': in_set_percentile,
                    't_av_or_not': time_av_or_not,
                    'contour_field_in': contour_data,
                    'plotdir': plotdir,
                    'start_end': x_axis_start_end,
                    'deltas': Deltas
                    }

LijMij_options = {'field': 'LM_field',
                   'data_field_in': dir_s
                  }

MijMij_options = {'field': 'MM_field',
                   'data_field_in': dir_s
           }

Cs_options = {'field': 'Cs_field',
               'data_field_in': dir_s,
              'set_percentile': in_set_percentile_C,
              'set_percentile_C2': in_set_percentile_C2
           }


HjRj_th_options = {'field': 'HR_th_field',
                   'data_field_in': dir_th
           }

RjRj_th_options = {'field': 'RR_th_field',
                   'data_field_in': dir_th
           }

Cth_options = {'field': 'Cth_field',
               'data_field_in': dir_th,
               'set_percentile': in_set_percentile_C,
               'set_percentile_C2': in_set_percentile_C2
           }


HjRj_qt_options = {'field': 'HR_q_total_field',
                   'data_field_in': dir_qt
           }

RjRj_qt_options = {'field': 'RR_q_total_field',
                   'data_field_in': dir_qt
           }

Cqt_options = {'field': 'Cqt_field',
               'data_field_in': dir_qt,
               'set_percentile': in_set_percentile_C,
               'set_percentile_C2': in_set_percentile_C2
           }




# apf.plotfield(plotdir, start_end=x_axis_start_end, **LijMij_options)
#
# apf.plotfield(plotdir, start_end=x_axis_start_end, **MijMij_options)
#
apf.plotfield(**general_options, **Cs_options)
#
#
# apf.plotfield(plotdir, start_end=x_axis_start_end, **HjRj_th_options)

# apf.plotfield(plotdir, start_end=x_axis_start_end, **RjRj_th_options)
#
apf.plotfield(**general_options, **Cth_options)
#
#
# apf.plotfield(plotdir, start_end=x_axis_start_end, **HjRj_qt_options)

# apf.plotfield(plotdir, start_end=x_axis_start_end, **RjRj_qt_options)

apf.plotfield(**general_options, **Cqt_options)
