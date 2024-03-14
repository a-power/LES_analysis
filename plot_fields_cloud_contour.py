import numpy as np
import os
import analysis_plot_fns as apf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case_in', type=str, default='ARM') #BOMEX
parser.add_argument('--times', type=str, default='32400')
parser.add_argument('--var_f', type=str, default='w') #'w', 'TKE', 'w_th_v'
parser.add_argument('--x_y', type=str, default='y')
parser.add_argument('--axis', type=int, default=297)
parser.add_argument('--x_s', type=float, default=3)#0
parser.add_argument('--x_e', type=float, default=6)#19.2) 5.5)
args = parser.parse_args()
case = args.case_in
set_time = args.times
my_x_y = args.x_y # must be y due to direction of wind in BOMEX
my_axis = args.axis
x_start = args.x_s
x_end = args.x_e
set_var_field = args.var_f

deltas_in = ['0_0']#, '1_0', '2_0', '3_0', '4_0', '5_0']
#['2D', '4D', '8D', '16D', '32D', '64D']

set_cb_in = [-1.5, 2.0] #[None, None]


set_C_perc_1st = 98
set_C_perc_2nd = None


if case =='BOMEX':
    homedir = '/work/scratch-pw3/apower/BOMEX/second_filt/'
    mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
    contour_data = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

    plotdir_in = '/gws/nopw/j04/paracon_rdg/users/apower/plots/BOMEX_fields/cloud_contour/'

    Dx_grid = 20
    z_top = 101
    time_av_or_not = np.array([0, 1, 2])
    z_tix = np.linspace(0, z_top, 5)
    z_labels = np.linspace(0, 2, 5)
    #0, 1, 2 #'yes' (in the array)
    # #if not then give the time stamp index/indices (integer) you want to look at (eg 0, 1, ..)

elif case == 'ARM':
    homedir = '/work/scratch-pw3/apower/ARM/second_filt/'
    mydir = homedir + f"diagnostics_3d_ts_{set_time}_gaussian_filter_"
    contour_data = homedir + f"diagnostics_3d_ts_{set_time}_gaussian_filter_ga0"

    plotdir_in = f'/gws/nopw/j04/paracon_rdg/users/apower/plots/ARM_fields/cloud_contour/'

    z_top = 250
    z_tix = np.linspace(0, z_top, 6)
    z_labels = np.linspace(0, 2.5, 6)

    Dx_grid = 25
    time_av_or_not = np.array([0, 1])
    #0, 1 #'yes' (in the array)
    # #if not then give the time stamp index/indices (integer) you want to look at (eg 0, 1, ..)

else:
    print("case isn't yet coded for")

os.makedirs(plotdir_in, exist_ok = True)

dir_s = mydir + 'Cs_'
dir_th = mydir + 'C_th_'
dir_qt = mydir + 'C_qt_'

in_set_percentile = None #[25,99] for C
in_set_percentile_C = [1, 99] # None #[70,99]
in_set_percentile_C2 = None #['min',99]
# #note that the first entry of this can be 'min' or a number representing the percentile, or None

x_axis_start_end = [x_start, x_end] #start and end points in km



general_options = {'set_cb': set_cb_in,
                    'axis_set': my_axis,
                    'x_or_y': my_x_y,
                    't_av_or_not': time_av_or_not,
                    'contour_field_in': contour_data,
                    'plot_dir': plotdir_in,
                    'start_end': x_axis_start_end,
                    'deltas': deltas_in,
                    'delta_grid': Dx_grid,
                    'z_top_in': z_top,
                    'z_tix_in': z_tix,
                    'z_labels_in': z_labels
                    }


plot_var_fields_options = {'plot_dir': plotdir_in,
                            'axis_set': my_axis,
                            'var_path': contour_data,
                            'x_or_y': my_x_y,
                            't_av_or_not': time_av_or_not,
                            'var_field': set_var_field,
                            'start_end': x_axis_start_end,
                            'deltas': deltas_in,
                            'delta_grid': Dx_grid,
                            'z_top_in': z_top,
                            'z_tix_in': z_tix,
                            'z_labels_in': z_labels,
                            'C_perc_1st': set_C_perc_1st,
                            'C_perc_2nd': set_C_perc_2nd
                            }



LijMij_options = {'field': 'LM_field',
                   'data_field_in': dir_s,
                    'set_percentile': in_set_percentile
                  }

MijMij_options = {'field': 'MM_field',
                   'data_field_in': dir_s,
                    'set_percentile': in_set_percentile
           }

Cs_options = {'field': 'Cs_field',
               'data_field_in': dir_s,
              'set_percentile': in_set_percentile_C,
              'set_percentile_C_sq': in_set_percentile_C2
           }


HjRj_th_options = {'field': 'HR_th_field',
                   'data_field_in': dir_th,
                    'set_percentile': in_set_percentile
           }

RjRj_th_options = {'field': 'RR_th_field',
                   'data_field_in': dir_th,
                    'set_percentile': in_set_percentile
           }

Cth_options = {'field': 'Cth_field',
               'data_field_in': dir_th,
               'set_percentile': in_set_percentile_C,
               'set_percentile_C_sq': in_set_percentile_C2
           }


HjRj_qt_options = {'field': 'HR_q_total_field',
                   'data_field_in': dir_qt,
                    'set_percentile': in_set_percentile
           }

RjRj_qt_options = {'field': 'RR_q_total_field',
                   'data_field_in': dir_qt,
                    'set_percentile': in_set_percentile
           }

Cqt_options = {'field': 'Cqt_field',
               'data_field_in': dir_qt,
               'set_percentile': in_set_percentile_C,
               'set_percentile_C_sq': in_set_percentile_C2
           }


# apf.plot_cloud_field(plotdir_in, x_or_y=my_x_y, axis_set=my_axis, set_percentile=None, var_field=set_var_field,
#                  var_path=contour_data, t_av_or_not=time_av_or_not, start_end=x_axis_start_end, z_top_in=z_top,
#                  z_tix_in=z_tix, z_labels_in=z_labels, deltas=deltas_in, set_cb=set_cb_in, delta_grid=25)

# # apf.plotfield(plotdir, start_end=x_axis_start_end, **LijMij_options)
# #
# # apf.plotfield(plotdir, start_end=x_axis_start_end, **MijMij_options)
# #
apf.plot_C_contours(**plot_var_fields_options, **Cs_options)
# #
# #
# # apf.plotfield(plotdir, start_end=x_axis_start_end, **HjRj_th_options)
#
# # apf.plotfield(plotdir, start_end=x_axis_start_end, **RjRj_th_options)
# #
# apf.plot_C_contours(**plot_var_fields_options, **Cth_options)
# #
# #
# # apf.plotfield(plotdir, start_end=x_axis_start_end, **HjRj_qt_options)
#
# # apf.plotfield(plotdir, start_end=x_axis_start_end, **RjRj_qt_options)
#
# apf.plot_C_contours(**plot_var_fields_options, **Cqt_options)
