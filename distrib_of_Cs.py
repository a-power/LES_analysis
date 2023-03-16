import xarray as xr
import os
import analysis_plot_fns as apf
import numpy as np


homedir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/smoothed_LM_HR_fields/'
mydir = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_'
dir_contour = homedir + 'BOMEX_m0020_g0800_all_14400_gaussian_filter_ga0'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir, exist_ok = True)

dir_s = mydir + 'Cs_'
dir_th = mydir + 'C_th_'
dir_qt = mydir + 'C_qt_'

mygrid = 'p'
all_times_or_not = 'av' #np.array([0, 1, 2])

cloud_field = f'f(f(q_cloud_liquid_mass_on_{mygrid})_r_on_{mygrid})_r'
w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'



gen_options = {'plotdir': plotdir,
            'data_contour': dir_contour,
            'deltas': None,
            'other_vars': [w_field, th_v_field],
            'cloud_liquid_threshold_in': 10**(-5),
            'other_var_thres': [0.5, 0],
            'less_greater_in': ['less', 'less'],
            'and_or_in': ['and', 'and'],
            'times': all_times_or_not,
            'grid': mygrid,
            'return_all_in': False,
            'set_bins':50
            }


LijMij_options = {'field': 'f(LM_field_on_w)_r',
           'data_field_list': dir_s
           }
Cs_options = {'field': 'Cs_field',
           'data_field_list': dir_s
           }
Cs_sq_options = {'field': 'Cs_sq_field',
           'data_field_list': dir_s
           }


HjRj_th_options = {'field': 'f(HR_th_field_on_w)_r',
           'data_field_list': dir_th
           }
Cth_options = {'field': 'Cth_field',
           'data_field_list': dir_th
           }
Cth_sq_options = {'field': 'Cth_sq_field',
           'data_field_list': dir_th
           }


HjRj_qt_options = {'field': 'f(HR_q_total_field_on_w)_r',
           'data_field_list': dir_qt
           }
Cqt_options = {'field': 'Cqt_field',
           'data_field_list': dir_qt
           }
Cqt_sq_options = {'field': 'Cqt_sq_field',
           'data_field_list': dir_qt
           }




apf.C_values_dist(**Cs_sq_options, **gen_options)
print('Cs_sq fn done')

apf.C_values_dist(**Cth_sq_options, **gen_options)
print('Cth_sq fn done')

apf.C_values_dist(**Cqt_sq_options, **gen_options)
print('Cqt_sq fn done')
