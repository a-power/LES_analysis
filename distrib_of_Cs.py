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

all_times_or_not = np.array([0, 1, 2])


LijMij_options = {'plotdir': plotdir,
                  'field': 'f(LM_field_on_w)_r',
           'data_field_list': dir_s,
           'data_cl_list': dir_contour
           }

print('LijMij options done')

HjRj_th_options = {'plotdir': plotdir,
                   'field': 'f(HR_th_field_on_w)_r',
           'data_field_list': dir_th,
           'data_cl_list': dir_contour
           }

print('HjRj_th options done')

HjRj_qt_options = {'plotdir': plotdir,
                   'field': 'f(HR_q_total_field_on_w)_r',
           'data_field_list': dir_qt,
           'data_cl_list': dir_contour
           }

print('HjRj_qt options done')

apf.C_values(**LijMij_options)
print('LijMij fn done')

apf.C_values(**HjRj_th_options)
print('HjRj_th fn done')

apf.C_values(**HjRj_qt_options)
print('HjRj_qt fn done')
