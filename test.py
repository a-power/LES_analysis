import numpy as np
import xarray as xr

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"
dataset_in = path20f+file20
delta = '2D'

data_s = xr.open_dataset(dataset_in + f'Cs_{delta}.nc')
data_th = xr.open_dataset(dataset_in + f'C_th_{delta}.nc')
data_qtot = xr.open_dataset(dataset_in + f'C_qt_{delta}.nc')

LijMij = data_s['LM_field'].data[...]
MijMij = data_s['MM_field'].data[...]

HjRj_th = data_th['HR_th_field'].data[...]
RjRj_th = data_th['RR_th_field'].data[...]

HjRj_qt = data_qtot['HR_q_total_field'].data[...]
RjRj_qt = data_qtot['RR_q_total_field'].data[...]

print('shape of LijMij is = ', np.shape(LijMij), 'shape of MijMij is = ', np.shape(MijMij), \
      'and the shape of HjRj_th is = ', np.shape(HjRj_th), 'and the shape of RjRj_th is = ', np.shape(RjRj_th),\
      'and the shape of HjRj_qt is = ', np.shape(HjRj_qt), 'and the shape of RjRj_qt is = ', np.shape(RjRj_qt))