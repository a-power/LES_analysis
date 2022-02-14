import matplotlib.pyplot as plt
import time_av_dynamic as tdy

set_time = '14400'
indir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
model_res_list = ['0020_g0800']
outdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
odir = outdir + '20m_gauss_dyn' +'/'

os.makedirs(odir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40])
opgrid = 'w'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
          }

