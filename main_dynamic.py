import matplotlib.pyplot as plt
import time_av_dynamic as tdy #dot to get folder outside
import numpy as np
import os

set_time = '14400'
in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
model_res_list = ['0020_g0800']
outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
outdir = outdir_og + '20m_gauss_dyn' +'/'

plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40])
opgrid = 'p'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
          }


for i, model_res in enumerate(model_res_list):
        tdy.time_av_dyn(model_res, set_time, filter_name, sigma_list, in_dir, outdir, options, opgrid, domain_in=16, ref_file = None)


