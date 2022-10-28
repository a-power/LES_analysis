import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['48600'] # ,'21600', '14400'
in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
model_res_list = ['0020_g0800']

start=0

outdir_og = '/work/scratch-pw/apower/'
outdir = outdir_og + '20m_gauss_dyn_13hrs' +'/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40, 80]) #change start
#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'w'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0
          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
                dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options, \
                            opgrid, start_point=start, ref_file = None)
