import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['14400'] # ,'12600', '16200', '18000'
in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
model_res_list = ['0020_g0800']



outdir_og = '/work/scratch-pw/apower/'
outdir = outdir_og + '20m_gauss_dyn' +'/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40, 80]) # ([160, 320, 640]) #dont forget change start timr

start=0
#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'w'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,

        'aliases': {'u': 'f(u_on_p)_r',
                    'v': 'f(v_on_p)_r',
                    'w': 'f(w_on_p)_r',
                    'th': 'f(th_on_p)_r',
                    'q_total': 'f(q_total_on_p)_r'
                    }

          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
                dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options, \
                            opgrid, start_point=start, ref_file = None)
