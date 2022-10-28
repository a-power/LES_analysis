import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['14400'] # , '21600', '48600'
in_dir = '/work/scratch-pw/apower/20m_gauss_dyn/BOMEX_m'
model_res_list = ['0020_g0800']

use_filtered_data = 'ga00' #set to string if you want to filter a previously filtered dataset,
# and give 'ga00' or whatever, if want to filter LES data set to 0 or something

#dont forget to change dx and dy in options
start_point_filtering = 0 #for labelling output files: ie skiping ga00.nc and going straight to ga03.nc if set =3

outdir_og = '/work/scratch-pw/apower/'
outdir = outdir_og + '40m_gauss_dyn_filtering_filtered' +'/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([40, 80, 160,])  #([20, 40, 80]) ([160, 320, 640])
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

        'aliases': {'u': ['f(u_on_p)_r'],
                    'v': ['f(v_on_p)_r'],
                    'w': ['f(w_on_p)_r'],
                    'th': ['f(th_on_p)_r'],
                    'q_total': ['f(q_total_on_p)_r']
                    }

          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
                dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options, \
                            opgrid, start_point=start_point_filtering, filtered_data = use_filtered_data, ref_file = None)
