import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['14400'] # , '21600', '48600'
in_dir = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/BOMEX_m'
model_res_list = ['0020_g0800']

#use_filtered_data = 'ga00' #set in loop #set to string if you want to filter a previously filtered dataset,
# and give 'ga00' or whatever, if want to filter LES data set to an int 0 or something

#dont forget to change dx and dy in options
start_point_filtering = 0 #for labelling output files: ie skiping ga00.nc and going straight to ga03.nc if set =3

outdir_og = '/work/scratch-pw2/apower/'
outdir = outdir_og + 'filtering_filtered' +'/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40])
how_many_filters = 6 #eg 6 = 0->5: ga00.nc -> ga05.nc
#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'p'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,

        'aliases': {'u': ['f(u_on_w)_r'],
                    'v': ['f(v_on_w)_r'],
                    'w': ['f(w_on_w)_r'],
                    'th': ['f(th_on_w)_r'],
                    'q_total_f': ['f(q_total_on_w)_r']
                    }

          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
            for k in range(how_many_filters):
                dy_s.run_dyn_on_filtered(model_res, set_time[j], filter_name, sigma_list*2**k, in_dir, outdir, options, \
                            opgrid, start_point=start_point_filtering, filtered_data = f'ga0{k}', ref_file = None, \
                                         time_name='time')
