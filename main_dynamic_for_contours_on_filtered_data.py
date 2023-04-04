import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os


set_time = ['14400'] # , '21600', '48600'
in_dir = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/BOMEX_m'
model_res_list = ['0020_g0800']

#use_filtered_data = 'ga00' #set in loop #set to string if you want to filter a previously filtered dataset,
# and give 'ga00' or whatever, if want to filter LES data set to an int 0 or something

#dont forget to change dx and dy in options
start_point_filtering = 0 #for labelling output files: ie skiping ga00.nc and going straight to ga03.nc if set =3

outdir_og = '/work/scratch-pw3/apower/'
outdir = outdir_og + '20m_gauss_dyn/on_p_grid/beta_filtered_filters/contours/'
plotdir = outdir_og+'plots/dyn/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20])
how_many_filters = 6 #eg 6 = 0->5: ga00.nc -> ga05.nc
filters_start=0
#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'p'

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'override': True,
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0,

        'aliases': {'u': [f'f(u_on_{opgrid})_r'],
                    'v': [f'f(v_on_{opgrid})_r'],
                    'w': [f'f(w_on_{opgrid})_r'],
                    'th': [f'f(th_on_{opgrid})_r'],
                    'q_total_f': [f'f(q_total_on_{opgrid})_r'],
                    'th_v': [f'f(th_v_on_{opgrid})_r'],
                    'q_cloud_liquid_mass': [f'f(q_cloud_liquid_mass_on_{opgrid})_r']
                    }

          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
            for k in range(how_many_filters-filters_start):
                dy_s.run_dyn_on_filtered_for_beta_contour(model_res, set_time[j], filter_name,
                                                          sigma_list*2**(k+filters_start), in_dir, outdir, options, \
                            opgrid, start_point=start_point_filtering, filtered_data = f'ga0{str(k+filters_start)}',
                                                          ref_file = None, \
                                         time_name='time')
