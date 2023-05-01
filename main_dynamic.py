import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os

case='ARM'
set_time = ['3600', '7200', '10800', '14400', '18000', '21600', '25200'] # ,'12600', '16200', '18000'
opgrid = 'p'

if case=='BOMEX':
        in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
        model_res_list = ['0020_g0800']
        outdir_og = '/work/scratch-pw3/apower/'
        outdir = outdir_og + f'20m_gauss_dyn/on_{opgrid}_grid/'
        plotdir = outdir_og + 'plots/dyn/'
elif case=='ARM':
        in_dir = '/work/scratch-pw3/apower/ARM/'
        outdir = in_dir + f'on_{opgrid}_grid/'
        plotdir = outdir + 'plots/dyn/'


os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"
sigma_list = np.array([20, 40, 80, 160, 320, 640]) #dont forget CHANGE start time if youre short-serial filtering
# #([20, 40, 80] ([160, 320, 640])

start=0

#Sigma = hat(Delta)/2

options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'override': True,
        'dx': 25.0,
        'dy': 25.0,
        'domain' : 19.2
          }


options_BOMEX = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'override': True,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0
          }

for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
                dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options, \
                            opgrid, start_point=start, ref_file = None)