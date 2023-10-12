import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=str, default='14400')
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--start_in', type=int, default=0)

args = parser.parse_args()
set_time = [ args.times ]
case = args.case
start = args.start_in


print('start type = ', type(start))
if case=='ARM':
    MY_dx = 25
elif case=='BOMEX':
    MY_dx = 20
else:
    print('case not recognised, need to input dx for this case')


opgrid = 'p'

#set_time = ['10800', '14400', '18000', '21600', '25200'] # '3600', '7200',

filter_name = 'gaussian'  # "wave_cutoff"
#Sigma = hat(Delta)/2

###################################
#                                 #
#   NEEDS TO CHANGE BASED ON DX   #
#                                 #
###################################

if start == 0:
    sigma_list = np.array([MY_dx]) #, 2*MY_dx, 4*MY_dx, 8*MY_dx, 16*MY_dx, 32*MY_dx])
elif start == 1:
    sigma_list = np.array([2*MY_dx]) #, 4*MY_dx, 8*MY_dx, 16*MY_dx, 32*MY_dx])
elif start == 2:
    sigma_list = np.array([4*MY_dx]) #, 8*MY_dx, 16*MY_dx, 32*MY_dx])
elif start == 3:
    sigma_list = np.array([8*MY_dx]) #, 16*MY_dx, 32*MY_dx])
elif start == 4:
    sigma_list = np.array([16*MY_dx]) #, 32*MY_dx])
elif start == 5:
    sigma_list = np.array([32*MY_dx])
else:
    print('need to set up the sigma list for start = ', start)
# #([20, 40, 80] ([160, 320, 640])



if case=='BOMEX':
    in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
    model_res_list = ['0020_g0800']
    outdir_og = '/work/scratch-pw3/apower/'
    outdir = outdir_og + f'BOMEX/'
    plotdir = outdir_og + 'plots/dyn/'

elif case=='ARM':
    in_dir = '/work/scratch-pw3/apower/ARM/MONC_out/'
    outdir = '/work/scratch-pw3/apower/ARM/'
    plotdir = outdir + 'plots/dyn/'
    model_res_list = [None]


os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

options_ARM = {
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
        dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, options_ARM, \
                            opgrid, start_point=start, ref_file = None)