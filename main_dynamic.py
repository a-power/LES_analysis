import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import argparse
import monc_utils
import sys
from loguru import logger
# sys.stdout.write


monc_utils.global_config['output_precision'] = "float32"

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, default='32400')
parser.add_argument('--case', type=str, default='ARM')
parser.add_argument('--start_in', type=int, default=0)

args = parser.parse_args()
set_time = [ args.time ]
case = args.case
start = args.start_in



logger.remove()
logger.add(sys.stderr,

           format="<c>{time:HH:mm:ss.SS}</c>" \
 \
                  + " | <level>{level:<8}</level>" \
 \
                  + " | <green>{function:<22}</green> : {message}",

           colorize=True,

           level="ERROR")

logger.add(sys.stdout,

           format="<c>{time:HH:mm:ss.SS}</c>" \
 \
                  + " | <level>{level:<8}</level>" \
 \
                  + " | <green>{function:<22}</green> : {message}",

           colorize=True,

           level="INFO")


logger.enable("subfilter")
logger.enable("monc_utils")
logger.info("Logging 'INFO', 'ERROR', and higher messages only")


print(case)


print('start type = ', type(start))
if case=='ARM':
    MY_dx = 25
elif case=='BOMEX' or case == 'dry':
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

options_ARM = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'override': True,
        'dx': 25.0,
        'dy': 25.0,
        'domain' : 19.2
          }


options_BOMEX = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'override': True,
        'dx': 20.0,
        'dy': 20.0,
        'domain' : 16.0
          }

options_dry = {
        'FFT_type': 'RFFT',
        'th_ref': 300.0,
        'save_all': 'Yes',
        'override': True,
        'dx': 20.0,
        'dy': 20.0,
        'domain': 4.8
          }


if case=='BOMEX':
    in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'
    model_res_list = ['0020_g0800']
    outdir_og = '/work/scratch-pw3/apower/'
    outdir = outdir_og + f'BOMEX/C_th/'
    #plotdir = outdir_og + 'plots/dyn/'
    time_name_in = 'time_series_600_600'
    my_opt = options_BOMEX

elif case=='ARM':
    in_dir = '/work/scratch-pw3/apower/ARM/MONC_out/25m/'
    outdir = '/work/scratch-pw3/apower/ARM/C_th/'
    #plotdir = outdir + 'plots/dyn/'
    model_res_list = [None]
    time_name_in = 'time_series_600_600'
    my_opt = options_ARM

elif case=='dry':
    in_dir = '/storage/silver/MONC_data/Alanna/dry_CBL/MONC_runs/20m/'
    outdir = '/storage/silver/greybls/si818415/dry_CBL/'
    #plotdir = outdir + 'plots/'
    model_res_list = [None]
    time_name_in = 'time_series_300_300'
    vapour = False
    my_opt = options_dry


os.makedirs(outdir, exist_ok = True)
# os.makedirs(plotdir, exist_ok = True)



for j in range(len(set_time)):
    for i, model_res in enumerate(model_res_list):
        dy_s.run_dyn(model_res, set_time[j], filter_name, sigma_list, in_dir, outdir, my_opt, \
                            opgrid, start_point=start, time_name = time_name_in, vapour=False)