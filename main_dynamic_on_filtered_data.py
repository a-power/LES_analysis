import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import argparse
import monc_utils
import dynamic_functions as df
import sys
from loguru import logger

monc_utils.global_config['output_precision'] = "float32"

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





parser = argparse.ArgumentParser()
parser.add_argument('--times', type=str, default='14400')
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--filt1', type=int, default=0)
parser.add_argument('--beta_filt', type=int, default=0)

args = parser.parse_args()
case_in = args.case
set_time = args.times
first_filt_res = args.filt1
beta = args.beta_filt


opgrid = 'p'

#set_time = ['10800', '14400', '18000', '21600', '25200'] # '3600', '7200',

filter_name = 'gaussian'  # "wave_cutoff"
#Sigma = hat(Delta)/2



if case_in=='BOMEX':
    in_dir = '/work/scratch-pw3/apower/BOMEX/first_filt/BOMEX_m'
    model_res = '0020_g0800'
    outdir = '/work/scratch-pw3/apower/BOMEX/second_filt/'
    dx=20.0
    #time = 14400

    options = {
                'FFT_type': 'RFFT',
                'save_all': 'Yes',
                'override': True,
                'th_ref': 0.0,
                'dx': 20.0,
                'dy': 20.0,
                'domain' : 16.0,

                'aliases': {'u': [f'f(u_on_{opgrid})_r'],
                            'v': [f'f(v_on_{opgrid})_r'],
                            'w': [f'f(w_on_{opgrid})_r'],
                            'th': [f'f(th_on_{opgrid})_r'],
                            'q_total': [f'f(q_total_on_{opgrid})_r'],
                            'q_vapour': [f'f(q_vapour_on_{opgrid})_r'],
                            'buoyancy': [f'f(buoyancy_on_{opgrid})_r'],
                            'th_v': [f'f(th_v_on_{opgrid})_r'],
                            'q_cloud_liquid_mass': [f'f(q_cloud_liquid_mass_on_{opgrid})_r']
                            }

                  }

elif case_in=='ARM':
    in_dir = '/work/scratch-pw3/apower/ARM/first_filt/'
    outdir = '/work/scratch-pw3/apower/ARM/second_filt/'
    model_res = None
    dx=25

    options = {
                'FFT_type': 'RFFT',
                'save_all': 'Yes',
                'override': True,
                'th_ref': 0.0,
                'dx': 25.0,
                'dy': 25.0,
                'domain' : 19.2,

                'aliases': {'u': [f'f(u_on_{opgrid})_r'],
                            'v': [f'f(v_on_{opgrid})_r'],
                            'w': [f'f(w_on_{opgrid})_r'],
                            'th': [f'f(th_on_{opgrid})_r'],
                            'q_total': [f'f(q_total_on_{opgrid})_r'],
                            'q_cloud_liquid_mass': [f'f(q_cloud_liquid_mass_on_{opgrid})_r']
                            }

      }

elif case_in=='dry':
    in_dir = f'/storage/silver/greybls/si818415/dry_CBL/'
    outdir = in_dir + 'second_filt/'
    model_res = None
    dx=20
    #time = 13800

    options = {
                'FFT_type': 'RFFT',
                'save_all': 'Yes',
                'override': True,
                'th_ref': 0.0,
                'dx': 20.0,
                'dy': 20.0,
                'domain' : 4.8,

                'aliases': {'u': [f'f(u_on_{opgrid})_r'],
                            'v': [f'f(v_on_{opgrid})_r'],
                            'w': [f'f(w_on_{opgrid})_r'],
                            'th': [f'f(th_on_{opgrid})_r']
                            }
                }

else:
    print(case_in, ": case isn't coded for yet")



if beta == 0:
        sigma_list = np.array([df.sigma_2(2**(first_filt_res+2), dx)])
elif beta == 1:
        sigma_list = np.array([df.sigma_2(2**(first_filt_res+2), dx), df.sigma_2(2**(first_filt_res+3), dx)])
else:
        print('need to set up the sigma list for first_filt_res = ', first_filt_res)

################################


os.makedirs(outdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"

#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'p'



dy_s.run_dyn_on_filtered(model_res, set_time, filter_name, sigma_list, in_dir, outdir, options,
                        opgrid, filtered_data = f'ga0{str(first_filt_res)}', ref_file = None,
                        time_name='time', case=case_in)
