import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import argparse
import dynamic_functions as df

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=str, default='14400')
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--start_in', type=int, default=0)
parser.add_argument('--start_filt', type=int, default=0)
parser.add_argument('--n_filts', type=int, default=6)

args = parser.parse_args()
case_in = args.case
set_time = [ args.times ]
start = args.start_in
filters_start = args.start_filt
how_many_filters = args.n_filts #eg 6 = 0->5: ga00.nc -> ga05.nc (1 for ga00.nc)


opgrid = 'p'

#set_time = ['10800', '14400', '18000', '21600', '25200'] # '3600', '7200',

filter_name = 'gaussian'  # "wave_cutoff"
#Sigma = hat(Delta)/2


if case_in=='BOMEX':
    in_dir = '/work/scratch-pw3/apower/BOMEX/first_filt/BOMEX_m'
    model_res_list = ['0020_g0800']
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
                            'th_v': [f'f(th_v_on_{opgrid})_r'],
                            'q_cloud_liquid_mass': [f'f(q_cloud_liquid_mass_on_{opgrid})_r']
                            }

                  }

elif case_in=='ARM':
    in_dir = '/work/scratch-pw3/apower/ARM/first_filt/'
    outdir = '/work/scratch-pw3/apower/ARM/second_filt/'
    model_res_list = [None]
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
    model_res_list = [None]
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


if start == 0:
        sigma_list = np.array([df.sigma_2(4, dx)])
elif start == 1:
        sigma_list = np.array([df.sigma_2(4, dx), df.sigma_2(8, dx)])
else:
        print('need to set up the sigma list for start = ', start)

################################


os.makedirs(outdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"

#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'p'

if start==0:
    for j in range(len(set_time)):
            for i, model_res in enumerate(model_res_list):
                for k in range(how_many_filters - filters_start):
                    dy_s.run_dyn_on_filtered(model_res, set_time[j], filter_name, sigma_list*2**(k+filters_start), in_dir,
                                             outdir, options, opgrid, start_point=start, filtered_data =
                                             f'ga0{str(k+filters_start)}', ref_file = None, time_name='time', case=case_in)

elif start == 1:
    for j in range(len(set_time)):
        for i, model_res in enumerate(model_res_list):
            dy_s.run_dyn_on_filtered(model_res, set_time[j], filter_name, sigma_list * 2 ** (filters_start),
                                         in_dir, outdir, options, opgrid, start_point=start, filtered_data=
                                         f'ga0{str(filters_start)}', ref_file=None, time_name='time', case=case_in)

else:
    print('start needs to be 1 or 0')

#
# dy_s.run_dyn_on_filtered(None, '32400', filter_name, np.array([25]), in_dir,
#                          outdir, options, opgrid, start_point=start, filtered_data =
#                          f'ga01', ref_file = None, time_name='time', case=case_in)

# dy_s.run_dyn_on_filtered(None, '32400', filter_name, np.array([50]), in_dir,
#                          outdir, options, opgrid, start_point=start, filtered_data =
#                          f'ga02', ref_file = None, time_name='time', case=case_in)