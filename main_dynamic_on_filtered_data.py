import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=str, default='32400')
parser.add_argument('--start_in', type=int, default=0)
parser.add_argument('--start_filt', type=int, default=0)
parser.add_argument('--n_filts', type=int, default=6)

args = parser.parse_args()
set_time = [ args.times ]
start = args.start_in
filters_start = args.start_filt
how_many_filters = args.n_filts #eg 6 = 0->5: ga00.nc -> ga05.nc

case_in='ARM'

opgrid = 'p'

#set_time = ['10800', '14400', '18000', '21600', '25200'] # '3600', '7200',

filter_name = 'gaussian'  # "wave_cutoff"
#Sigma = hat(Delta)/2


if case_in=='BOMEX':
    in_dir = f'/work/scratch-pw3/apower/20m_gauss_dyn/on_{opgrid}_grid/BOMEX_m'
    model_res_list = ['0020_g0800']
    outdir_og = '/work/scratch-pw3/apower/'
    outdir = outdir_og + f'/20m_gauss_dyn/on_{opgrid}_grid/filtering_filtered/'
    plotdir = outdir_og + 'plots/dyn/'
    dx=20.0

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

elif case_in=='ARM':
    in_dir = f'/work/scratch-pw3/apower/ARM/corrected_sigmas/'
    outdir = in_dir + 'filtering_filtered_check/'
    plotdir = outdir + 'plots/dyn/'
    model_res_list = [None]
    plotdir = outdir + 'plots/dyn/'
    dx=25

    options = {
                'FFT_type': 'RFFT',
                'save_all': 'Yes',
                'override': True,
                'th_ref': 300.0,
                'dx': 25.0,
                'dy': 25.0,
                'domain' : 19.2,

                'aliases': {'u': [f'f(u_on_{opgrid})_r'],
                            'v': [f'f(v_on_{opgrid})_r'],
                            'w': [f'f(w_on_{opgrid})_r'],
                            'th': [f'f(th_on_{opgrid})_r'],
                            'q_total_f': [f'f(q_total_on_{opgrid})_r'],
                            'th_v': [f'f(th_v_on_{opgrid})_r'],
                            'q_cloud_liquid_mass': [f'f(q_cloud_liquid_mass_on_{opgrid})_r']
                            }

      }

else:
    print(case_in, ": case isn't coded for yet")


if start == 0:
        sigma_list = np.array([dx, 2*dx])
elif start == 1:
        sigma_list = np.array([dx])
else:
        print('need to set up the sigma list for start = ', start)

################################


os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

filter_name = 'gaussian'  # "wave_cutoff"

#Note short serial queue on JASMIN times out after 3 filter scales
#Sigma = hat(Delta)/2

opgrid = 'p'

# if start==0:
#     for j in range(len(set_time)):
#             for i, model_res in enumerate(model_res_list):
#                 for k in range(how_many_filters - filters_start):
#                     dy_s.run_dyn_on_filtered(model_res, set_time[j], filter_name, sigma_list*2**(k+filters_start), in_dir,
#                                              outdir, options, opgrid, start_point=start, filtered_data =
#                                              f'ga0{str(k+filters_start)}', ref_file = None, time_name='time', case=case_in)
#
# elif start == 1:
#     for j in range(len(set_time)):
#         for i, model_res in enumerate(model_res_list):
#             dy_s.run_dyn_on_filtered(model_res, set_time[j], filter_name, sigma_list * 2 ** (filters_start),
#                                          in_dir, outdir, options, opgrid, start_point=start, filtered_data=
#                                          f'ga0{str(filters_start)}', ref_file=None, time_name='time', case=case_in)
#
# else:
#     print('start needs to be 1 or 0')


dy_s.run_dyn_on_filtered(None, '32400', filter_name, np.array([25]), in_dir,
                         outdir, options, opgrid, start_point=start, filtered_data =
                         f'ga01', ref_file = None, time_name='time', case=case_in)

dy_s.run_dyn_on_filtered(None, '32400', filter_name, np.array([50]), in_dir,
                         outdir, options, opgrid, start_point=start, filtered_data =
                         f'ga02', ref_file = None, time_name='time', case=case_in)