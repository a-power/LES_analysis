import numpy as np
import os
import argparse
import monc_utils
import xarray as xr

monc_utils.global_config['output_precision'] = "float32"

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=0)
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--b', type=int, default=0)

args = parser.parse_args()
t_in = args.t
case = args.case
beta=args.b

th_type = 'th_v'
q_type = 'qt'
folder = th_type

numer = ['LM', f'HR_{th_type}', f'HR_{q_type}']
denom = ['MM', f'RR_{th_type}', f'RR_{q_type}']
partitions = ['prof', 'env_prof', 'cloud_prof', 'w_prof', 'w_th_v_prof']

filt_num = 2

data_dir = f'/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/data/{folder}/{case}/'
os.makedirs(data_dir, exist_ok = True)

if case=='BOMEX':
    dx=20
    zn_set = np.arange(0, 3020, 20)
    time = '14400'
    if filt_num == 1:
        homedir = f'/work/scratch-pw3/apower/BOMEX/first_filt/LM/{folder}/BOMEX_m0020_g0800_all_14400_C_cond_profs_'
        res = ['20_40', '20_80', '20_160', '20_320', '20_640', '20_1280']
        #res = ['20_40', '40_80', '80_160', '160_320', '320_640', '640_1280']
    elif filt_num == 2:
        homedir = f'/work/scratch-pw3/apower/BOMEX/second_filt/LM/{folder}/BOMEX_m0020_g0800_all_14400_C_cond_profs_'
        if beta == 0:
            res = ['40_80_0', '80_160_0', '160_320_0', '320_640_0', '640_1280_0', '1280_2560_0']
        else:
            res = ['40_160_1']
        #res = ['40_80', '80_160', '160_320', '320_640', '640_1280', '1280_2560']

elif case=='ARM':
    dx=25
    zn_set = np.arange(0, 4410, 10)
    times_analysed = ['18000', '25200', '32400', '39600']
    time = times_analysed[t_in]
    if filt_num == 1:
        homedir = f'/work/scratch-pw3/apower/ARM/first_filt/LM/{folder}/diagnostics_3d_ts_{time}_C_cond_profs_'
        res = ['2D', '4D', '8D', '16D', '32D', '64D']
        #res = ['25_50', '50_100', '100_200']
    elif filt_num == 2:
        homedir = f'/work/scratch-pw3/apower/ARM/second_filt/LM/{folder}/diagnostics_3d_ts_{time}_C_cond_profs_'
        if beta == 0:
            res = ['50_100_0', '100_200_0', '200_400_0', '400_800_0', '800_1600_0', '1600_3200_0']
        else:
            res = ['50_200_1']
        #res = ['50_100', '100_200', '200_400']



C_sq =  np.zeros( (len(numer), len(partitions), len(res), len(zn_set)) )

for r_ind, r in enumerate(res):
    C_data = xr.open_dataset(homedir + f'{r}.nc')
    for p_ind, p in enumerate(partitions):
        for s in range(len(denom)):

            LM = C_data[f'{numer[s]}_{p}'].data[...]
            MM = C_data[f'{denom[s]}_{p}'].data[...]
            C_sq[s, p_ind, r_ind, :] = LM / MM


np.save(data_dir+f'C_sq_cond_{time}_{beta}.npy', C_sq)

            # if s == '_q':
            #     s = '_qt'
            # C_data = xr.open_dataset(homedir+f'{s}_{r}.nc')
            # if s == '_qt':
            #     s = '_q'

            # C_sq = C_data[f'C{i}_sq_prof'].data[...]
            # z = dx*C_sq[..., :]
