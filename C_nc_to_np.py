import numpy as np
import os
import argparse
import monc_utils
import xarray as xr

monc_utils.global_config['output_precision'] = "float32"

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, default='14400')
parser.add_argument('--case', type=str, default='BOMEX')
parser.add_argument('--start_in', type=int, default=0)
parser.add_argument('--b', type=int, default=0)

args = parser.parse_args()
set_time = args.time
case = args.case
start = args.start_in
beta=args.b

scalars = ['s', '_th', '_qt']

filt_num = 2



if case=='BOMEX':
    dx=20
    if filt_num == 1:
        homedir = '/work/scratch-pw3/apower/BOMEX/first_filt/LM/BOMEX_m0020_g0800_all_14400_C'
        res = ['20_40', '40_80', '80_160', '160_320', '320_640', '640_1280']
    elif filt_num == 2:
        homedir = '/work/scratch-pw3/apower/BOMEX/second_filt/LM/BOMEX_m0020_g0800_all_14400_C'
        res = ['40_80', '80_160', '160_320', '320_640', '640_1280', '1280_2560']

elif case=='ARM':
    dx=25
    if filt_num == 1:
        homedir = '/work/scratch-pw3/apower/ARM/first_filt/LM/diagnostics_3d_ts_32400_C'
        res = ['25_50', '50_100', '100_200']
    elif filt_num == 2:
        homedir = '/work/scratch-pw3/apower/ARM/second_filt/LM/diagnostics_3d_ts_32400_C'
        res = ['50_100', '100_200', '200_400']

for j in res:
    for i in scalars:
        if i == '_q':
            i = '_qt'
        C_data = xr.open_dataset(homedir+f'{i}_{j}.nc')
        if i == '_qt':
            i = '_q'

        C_sq = C_data[f'C{i}_sq_prof'].data[...]
        z = dx*C_sq[..., :]

        np.save(f'C{i}_sq_cond_{set_time}_{j}.npy', C_sq)
