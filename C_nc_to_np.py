import numpy as np
import os
import argparse
import monc_utils
import xarray as xr

monc_utils.global_config['output_precision'] = "float32"

parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, default='34200')
parser.add_argument('--case', type=str, default='ARM')
parser.add_argument('--start_in', type=int, default=0)

args = parser.parse_args()
set_time = args.time
case = args.case
start = args.start_in

scalars = ['s', '_th', '_qt']



if case=='BOMEX':
    homedir = '/work/scratch-pw3/apower/BOMEX/first_filt/LM/'
    dx=20
    res = '20_40'

elif case=='ARM':
    homedir = '/work/scratch-pw3/apower/ARM/first_filt/LM/diagnostics_3d_ts_32400_C'
    dx=25
    res = '100_200'


for i in scalars:

    C_data = xr.open_dataset(homedir+f'{i}_{res}.nc')
    if i == '_qt':
        i = '_q'

    C_sq = C_data[f'C{i}_sq_prof'].data[...]
    z = dx*C_sq[..., :]

    np.save('C{i}_sq_cond_{set_time}.npy', C_sq)
