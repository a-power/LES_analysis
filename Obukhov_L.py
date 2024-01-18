import dynamic_script as dy_s #dot to get folder outside
import numpy as np
import os
import xarray as xr

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='dry')

args = parser.parse_args()
case_in = [ args.case ]

print(case_in)




if case_in=='BOMEX':
    in_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/'
    file = ['BOMEX_m0020_g0800_all_14400.nc']
    time_name_in = 'time_series_120_600'

elif case_in=='ARM':
    in_dir = '/work/scratch-pw3/apower/ARM/MONC_out/'
    file = ['diagnostics_ts_18000.nc', 'diagnostics_ts_25200.nc', 'diagnostics_ts_32400.nc', 'diagnostics_ts_39600.nc']
    time_name_in = 'time_series_60_60'

elif case_in=='dry':
    in_dir = '/storage/silver/MONC_data/Alanna/dry_CBL/MONC_runs/20m/'
    file = ['cbl_13200.nc']
    time_name_in = 'time_series_25_300'


def calc_L(file_in, time_name)
    ds_in = xr.open_dataset(file_in)
    time_data = ds_in[time_name]
    times = time_data.data
    nt = len(times)
    print(nt)

    # dx_in = float(opt['dx'])
    # domain_in = float(opt['domain'])
    # N = int((domain_in*(1000))/dx_in)

    uw_in = ds_in[uw_mean]
    uw = time_data.data

    vw_in = ds_in[vw_mean]
    vw = time_data.data

    wtheta_in = ds_in[wtheta_cn_mean]
    wtheta = time_data.data

    

