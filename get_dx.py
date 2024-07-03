import numpy as np
import re
import xarray as xr

data_test_path = '/storage/silver/scenario/si818415/altered_MONC/400m/arm_18000.nc'
data_test = xr.open_dataset(data_test_path)

def _bytarr_to_dict(d):

    # Converted for xarray use

    while len(np.shape(d))>2:
        d = d[0]
    res = {}
    for i in range(np.shape(d)[0]):
        opt = d[i,0].decode('utf-8')
        val = d[i,1].decode('utf-8')

        res[opt] = val
    return res

def options_database(source_dataset):

    if 'options_database' in source_dataset.variables:
        options_database = _bytarr_to_dict(
            source_dataset['options_database'].values)
    else:
        options_database = None
    return options_database

od = options_database(data_test)
attrs = data_test.attrs

# 1st priority: pull from options_database, if present
if type(od) is dict:
    dx = float(od['dxx'])
    dy = float(od['dyy'])
# 2nd priority: pull from dataset attributes
elif ('dx' in attrs and 'dy' in attrs):
    dx = attrs['dx']
    dy = attrs['dy']

print('dx = ', dx)