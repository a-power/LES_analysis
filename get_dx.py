import numpy as np
import re

data_test = '/storage/silver/scenario/si818415/altered_MONC/400m/arm_18000.nc'

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