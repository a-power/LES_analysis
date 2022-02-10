from netCDF4 import Dataset

dir_in = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files'
file_in = dir_in+"BOMEX_m0020_g0800_all_12600.nc"
data_in = Dataset(file_in, mode='r')
print(data_in.variables)