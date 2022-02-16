from netCDF4 import Dataset

dir_in = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
file_in = dir_in+"BOMEX_m0020_g0800_all_14400.nc"
data_in = Dataset(file_in, mode='r')
print('time 0', data_in.variables['time_series_120_600'][:], "time 1", data_in.variables['time_series_600_600'][:])
#print(data_in.variables)



