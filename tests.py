import multi_times_plot_final_C_and_max_vs_D as zc

file_path = '/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_ts_18000.nc'

z_ML, z_CL = zc.calc_z_ML_and_CL(file_path, time_stamp=-1)

print('z_ML = ', z_ML, 'z_CL = ', z_CL)