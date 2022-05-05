import time_av_dynamic as t_dy
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/update_subfilt/'
path20f = '/work/scratch-pw/apower/20m_gauss_dyn_update_subfilt/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
outdir = outdir_og + '20m_update_subfilt' + '/'

os.makedirs(outdir, exist_ok = True)
os.makedirs(plotdir, exist_ok = True)

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')


dataset_name2 = path20f+file20+'profiles_2D.nc'
dataset_name4 = path20f+file20+'profiles_4D.nc'

Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d, Cs_sq_field_2d, LM_field_2d, MM_field_2d = \
    t_dy.Cs(data_2D, dx=20, dx_hat=40, ingrid = mygrid, t_in=0, save_all=2)

print(Cs_prof_sq_2d, Cs_prof_2d, LM_prof_2d, MM_prof_2d, Cs_sq_field_2d, LM_field_2d, MM_field_2d)

ds_2 = xr.Dataset()#coords =
                        #{'z':ds.coords['z']})

# Ensure bool, dict, and None items can be stored
# atts_out = {**atts, **subfilter.global_config, **options}
# for inc in atts_out:
#     if isinstance(atts_out[inc], (dict, bool, type(None))):
#         atts_out[inc] = str(atts_out[inc])
# derived_dataset.attrs = atts_out
ds_2.to_netcdf(dataset_name2, mode='w')
#ds_2 = xr.open_dataset(data_2D, mode='a', chunks='auto')
ds_in2 = {'file':dataset_name2, 'ds': ds_2}

save_field(ds_in2, Cs_prof_sq_2d)
save_field(ds_in2, Cs_prof_2d)
save_field(ds_in2, LM_prof_2d)
save_field(ds_in2, MM_prof_2d)
save_field(ds_in2, Cs_sq_field_2d)
save_field(ds_in2, LM_field_2d)
save_field(ds_in2, MM_field_2d)

Cs_prof_sq_2d = None        #free memory
Cs_prof_2d = None           #free memory
LM_prof_2d = None           #free memory
MM_prof_2d = None           #free memory
Cs_sq_field_2d = None       #free memory
LM_field_2d = None          #free memory
MM_field_2d = None          #free memory

ds_2.close()


Cs_prof_sq_4d, Cs_prof_4d, LM_prof_4d, MM_prof_4d, Cs_sq_field_4d, LM_field_4d, MM_field_4d = \
    t_dy.Cs(data_4D, dx=20, dx_hat=80, ingrid = mygrid, t_in=0, save_all=2)

ds_4 = xr.Dataset()
ds_4.to_netcdf(dataset_name4, mode='w')
ds_in4 = {'file':dataset_name4, 'ds': ds_4}

save_field(ds_in4, Cs_prof_sq_4d)
save_field(ds_in4, Cs_prof_4d)
save_field(ds_in4, LM_prof_4d)
save_field(ds_in4, MM_prof_4d)
save_field(ds_in4, Cs_sq_field_4d)
save_field(ds_in4, LM_field_4d)
save_field(ds_in4, MM_field_4d)

ds_4.close()

#########################plots#########################
#
# #if times_2D.all() == times_4D.all():
#
# z = np.arange(0,3020,20)
#
#
# y = 200
# levels1 = np.linspace(0, 0.4, 6)
#
#
#
# fig = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z / 500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z / 500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_av, z, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_av, z, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('z (m)')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_t_av_line_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig2 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_t_av_line_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig3 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z / 500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z / 500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.ylim(0, 2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i$')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_t_av_line_scaled_zoomed_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig4 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# #plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# #plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# #plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# #plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_t_av_2D_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig5 = plt.figure(figsize=(10, 8))
# #plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# #plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# #plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# #plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_t_av_4D_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig6 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# # plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# # plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_2D_4D_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
#
# fig7 = plt.figure(figsize=(10, 8))
# plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
# plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
# # plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
# # plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
# # plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
# # plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
# # plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
# # plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
# plt.xlim(-0.01, 0.2)
# plt.title(f'Cs averaged over times: {times}')
# plt.ylabel('$z/z_i')
# plt.xlabel('$ C_{s} $', fontsize=14)
# plt.legend(fontsize=16, loc='upper right')
# plt.savefig(plotdir + "Cs_profiles_2D_4D_t0_scaled_" + str(av_type) + ".png", pad_inches=0)
# plt.clf()
# plt.cla()
# plt.close()
# # else:
#   print('times data = ', times)