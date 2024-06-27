import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt


def get_cloud_only(CT_or_CB_field, dist_from_surf_threas=1):

    mask_no_cloud = ma.masked_less_equal(CT_or_CB_field, dist_from_surf_threas)

    return mask_no_cloud




path_ARM25 = '/work/scratch-pw3/apower/ARM/MONC_out/25m/'
path_MONC_alt = '/storage/silver/scenario/si818415/altered_MONC/'

plotdir = '/home/users/si818415/phd/plots/MONC_alt/'

list_timestamps = [17400, 18000, 24600, 25200, 31800, 32400, 39000, 39600]

#np.ndarray.tolist( np.arange(17400, 40000, 600) )



CB_mean_height_ts = np.zeros(len(list_timestamps))
CT_mean_height_ts = np.zeros(len(list_timestamps))

for nt, time in enumerate(list_timestamps):

    filein = f'arm_3d_{str(time)}.nc'

    ds_in = xr.open_dataset(path_MONC_alt+filein)
    CB_field = ds_in['clbas'].data
    CT_field = ds_in['cltop'].data

    CB_cloud_only = get_cloud_only(CB_field)
    CT_cloud_only = get_cloud_only(CT_field)

    CB_mean_height_ts[nt] = np.mean(CB_cloud_only)
    CT_mean_height_ts[nt] = np.mean(CT_cloud_only)




fig = plt.plot(figsize=(10, 5))
plt.tight_layout(pad=0.5)

plt.plot(list_timestamps, CB_mean_height_ts, 'k')
plt.plot(list_timestamps, CT_mean_height_ts, 'k', label='HCs $\\widehat{\\bar{\\Delta}} = 200$m')

plt.legend(fontsize=13, loc='upper right')


bottom, top = plt.ylim()
plt.ylim(bottom=0, top = 2200)

og_xtic = plt.xticks()
print(og_xtic)

time_label_temp = np.round(05.50 + og_xtic[0]/(60*60), 2 )
time_label_temp_min = (( np.round(05.50 + og_xtic[0]/(60*60), 2 ) - np.round(05.50 + og_xtic[0]/(60*60), 0) )*60 )/100
time_label = time_label_temp + time_label_temp_min

plt.xticks(og_xtic[0], time_label)

plt.xlabel('Local time (hh.mm)', fontsize=14)
plt.ylabel('z (m)', fontsize=14)
plt.title('Cloud top and base height', fontsize=14)

plt.tight_layout()

plt.savefig(plotdir+f'ARM_cloud_top_and_base_ts.png', bbox_inches='tight')
plt.savefig(plotdir + f'ARM_cloud_top_and_base_ts.pdf', bbox_inches='tight')
plt.close()