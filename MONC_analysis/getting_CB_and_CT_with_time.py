import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
import datetime


def get_cloud_only(CT_or_CB_field, dist_from_surf_threas=10):

    mask_no_cloud_temp = ma.masked_less_equal(CT_or_CB_field, dist_from_surf_threas)
    mask_no_cloud =np.ma.filled(mask_no_cloud_temp, np.nan)

    return mask_no_cloud




path_ARM25 = '/storage/silver/MONC_data/Alanna/ARM/MONC_out/25m/'
path_MONC_alt_HCs = '/storage/silver/scenario/si818415/altered_MONC/'
path_MONC_stand = '/storage/silver/scenario/si818415/altered_MONC/'

plotdir = '/home/users/si818415/phd/plots/MONC_alt/'

list_timestamps = [17400, 18000, 24600, 25200, 31800, 32400, 39000, 39600]

#np.ndarray.tolist( np.arange(17400, 40000, 600) )

colour_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
line_list = ['--', '--', '--', ':', ':', ':']
model_param = ['Stand', 'Stand', 'Stand', 'HCs', 'HCs', 'HCs'] #'HCs $\\widehat{\\bar{\\Delta}}'



CB_mean_height_ts = np.zeros( (6, len(list_timestamps)) )
CT_mean_height_ts = np.zeros( (6, len(list_timestamps)) )

for n in range(6):
    if n <3:
        path_in = path_MONC_stand + f'{2 ** n}00m/'
    else:
        path_in = path_MONC_alt_HCs + f'{2 ** (n % 3)}00m/'

    for nt, time in enumerate(list_timestamps):

        filein = f'arm_3d_{str(time)}.nc'

        ds_in = xr.open_dataset(path_in+filein)
        CB_field = ds_in['clbas'].data
        CT_field = ds_in['cltop'].data

        CB_cloud_only = get_cloud_only(CB_field)
        CT_cloud_only = get_cloud_only(CT_field)

        CB_mean_height_ts[n, nt] = np.nanpercentile(CB_cloud_only, 5)
        CT_mean_height_ts[n, nt] = np.nanpercentile(CT_cloud_only,95)




fig = plt.plot(figsize=(12, 4))
plt.tight_layout(pad=0.5)
for i in range(6):
    plt.plot(list_timestamps, CB_mean_height_ts[i,:], colour_cycle[i%3], line_list[i],)
    plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 3)}$\\Delta$')

plt.legend(fontsize=13, loc='upper right')


bottom, top = plt.ylim()
plt.ylim(bottom=0, top = 4000)

og_xtic = plt.xticks()
print(og_xtic)

# time_label_temp = "%.2f"%(05.50 + og_xtic[0]/(60*60))
# time_label_temp_min = (( np.round(05.50 + og_xtic[0]/(60*60), 2 ) - "%.2f"%(05.50 + og_xtic[0]/(60*60)) )*60 )/100
# time_label = np.round(time_label_temp + time_label_temp_min, 2)

time_label = []

for i in range(len(og_xtic[0])):
    time_label.append(datetime.timedelta(seconds=og_xtic[0][i] + 19800))

plt.xticks(og_xtic[0], time_label)

plt.xlabel('Local time (hh:mm:ss)', fontsize=14)
plt.ylabel('z (m)', fontsize=14)
plt.title('Cloud top and base height', fontsize=14)

plt.tight_layout()

plt.savefig(plotdir+f'ARM_cloud_top_and_base_ts.png', bbox_inches='tight')
plt.savefig(plotdir + f'ARM_cloud_top_and_base_ts.pdf', bbox_inches='tight')
plt.close()