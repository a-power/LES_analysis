import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import mask_cloud_vs_env as clo
import matplotlib.ticker as ticker
import numpy.ma as ma
import dynamic_functions as dyn
import functools

print = functools.partial(print, flush=True)

np.seterr(divide='ignore')



print('test')


BOMEX_dir_contour = '/work/scratch-pw3/apower/BOMEX/second_filt/BOMEX_m0020_g0800_all_'

ARM_dir_contour = f'/work/scratch-pw3/apower/ARM/second_filt/diagnostics_3d_ts_'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/plots/cloud_percent/'
save_dir = '/work/scratch-pw3/apower/neg_Csq_values/'
#'/gws/nopw/j04/paracon_rdg/users/apower/on_p_grid/plots/distribs/'
os.makedirs(plotdir, exist_ok = True)
os.makedirs(save_dir, exist_ok = True)

cloud_field = f'f(q_cloud_liquid_mass_on_p)_r'

Deltas = ['4$\\Delta$', '8$\\Delta$', '16$\\Delta$', '32$\\Delta$', '64$\\Delta$', '128$\\Delta$']

times = ['14400', '18000', '25200', '32400', '39600']

B1 = 25
B2 = 75

A1 = [90, 110, 125, 140]
A2 = [105, 140, 185, 215]

c_b = [25, 90, 110, 125, 140]
c_t = [75, 105, 140, 185, 215]



def get_data_per_delta(dir_cloud, time):

    data_cl_list = []

    for i in range(6):
        data_cl = dir_cloud+f'{time}_gaussian_filter_ga0{i}_gaussian_filter_ga00.nc'

        data_cl_list.append(data_cl)

    return data_cl_list





def cloud_percentage(dataset_in, cl_bottom, cl_top, cloud_liquid_threshold=10**(-7), grid='p'):

    ds_in = xr.open_dataset(dataset_in)

    if f'f(q_cloud_liquid_mass_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(q_cloud_liquid_mass_on_{grid})_r']
    elif f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r' in ds_in:
        q_in = ds_in[f'f(f(q_cloud_liquid_mass_on_{grid})_r_on_{grid})_r']
    elif 'q_cloud_liquid_mass' in ds_in:
        q_in = ds_in['q_cloud_liquid_mass']


    q_cloud = q_in.data[-1,...]

    cl_depth = cl_top-cl_bottom

    num_points_in_horiz_domain = len(q_cloud[:,0,0])*len(q_cloud[0,:,0])
    numb_points_in_CL = num_points_in_horiz_domain*cl_depth

    if cloud_liquid_threshold == 0:
        masked_q_cloud = ma.masked_less_equal(q_cloud, cloud_liquid_threshold)  # masking lower values

    else:
        masked_q_cloud = ma.masked_less(q_cloud, cloud_liquid_threshold) #masking lower values

    num_cloud_points = ma.MaskedArray.count(masked_q_cloud[:, :, cl_bottom:cl_top])

    percentage_cloud = num_cloud_points/numb_points_in_CL


    return percentage_cloud


labels_title = ['BOMEX', 'ARM 10:30L', 'ARM 12:30L', 'ARM 14:30L', 'ARM 16:30L']


plt.figure(figsize=(4, 6))

for nt, t in enumerate(times):
    if t == '14400':
        dir_cloud = BOMEX_dir_contour
    else:
        dir_cloud = ARM_dir_contour

    data_cl_list_out = get_data_per_delta(dir_cloud, t)
    perc = np.zeros(len(data_cl_list_out))

    for i in range(len(data_cl_list_out)):

        perc[i] = cloud_percentage(data_cl_list_out[i], c_b[i], c_t[i])

    plt.plot(Deltas, perc, label=labels_title)

plt.legend()

# og_xtic = plt.xticks()
# plt.xticks(og_xtic[0],
#            np.round(np.linspace((0) * (20 / 480), (151) * (20 / 480), len(og_xtic[0])), 1))

# ax1.title(f"{labels_title[nt_in]}", fontsize=13)
plt.ylabel('Percentage of Cloud Cover', fontsize=16)
plt.xlabel("$\\widehat{\\bar{\\Delta}}$", fontsize=13)
plt.savefig(plotdir + f'percent_cloud_cover.pdf', bbox_inches='tight')