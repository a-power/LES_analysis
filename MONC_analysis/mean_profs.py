import numpy as np
import numpy.ma as ma
import xarray as xr
import matplotlib.pyplot as plt
import datetime
import os


def get_cloud_only(CT_or_CB_field, dist_from_surf_threas=1):

    mask_no_cloud = ma.masked_less_equal(CT_or_CB_field, dist_from_surf_threas)

    return mask_no_cloud




path_ARM25 = '/storage/silver/MONC_data/Alanna/ARM/MONC_out/25m/'
path_MONC_alt_HCs = '/storage/silver/scenario/si818415/altered_MONC/'
path_MONC_stand = '/storage/silver/scenario/si818415/og_monc/'

plotdir = '/home/users/si818415/phd/plots/MONC_alt/mean_profs/'
os.makedirs(plotdir, exist_ok = True)

list_timestamps = [18000, 25200, 32400, 39600]
#list_timestamps = [17400, 18000, 24600, 25200, 31800, 32400, 39000, 39600]
#list_timestamps = np.arange(600, 40200, 600)

var_list = ['wtheta_cn_mean', 'w_qt', 'ww_mean', 'wwsg_mean', 'total_cloud_fraction', 'tke_tendency', 'tkesg_mean',
            'resolved_buoyant_production', 'resolved_shear_production', 'resolved_turbulent_transport',
            'subgrid_buoyant_production', 'subgrid_shear_stress', 'subgrid_turbulent_transport']

zn = np.arange(0, 4410, 10)
zn_440 = np.arange(0, 4400, 10)
zn_40 = np.arange(0, 4410, 40)



colour_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
line_list = ['--', '--', '--', ':', ':', ':']
model_param = ['Stand', 'Stand', 'Stand', 'HCs', 'HCs', 'HCs'] #'HCs $\\widehat{\\bar{\\Delta}}'





for nv, var in enumerate(var_list):
    print(var)
    for nt, time in enumerate(list_timestamps):

        var_prof = np.zeros( (6, len(zn) ) )
        var_prof_40 = np.zeros((6, len(zn_40)))
        var_prof_440 = np.zeros((6, len(zn_440)))

        for n in range(6):
            print(n)
            if n <3:
                path_in = path_MONC_stand + f'{2 ** (n)}00m/'
                if n == 1:
                    if time > 30000:
                        var_prof[n, :] = np.nan
                    else:
                        filein = f'arm_{str(time)}.nc'
                        ds_in = xr.open_dataset(path_in + filein)
                        var_prof[n, :] = np.nan
                elif n==2:
                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof_40[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                else:
                    filein = f'arm_{str(time)}.nc'
                    ds_in = xr.open_dataset(path_in + filein)
                    var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis = 0)


            else:

                path_in = path_MONC_alt_HCs + f'{2 ** (n - 3)}00m/'
                filein = f'arm_{str(time)}.nc'

                ds_in = xr.open_dataset(path_in + filein)
                if n == 4:
                    var_prof_440[n, :] = np.mean(ds_in[f'{var}'].data, axis=0)
                else:
                    var_prof[n, :] = np.mean(ds_in[f'{var}'].data, axis = 0)



        plt.plot(figsize=(5, 8))
        for i in range(6):
            # plt.plot(list_timestamps, CT_mean_height_ts[i,:], colour_cycle[i%3], linestyle=line_list[i], label=model_param[i]+f'{2 ** ((i+1) % 8)}$\\Delta$')
            if i < 3:
                if i == 2:
                    plt.plot(var_prof_40[i, :], zn_40, colour_cycle[i % 3], linewidth=4,
                             label=f'{2 ** ((i + 2))}$\\Delta$')
                else:
                    plt.plot(var_prof[i, :], zn, colour_cycle[i % 3], linewidth=4,
                         label=f'{2 ** ((i + 2))}$\\Delta$')
            else:
                if i == 4:
                    plt.plot(var_prof_440[i, :], zn_440, colour_cycle[i % 3], linestyle=line_list[i],
                         marker='*')
                else:
                    plt.plot(var_prof[i, :], zn, colour_cycle[i % 3], linestyle=line_list[i],
                         marker='*')
        plt.tight_layout(pad=0.5)
        plt.gcf().set_size_inches(4, 9)
        plt.legend(fontsize=13, loc='upper right')

        bottom, top = plt.ylim()
        # plt.ylim(bottom=0, top=4000)

        # og_xtic = plt.xticks()
        # print(og_xtic)
        #
        # # time_label_temp = "%.2f"%(05.50 + og_xtic[0]/(60*60))
        # # time_label_temp_min = (( np.round(05.50 + og_xtic[0]/(60*60), 2 ) - "%.2f"%(05.50 + og_xtic[0]/(60*60)) )*60 )/100
        # # time_label = np.round(time_label_temp + time_label_temp_min, 2)
        #
        # time_label = []
        #
        # for i in range(len(og_xtic[0])):
        #     time_label.append(datetime.timedelta(seconds=og_xtic[0][i] + 19800))
        #
        # plt.xticks(og_xtic[0], time_label)

        plt.xlabel(f'{var}', fontsize=14)
        plt.ylabel('z (m)', fontsize=14)

        plt.tight_layout()

        plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_stand.png', bbox_inches='tight')
        plt.savefig(plotdir + f'ARM_{var}_{time}_mean_prof_HCs_vs_stand.pdf', bbox_inches='tight')
        plt.close()