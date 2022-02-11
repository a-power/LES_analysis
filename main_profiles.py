import numpy as np
import xarray as xr
import functions as f
import matplotlib.pyplot as plt

time_in = '14400'

indir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'

model_res_list = ['0020_g0800', '0025_g0600', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
model_res_list_int = np.array([20, 25, 40, 80, 160, 320])

vars = ['ww_mean']
z_i = 1000

for i, var in enumerate(vars):

    figs = plt.figure(figsize=(6, 6))
    plt.ylabel(r'z')
    plt.xlabel(f'${var}$')

    for m, res in enumerate(model_res_list_int):
        var_plot = np.load(f'files/{res}_{var}_time_av.npy')
        plt.plot(var_plot, z/z_i, label=f'$\\Delta x$ = {str(res)} m')
    plt.legend(fontsize=12)
    figs.savefig(f'plots/{var}_{timein}_time_av_profile.png')
