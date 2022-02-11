import numpy as np
import time_av_profiles as avp
import matplotlib.pyplot as plt

set_time = '14400'

mydir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m'

model_res_list = ['0020_g0800', '0025_g0640', '0040_g0400', '0080_g0200', '0160_g0100', '0320_g0050']
model_res_list_int = np.array([20, 25, 40, 80, 160, 320])

myvars = ['ww_mean']
z_i = 1000
z = avp.time_av_prof(myvars, model_res_list, set_time, mydir)
print("retrieved data and time averaged complete")

for i, var in enumerate(myvars):
    print('plotting', var, "number of vars = ", i)

    figs = plt.figure(figsize=(6, 6))
    plt.ylabel(r'z')
    plt.xlabel(f'{var}')

    for m, res in enumerate(model_res_list_int):
        var_plot = np.load(f'files/{res}_{var}_time_av.npy')
        print(var_plot)
        plt.plot(var_plot, z/z_i, label=f'$\\Delta x$ = {str(res)} m')
    plt.legend(fontsize=12)
    figs.savefig(f'plots/{var}_{set_time}_time_av_profile.png')
    plt.show()
    print('end fig')

