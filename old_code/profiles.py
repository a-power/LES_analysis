import os
import netCDF4
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy import integrate
import scipy.stats as sci
import seaborn as sns
from importlib import reload
import functions as f


file5m = '/storage/silver/scenario/si818415/phd/5mLES/runs2.nc'
data5m = Dataset(file5m, mode='r')
my_w_th = data5m.variables['w_th'][:,:,:]


Q_star = 241/(1*1004) # 0.24
w_star = 2


av_r_wth = np.zeros((len(s_w_th_filt[:,0,0,0]), len(s_w_th_filt[0,0,0,:])))

for it in range(len(s_w_th_filt[:,0,0,0])):
    av_r_wth[it,:] = f.mean_prof(s_w_th_filt[it,:,:,:])


plt.figure(figsize=(6,8))

plt.plot(w_th_run5/Q_star, zn/z_i[0], linewidth=2, label = "5m grid spacing")
plt.plot(w_th_run20/Q_star, zn1/z_i[1], label = "20 m grid spacing")

plt.plot(w_th_run25/Q_star, zn1/z_i[2], label = "25 m grid spacing")
#plt.plot(w_th_run25/Q_star, zn1/z_i[2], "--", color="tab:green")

plt.plot(w_th_run50/Q_star, zn1/z_i[3], label = "50 m grid spacing")
plt.plot(w_th_run100/Q_star, zn1/z_i[4], label = "100 m grid spacing")
plt.plot(w_th_run200/Q_star, zn1/z_i[5], label = "200 m grid spacing")
plt.plot(w_th_run400/Q_star, zn1/z_i[6], label = "400 m grid spacing")
plt.plot(w_th_run800/Q_star, zn1/z_i[7], label = "800 m grid spacing")
plt.legend(fontsize=14)
plt.xlabel("$w' \\theta '$/Q$_*$", fontsize=16)
plt.ylabel("$z/z_i$", fontsize=16)
plt.savefig(plot_dir+"runs_profile.png", pad_inches=0)




var_err = np.zeros(len(res))

var_err[0] = np.std(w_run5[:,:,100])/np.sqrt(len(w_run5[:,:,100]))
var_err[1] = np.std(w_run20[:,:,25])/np.sqrt(len(w_run20[:,:,25]))
var_err[2] = np.std(w_run25a[0,:,:,25])/np.sqrt(len(w_run25a[0,:,:,25]))
var_err[3] = np.std(w_run50[:,:,25])/np.sqrt(len(w_run50[:,:,25]))
var_err[4] = np.std(w_run100[:,:,25])/np.sqrt(len(w_run100[:,:,25]))
var_err[5] = np.std(w_run200[:,:,25])/np.sqrt(len(w_run200[:,:,25]))
var_err[6] = np.std(w_run400[:,:,25])/np.sqrt(len(w_run400[:,:,25]))
var_err[7] = np.std(w_run800[:,:,25])/np.sqrt(len(w_run800[:,:,25]))