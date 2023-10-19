from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import functions as f

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/ARM/corrected_sigma/spectra/'
os.makedirs(plotdir, exist_ok=True)

z_cl_r_ind_set = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
z_ml_r_ind_list = [ [15, 75], [15, 80], [15, 85], [15, 90] ]

z_level_mid_CL = (z_cl_r_ind_set[2][1] + z_cl_r_ind_set[2][0]) / 2
z_level_mid_ML = (z_ml_r_ind_list[2][1] + z_ml_r_ind_list[2][0]) / 2

z_set = z_level_mid_ML

z_ml_height = z_ml_r_ind_list[2][1] * 10

set_time = '32400'
set_z_level = z_level_mid_ML

homedir = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/smoothed_LM_HR_fields/'

mydir20 = '/work/scratch-pw3/apower/ARM/corrected_sigmas/'
myfile20 = [f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga00', f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga01',
           f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga02', f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga03',
            f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga04', f'diagnostics_3d_ts_{set_time}_gaussian_filter_ga05']

data0_20 = Dataset(str(mydir20)+myfile20[0]+'.nc', mode='r')
data1_20 = Dataset(str(mydir20)+myfile20[1]+'.nc', mode='r')
data2_20 = Dataset(str(mydir20)+myfile20[2]+'.nc', mode='r')
data3_20 = Dataset(str(mydir20)+myfile20[3]+'.nc', mode='r')
data4_20 = Dataset(str(mydir20)+myfile20[4]+'.nc', mode='r')
data5_20 = Dataset(str(mydir20)+myfile20[5]+'.nc', mode='r')

print('ga00.nc sigma = ', data0_20.getncattr('sigma'))
print('ga01.nc sigma = ', data1_20.getncattr('sigma'))
print('ga02.nc sigma = ', data2_20.getncattr('sigma'))
print('ga03.nc sigma = ', data3_20.getncattr('sigma'))
print('ga04.nc sigma = ', data4_20.getncattr('sigma'))
print('ga05.nc sigma = ', data5_20.getncattr('sigma'))

w_filt_1st = np.zeros((6,768,768,441))
w_filt_1st[0,:,:,:] = data0_20.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_1st[1,:,:,:] = data1_20.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_1st[2,:,:,:] = data2_20.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_1st[3,:,:,:] = data3_20.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_1st[4,:,:,:] = data4_20.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_1st[5,:,:,:] = data5_20.variables['f(w_on_p)_r'][0,:,:,:]



data0_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[0]+'_gaussian_filter_ga00.nc', mode='r')
data1_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[1]+'_gaussian_filter_ga00.nc', mode='r')
data2_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[2]+'_gaussian_filter_ga00.nc', mode='r')
data3_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[3]+'_gaussian_filter_ga00.nc', mode='r')
data4_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[4]+'_gaussian_filter_ga00.nc', mode='r')
data5_2nd = Dataset(str(mydir20)+'filtering_filtered/'+myfile20[5]+'_gaussian_filter_ga00.nc', mode='r')

w_filt_2nd = np.zeros((6,768,768,441))
w_filt_2nd[0,:,:,:] = data0_2nd.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_2nd[1,:,:,:] = data1_2nd.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_2nd[2,:,:,:] = data2_2nd.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_2nd[3,:,:,:] = data3_2nd.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_2nd[4,:,:,:] = data4_2nd.variables['f(w_on_p)_r'][0,:,:,:]
w_filt_2nd[5,:,:,:] = data5_2nd.variables['f(w_on_p)_r'][0,:,:,:]


dx = dy = 25 #metres

options_spec = {
           'spec_method': 'Durran',    # [Durran, ndimage] Use Durran method
                                        # (which actually also uses ndimage),
                                       #   or faster, less accurate ndimage method
           'spec_compensation': False, # With spec_method: 'durran', use Durran/Tanguay method
                                        # to compensate for systematic
                                       # noise in the annular summation (does not preserve energy)
           'spec_restrict': False       # With spec_method: 'durran', restrict the spec_2d
                                        # result to values below the Nyquist frequency.
             }


#w_spec_filt0k, w_kpo_filt0k = f.spectra_2d(w_filt20k[0,:,:,:], dx, dy, options_spec)
#print("sigma 25")
w_filt_1st_0, w_kpo_filt1g1 = f.spectra_2d(w_filt_1st[0,:,:,:], dx, dy, options_spec)
#print("sigma 50")
w_filt_1st_1, w_kpo_filt2g1 = f.spectra_2d(w_filt_1st[1,:,:,:], dx, dy, options_spec)
#print("w_filt_1st_sigma 100")
w_filt_1st_2, w_kpo_filt3g1 = f.spectra_2d(w_filt_1st[2,:,:,:], dx, dy, options_spec)
#print("sigma 200")
w_filt_1st_3, w_kpo_filt4g1 = f.spectra_2d(w_filt_1st[3,:,:,:], dx, dy, options_spec)
#print("sigma 400")
w_filt_1st_4, w_kpo_filt5g1 = f.spectra_2d(w_filt_1st[4,:,:,:], dx, dy, options_spec)
#print("sigma 522")
w_filt_1st_5, w_kpo_filt6g1 = f.spectra_2d(w_filt_1st[5,:,:,:], dx, dy, options_spec)
#print("sigma 522")


w_filt_2nd_0, w_kpo_filt1g2 = f.spectra_2d(w_filt_2nd[0,:,:,:], dx, dy, options_spec)
#print("sigma 50")
w_filt_2nd_1, w_kpo_filt2g2 = f.spectra_2d(w_filt_2nd[1,:,:,:], dx, dy, options_spec)
#print("w_filt_1st_sigma 100")
w_filt_2nd_2, w_kpo_filt3g2 = f.spectra_2d(w_filt_2nd[2,:,:,:], dx, dy, options_spec)
#print("sigma 200")
w_filt_2nd_3, w_kpo_filt4g2 = f.spectra_2d(w_filt_2nd[3,:,:,:], dx, dy, options_spec)
#print("sigma 400")
w_filt_2nd_4, w_kpo_filt5g2 = f.spectra_2d(w_filt_2nd[4,:,:,:], dx, dy, options_spec)
#print("sigma 522")
w_filt_2nd_5, w_kpo_filt6g2 = f.spectra_2d(w_filt_2nd[5,:,:,:], dx, dy, options_spec)




fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)

turb_slope_x = np.linspace(0.02,0.4,100)
turb_slope_y = turb_slope_x**(-5/3)

filt_slope_x = np.linspace(0.5,1,50)
filt_slope_y = filt_slope_x**(-11/2)




ax.loglog(w_kpo_filt1g1*z_ml_height/(2*np.pi), w_filt_1st_0[:,z_set], label="$\\sigma$ = 25m") #\\Delta
ax.loglog(w_kpo_filt2g1*z_ml_height/(2*np.pi), w_filt_1st_1[:,z_set], label="$\\sigma$ = 50m")
ax.loglog(w_kpo_filt3g1*z_ml_height/(2*np.pi), w_filt_1st_2[:,z_set], label="$\\sigma$ = 100m")
ax.loglog(w_kpo_filt4g1*z_ml_height/(2*np.pi), w_filt_1st_3[:,z_set], label="$\\sigma$ = 200m")
ax.loglog(w_kpo_filt5g1*z_ml_height/(2*np.pi), w_filt_1st_4[:,z_set], label="$\\sigma$ = 400m")
ax.loglog(w_kpo_filt6g1*z_ml_height/(2*np.pi), w_filt_1st_5[:,z_set], label="$\\sigma$ = 800m")
#ax.loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
#ax.text(10, 1.6, r'$k^{-5/3}$', fontsize=14)

ax.loglog(w_kpo_filt1g2*z_ml_height/(2*np.pi), w_filt_2nd_0[:,z_set], '--', label="$\\sigma$ = 25m, $\\sigma$ = 25m")
ax.loglog(w_kpo_filt2g2*z_ml_height/(2*np.pi), w_filt_2nd_1[:,z_set], '--', label="$\\sigma$ = 50m, $\\sigma$ = 25m")
ax.loglog(w_kpo_filt3g2*z_ml_height/(2*np.pi), w_filt_2nd_2[:,z_set], '--', label="$\\sigma$ = 100m, $\\sigma$ = 25m")
ax.loglog(w_kpo_filt4g2*z_ml_height/(2*np.pi), w_filt_2nd_3[:,z_set], '--', label="$\\sigma$ = 200m, $\\sigma$ = 25m")
ax.loglog(w_kpo_filt5g2*z_ml_height/(2*np.pi), w_filt_2nd_4[:,z_set], '--', label="$\\sigma$ = 400m, $\\sigma$ = 25m")
ax.loglog(w_kpo_filt6g2*z_ml_height/(2*np.pi), w_filt_2nd_5[:,z_set], '--', label="$\\sigma$ = 800m, $\\sigma$ = 25m")



#ax.loglog(w_kpo20*z_i/(2*np.pi), w_spec20[:,z_set[1]]/my_w_star, label="20m res")
#ax.loglog(w_kpo_filt1k*z_i/(2*np.pi), w_spec_filt1k[:,z_set[1]]/my_w_star, label="25m res")
#ax.loglog(w_kpo5[36]*z_i/(2*np.pi), w_spec5[36,z_set[0]]/my_w_star, 'ko', label="5m crit")
#ax.loglog(w_kpo_filt2k*z_i/(2*np.pi), w_spec_filt2k[:,z_set[1]]/my_w_star, '-.', label="50m res")
# ax.loglog(w_kpo_filt3k*z_i/(2*np.pi), w_spec_filt3k[:,z_set[1]]/my_w_star, '-.', label="100m res")
# ax.loglog(w_kpo_filt4k*z_i/(2*np.pi), w_spec_filt4k[:,z_set[1]]/my_w_star, '-.', label="200m res")
# ax.loglog(w_kpo_filt5k*z_i/(2*np.pi), w_spec_filt5k[:,z_set[1]]/my_w_star, '-.', label="400m res")
# ax.loglog(w_kpo_filt6k*z_i/(2*np.pi), w_spec_filt6k[:,z_set[1]]/my_w_star, '-.', label="800m res")
#ax.loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')


#ax.loglog(125*filt_slope_x, 0.0006*filt_slope_y, 'k-')
#ax.text(80, 0.01, r'$k^{-11/2}$', fontsize=14)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlabel("$k z_i$", fontsize=14)
ax.set_ylabel("$\\mathcal{S}$ $w'^2$", fontsize=14) #("$\mathcal{S}$ ($w'$)", fontsize=14)
ax.set_ylim(ymax=1e2, ymin=1e-6)
ax.set_xlim(xmax=200, xmin=0.02)
#plt.xlim(xmax=1e0, xmin=1e-3)

def ktol(kz):
    l = (1/kz)*np.mean(z_i) #2*np.pi/k
    return l


def ltok(l):
    kz = (1/l)*np.mean(z_i) #2*np.pi/l
    return kz


secax = ax.secondary_xaxis('top', functions=(ltok, ktol))
secax.set_xlabel('$\\lambda$', fontsize=14)

plt.savefig(plotdir+"ARM_w_spectra_1st_and_2nd_filts.pdf", pad_inches=0)