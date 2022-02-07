import os
import sys
import numpy as np
import xarray as xr

import subfilter as sf
import filters as filt
import spectra as sp
import functions as f

import matplotlib.pyplot as plt

level = 200
indir = '/storage/silver/MONC_data/Alanna/MONC_runs/'
time_list = ["12600", "13200", "13800"]
options = {'FFT_type':'RFFT'}

model_res_list = ['20', '50', '100', '200', '400', '800']
model_res_list_int = np.array([int(x) for x in model_res_list ])

filter_name_list = ['gaussian',
                    'circular_wave_cutoff']

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

w2_av_list = np.zeros(len(model_res_list))
w2_filt_time = np.zeros((len(filter_name_list), len(model_res_list), len(time_list))) 
var_err = np.zeros(len(model_res_list))

for i, model_res in enumerate(model_res_list_int):
    
    infile_in = f'{indir}{model_res}m/cbl_{time_list[0]}.nc'
    ds_in = xr.open_dataset(infile_in)
    wa_in = ds_in['w'].sel(z=level, method='nearest')[0,...]
    w_in = wa_in.data     
    n_in = np.shape(w_in)[0]
    ds_in.close()
    
    w2_list = np.zeros(len(time_list))    
    w_save_err = np.zeros((len(time_list), n_in, n_in))
    w2_spec_sum = 0
    w2_kpo_sum = 0
    w2_filt_spec_sum = 0
    w2_filt_kpo_sum = 0
  
    for k, time_stamp in enumerate(time_list):
        
        infile = f'{indir}{model_res}m/cbl_{time_stamp}.nc'
        ds = xr.open_dataset(infile)

        wa = ds['w'].sel(z=level, method='nearest')[-1,...]
        w = wa.data
        #w = w-np.mean(w)       
        npoints = np.shape(w)[0]
        dx = model_res_list_int[i]
        w2_list[k] = np.mean(w * w)
        
        w2_spec, w2_kpo = f.spectra_2d(w, model_res, model_res, options_spec, z=0)       
        #np.save(f'files/{model_res}_{time_stamp}_w_spec', w2_spec)
        #np.save(f'files/{model_res}_{time_stamp}_w_kpo', w2_kpo)
        w2_spec_sum += w2_spec
        w2_kpo_sum += w2_kpo
        ds.close()
        
        if i == 0:
            
            w_ref = wa
            dx_ref = dx
            npoints_ref = npoints

            for j, filter_name in enumerate(filter_name_list):
                
                for l, dx in enumerate(model_res_list_int):
                    
                    if l == 0:
                        w2_filt_time[j,l,k] = w2_list[k]
                        
                    else:
                        if filter_name == 'gaussian':
                            filter_id = f'filter_ga{i:02d}'
                            sigma = dx / 2.0
                            print(f'sigma = {sigma}')
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name,
                                                      npoints=npoints_ref,
                                                      sigma=sigma,
                                                      delta_x=dx_ref)

                        elif filter_name == 'wave_cutoff':
                            filter_id = 'filter_wc{:02d}'.format(i)
                            wavenumber = np.pi / dx
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name,
                                                      npoints=npoints_ref,
                                                      wavenumber=wavenumber,
                                                      delta_x=dx_ref)

                        elif filter_name == 'circular_wave_cutoff':
                            filter_id = 'filter_cwc{:02d}'.format(i)
                            wavenumber = np.pi / dx
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name,
                                                      npoints=npoints_ref,
                                                      wavenumber=wavenumber,
                                                      delta_x=dx_ref)

                        elif filter_name == 'running_mean':
                            filter_id = 'filter_rm{:02d}'.format(i)
                            width = int(np.round( sigma/dx *  np.sqrt(2.0 *np.pi))+1)
                            twod_filter = filt.Filter(filter_id,
                                                      filter_name,
                                                      npoints=npoints,
                                                      width=width,
                                                      delta_x=dx)

                        (w_r, w_s) = sf.filtered_field_calc(w_ref, options, twod_filter)
                        
                        w2_filt_spec, w2_filt_kpo = f.spectra_2d(w, model_res, model_res, options_spec, z=0)       
                        #np.save(f'files/{model_res}_{time_stamp}_w_spec', w2_spec)
                        #np.save(f'files/{model_res}_{time_stamp}_w_kpo', w2_kpo)
                        w2_filt_spec_sum += w2_filt_spec
                        w2_filt_kpo_sum += w2_filt_kpo

                        w2_filt_time[j,l,k] = np.mean(w_r.data * w_r.data)
    
    w2_av_list[i] = np.mean(w2_list)
    
    var_err[i] = np.std(w_save_err)/(np.sqrt((n_in**2)*len(time_list)))
    
    av_w2_spec = w2_spec_sum/len(time_list)
    av_w2_kpo = w2_kpo_sum/len(time_list)
    np.save(f'files/{model_res_list[i]}_w_spec_av', av_w2_spec)
    np.save(f'files/{model_res_list[i]}_w_kpo_av', av_w2_kpo)
    
    
w2_av_filt = np.mean(w2_filt_time, axis=2)

z_i = 1000
col_list = ['g', 'c', 'r', 'y', 'm', 'k', 'tab:gray']

figs = plt.figure(figsize=(6,6))  
plt.ylabel(r'$w^2/w_{20}^2$')
plt.xlabel(r'$w^2/w_{20}^2$')
plt.semilogx(model_res_list_int/z_i, w2_av_list/w2_av_list[0], '-s', label='MONC run')
for m, filter_name in enumerate(filter_name_list):
    plt.semilogx(model_res_list_int/z_i, w2_av_filt[m,:]/w2_av_list[0], '-x', label=filter_name)   
for i in range(len(model_res_list_int)):
    plt.errorbar(model_res_list_int[i]/z_i, w2_av_list[i]/w2_av_list[0], yerr=var_err[i]/w2_av_list[0], label=str(model_res_list[i])+'m', 
                 color = col_list[i], ecolor='green', fmt='o', capsize=7)
plt.legend(fontsize=12, loc='lower left')
plt.xticks(np.array([0.01, 0.1, 1]), [0.01, 0.1, 1])
figs.savefig(f'plots/w_{level}_sigmoid.png')



w2_spec_20 = np.load('files/20_w_spec_av.npy')
w2_kpo_20 = np.load('files/20_w_kpo_av.npy')

# w2_spec_20a = np.load(f'files/20_{time_list[0]}_w_spec.npy')
# w2_spec_20b = np.load(f'files/20_{time_list[1]}_w_spec.npy')
# w2_spec_20c = np.load(f'files/20_{time_list[2]}_w_spec.npy')
# w2_kpo_20a = np.load(f'files/20_{time_list[0]}_w_kpo.npy')
# w2_kpo_20b = np.load(f'files/20_{time_list[1]}_w_kpo.npy')
# w2_kpo_20c = np.load(f'files/20_{time_list[2]}_w_kpo.npy')

w2_spec_50 = np.load('files/50_w_spec_av.npy')
w2_kpo_50 = np.load('files/50_w_kpo_av.npy')

w2_spec_100 = np.load('files/100_w_spec_av.npy')
w2_kpo_100 = np.load('files/100_w_kpo_av.npy')

w2_spec_200 = np.load('files/200_w_spec_av.npy')
w2_kpo_200 = np.load('files/200_w_kpo_av.npy')

w2_spec_400 = np.load('files/400_w_spec_av.npy')
w2_kpo_400 = np.load('files/400_w_kpo_av.npy')

w2_spec_800 = np.load('files/800_w_spec_av.npy')
w2_kpo_800 = np.load('files/800_w_kpo_av.npy')

fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
turb_slope_x = np.linspace(0.02,0.4,100)
turb_slope_y = turb_slope_x**(-5/3)
filt_slope_x = np.linspace(0.5,1,50)
filt_slope_y = filt_slope_x**(-11/2)
w_star = 2
my_w_star = w_star**2
z_i = 1000

ax.loglog(w2_kpo_20*z_i/(2*np.pi), w2_spec_20/my_w_star, label="20m av res")

# ax.loglog(w2_kpo_20a*z_i/(2*np.pi), w2_spec_20a/my_w_star, label="20m time 1")
# ax.loglog(w2_kpo_20b*z_i/(2*np.pi), w2_spec_20b/my_w_star, label="20m time 2")
# ax.loglog(w2_kpo_20c*z_i/(2*np.pi), w2_spec_20c/my_w_star, label="20m time 3")


ax.loglog(w2_kpo_50*z_i/(2*np.pi), w2_spec_50/my_w_star, label="50m res")
ax.loglog(w2_kpo_100*z_i/(2*np.pi), w2_spec_100/my_w_star, label="100m res")
# ax.loglog(w2_kpo_200*z_i/(2*np.pi), w2_spec_200/my_w_star, label="200m res")
# ax.loglog(w2_kpo_400*z_i/(2*np.pi), w2_spec_400/my_w_star, label="400m res")
# ax.loglog(w2_kpo_800*z_i/(2*np.pi), w2_spec_800/my_w_star, label="800m res")
ax.loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
ax.text(10, 1.6, r'$k^{-5/3}$', fontsize=14)

ax.legend(fontsize=12, loc='lower left')
ax.set_xlabel("$k z_i$", fontsize=14)
ax.set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14) #("$\mathcal{S}$ ($w'$)", fontsize=14)
ax.set_ylim(ymax=1e2, ymin=1e-6)
ax.set_xlim(xmax=200, xmin=0.003)
def ktol(kz):
    l_over_z = (1/kz)
    return l_over_z

def ltok(l_over_z):
    kz = (1/l_over_z)
    return kz
secax = ax.secondary_xaxis('top', functions=(ltok, ktol))
secax.set_xlabel('$\\lambda / z_i$', fontsize=14)
plt.savefig(f'plots/w2_spec_{level}_less_ext_domains.png', pad_inches=0)


####################################################################################

fig2, axs = plt.subplots(3,2, figsize=(10,6), constrained_layout=True)
turb_slope_x = np.linspace(0.02,0.4,100)
turb_slope_y = turb_slope_x**(-5/3)
filt_slope_x = np.linspace(0.5,1,50)
filt_slope_y = filt_slope_x**(-11/2)
w_star = 2
my_w_star = w_star**2
z_i = 1000

axs[0,0].loglog(w2_kpo_20*z_i/(2*np.pi), w2_spec_20/my_w_star, label="20m res")
axs[0,0].loglog(w2_kpo_50*z_i/(2*np.pi), w2_spec_50/my_w_star, label="50m res")
axs[0,0].loglog(w2_kpo_100*z_i/(2*np.pi), w2_spec_100/my_w_star, label="100m res")
axs[0,0].loglog(w2_kpo_200*z_i/(2*np.pi), w2_spec_200/my_w_star, label="200m res")
axs[0,0].loglog(w2_kpo_400*z_i/(2*np.pi), w2_spec_400/my_w_star, label="400m res")
axs[0,0].loglog(w2_kpo_800*z_i/(2*np.pi), w2_spec_800/my_w_star, label="800m res")
axs[0,0].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[0,0].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[0,0].legend(fontsize=12, loc='lower left')
axs[0,0].set_xlabel("$k z_i$", fontsize=14)
axs[0,0].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[0,0].set_ylim(ymax=1e2, ymin=1e-6)
axs[0,0].set_xlim(xmax=200, xmin=0.003)


axs[0,1].loglog(w2_kpo_50*z_i/(2*np.pi), w2_spec_50/my_w_star, label="50m res")
axs[0,1].loglog(w2_kpo_50*z_i/(2*np.pi), w2_av_filt[0,1]/my_w_star, label="Gaussian")
axs[0,1].loglog(w2_kpo_50*z_i/(2*np.pi), w2_av_filt[1,1]/my_w_star, label="Circular Wave Cutoff")
axs[0,1].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[0,1].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[0,1].legend(fontsize=12, loc='lower left')
axs[0,1].set_xlabel("$k z_i$", fontsize=14)
axs[0,1].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[0,1].set_ylim(ymax=1e2, ymin=1e-6)
axs[0,1].set_xlim(xmax=200, xmin=0.003)



axs[1,0].loglog(w2_kpo_100*z_i/(2*np.pi), w2_spec_100/my_w_star, label="100m res")
axs[1,0].loglog(w2_kpo_100*z_i/(2*np.pi), w2_av_filt[0,2]/my_w_star, label="Gaussian")
axs[1,0].loglog(w2_kpo_100*z_i/(2*np.pi), w2_av_filt[1,2]/my_w_star, label="Circular Wave Cutoff")
axs[1,0].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[1,0].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[1,0].legend(fontsize=12, loc='lower left')
axs[1,0].set_xlabel("$k z_i$", fontsize=14)
axs[1,0].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[1,0].set_ylim(ymax=1e2, ymin=1e-6)
axs[1,0].set_xlim(xmax=200, xmin=0.003)



axs[1,1].loglog(w2_kpo_200*z_i/(2*np.pi), w2_spec_200/my_w_star, label="200m res")
axs[1,1].loglog(w2_kpo_200*z_i/(2*np.pi), w2_av_filt[0,3]/my_w_star, label="Gaussian")
axs[1,1].loglog(w2_kpo_200*z_i/(2*np.pi), w2_av_filt[1,3]/my_w_star, label="Circular Wave Cutoff")
axs[1,1].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[1,1].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[1,1].legend(fontsize=12, loc='lower left')
axs[1,1].set_xlabel("$k z_i$", fontsize=14)
axs[1,1].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[1,1].set_ylim(ymax=1e2, ymin=1e-6)
axs[1,1].set_xlim(xmax=200, xmin=0.003)



axs[2,0].loglog(w2_kpo_400*z_i/(2*np.pi), w2_spec_400/my_w_star, label="400m res")
axs[2,0].loglog(w2_kpo_400*z_i/(2*np.pi), w2_av_filt[0,4]/my_w_star, label="Gaussian")
axs[2,0].loglog(w2_kpo_400*z_i/(2*np.pi), w2_av_filt[1,4]/my_w_star, label="Circular Wave Cutoff")
axs[2,0].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[2,0].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[2,0].legend(fontsize=12, loc='lower left')
axs[2,0].set_xlabel("$k z_i$", fontsize=14)
axs[2,0].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[2,0].set_ylim(ymax=1e2, ymin=1e-6)
axs[2,0].set_xlim(xmax=200, xmin=0.003)



axs[2,1].loglog(w2_kpo_800*z_i/(2*np.pi), w2_spec_800/my_w_star, label="800m res")
axs[2,1].loglog(w2_kpo_800*z_i/(2*np.pi), w2_av_filt[0,5]/my_w_star, label="Gaussian")
axs[2,1].loglog(w2_kpo_800*z_i/(2*np.pi), w2_av_filt[1,5]/my_w_star, label="Circular Wave Cutoff")
axs[2,1].loglog(90*turb_slope_x, 0.015*turb_slope_y, 'k-')
axs[2,1].text(10, 1.6, r'$k^{-5/3}$', fontsize=14)
axs[2,1].legend(fontsize=12, loc='lower left')
axs[2,1].set_xlabel("$k z_i$", fontsize=14)
axs[2,1].set_ylabel("$\\mathcal{S}$ $(w')^2/w_*^2$", fontsize=14)
axs[2,1].set_ylim(ymax=1e2, ymin=1e-6)
axs[2,1].set_xlim(xmax=200, xmin=0.003)


def ktol(kz):
    l_over_z = (1/kz)
    return l_over_z

def ltok(l_over_z):
    kz = (1/l_over_z)
    return kz
secax = ax.secondary_xaxis('top', functions=(ltok, ktol))
secax.set_xlabel('$\\lambda / z_i$', fontsize=14)
plt.savefig(f'plots/w2_spec_{level}_with_filters.png', pad_inches=0)