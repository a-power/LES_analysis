import os
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

import subfilter as sf
import filters as filt
import difference_ops as do
import dask


options = {
#        'FFT_type': 'FFTconvolve',
#        'FFT_type': 'FFT',
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
          }


dir = '/storage/silver/scenario/si818415/phd/20mLES/'
odir = '/storage/silver/MONC_data/Alanna/'
odir = odir + '20m_gauss_sig_Delta_match_25' +'/'

os.makedirs(odir, exist_ok = True)

file = 'cbl_13200.nc' 
ref_file = None

def k_cut_find(delta):
    return np.pi/(delta)

def sigma_find(delta):
    return delta/2 #(np.sqrt(6))

def L_ij_sym(u, v, w, uu, uv, uw, vv, vw, ww):
    
    L_ij = np.array([ (u*u-uu), (u*v-uv), (u*w-uw),
                                (v*v-vv), (v*w-vw),
                                          (w*w-ww) ] )
    return L_ij

def main():
    '''
	20m data to be filtered using xarray
    '''
#   Non-global variables that are set once
    dx = 20.0
    dy = 20.0
    N = 240
    filter_name =  'gaussian' #"wave_cutoff"
    width=-1
    cutoff=0.000001
    
    opgrid = 'w'
    
    #Delta = np.array([20, 40, 80, 160, 320])
    #k_cut_list = np.array([0.1571])#k_cut_find(Delta) 
    sigma_list = np.array([25])

    dask.config.set({"array.slicing.split_large_chunks": True})
    dataset = xr.open_dataset(dir+file)
    [itime, iix, iiy, iiz] = sf.find_var(dataset.dims, ['time', 'x', 'y', 'z'])
    timevar = list(dataset.dims)[itime]
    xvar = list(dataset.dims)[iix]
    yvar = list(dataset.dims)[iiy]
    zvar = list(dataset.dims)[iiz]
    max_ch = sf.subfilter_setup['chunk_size']

    # This is a rough way to estimate chunck size
    nch = np.min([int(dataset.dims[xvar]/(2**int(np.log(dataset.dims[xvar]
                                                *dataset.dims[yvar]
                                                *dataset.dims[zvar]
                                                /max_ch)/np.log(2)/2))),
                  dataset.dims[xvar]])

    dataset.close()

    defn = 1

    dataset = xr.open_dataset(dir+file, chunks={timevar: defn,
                                                xvar:nch, yvar:nch,
                                                'z':'auto', 'zn':'auto'})

    if ref_file is not None:
        ref_dataset = xr.open_dataset(dir+ref_file)
    else:
        ref_dataset = None

    od = sf.options_database(dataset)
    if od is None:
        dx = options['dx']
        dy = options['dy']
    else:
        dx = float(od['dxx'])
        dy = float(od['dyy'])
    
    fname = filter_name

    derived_data, exists = \
        sf.setup_derived_data_file( dir+file, odir, fname,
                                   options, override=True)

    filter_list = list([])
    
    if filter_name == 'gaussian':
        filt_scale_list = sigma_list
        
    elif filter_name == 'wave_cutoff':
        filt_scale_list = k_cut_list
        
    elif filter_name == 'running_mean':
        filt_scale_list = width_list

    for i,filt_set in enumerate(filt_scale_list):
        print(filt_set)
        if filter_name == 'gaussian':
            filter_id = 'filter_ga{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                       filter_name, npoints=N,
                                       sigma=filt_set, width=width,
                                       delta_x=dx, cutoff=cutoff)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id, filter_name, 
                                      wavenumber=filt_set, #np.pi/(2*sigma),
                                       width=width, npoints=N,
                                       delta_x=dx)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                       filter_name,
                                       width=filt_set,
                                       npoints=N,
                                       delta_x=dx)

        filter_list.append(twod_filter)

# Add whole domain filter
    filter_name = 'domain'
    #filter_id = 'filter_do'
    filter_id = 'filter_do{:02d}'.format(len(filter_list))
    twod_filter = filt.Filter(filter_id, filter_name, delta_x=dx)
    filter_list.append(twod_filter)

    print(filter_list)

    for j,new_filter in enumerate(filter_list):

        print("Processing using filter: ")
        print(new_filter)

        filtered_data, exists = \
            sf.setup_filtered_data_file( dir+file, odir, fname,
                                       options, new_filter, override=True)
        if exists :
            print('Derived data file exists' )
        else :
            

            var_list = ["u",
                       "v",
                       "w",
                       "th"
                       ]
                 

            
            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 options, new_filter,
                                                 var_list=var_list,
                                                 grid = opgrid)

            var_list = [ ["w","th"],
                       ["u","u"],
                       ["u","v"],
                       ["u","w"],
                       ["v","v"],
                       ["v","w"],
                       ["w","w"]
                       ]
                  
            quad_field_list = sf.filter_variable_pair_list(dataset,
                                                  ref_dataset,
                                                  derived_data, filtered_data,
                                                  options, new_filter,
                                                  var_list=var_list,
                                                           grid = opgrid)
            
            ##################### Lij
            
                #             u_hat = filtered_data['u_on_p_r']             
                #             v_hat = filtered_data['v_on_p_r']             
                #             w_hat = filtered_data['w_on_p_r']            
                #             uu_hat = filtered_data['u_on_p.u_on_p_r']             
                #             uv_hat = filtered_data['u_on_p.v_on_p_r']             
                #             uw_hat = filtered_data['u_on_p.w_on_p_r']             
                #             vv_hat = filtered_data['v_on_p.v_on_p_r']             
                #             vw_hat = filtered_data['v_on_p.w_on_p_r']             
                #             ww_hat = filtered_data['w_on_p.w_on_p_r'] 

                #             L_ij = L_ij_sym(u_hat, v_hat, w_hat, uu_hat, uv_hat, uw_hat, vv_hat, vw_hat, ww_hat)
                #             L_ij.name = "L_ij"
                #             L_ij = save_field(filtered_data, L_ij)


              ### Lines 204 to 217 don't run - grid / indexing problem ###
    
            #deform_filt_r, deform_filt_s = sf.filtered_deformation(dataset,
            #                                      ref_dataset,
            #                                      derived_data, filtered_data,
            #                                      options, new_filter)#,
                                                           #grid = opgrid)
            
          
            
            #             S_ij_temp_hat, abs_S_temp_hat = sf.shear(deform_filt_r)

            #             S_ij_hat = 1/2*S_ij_temp_hat
            #             abs_S_hat = np.sqrt(abs_S_temp_hat)
            
            
            ###############################################            
            
            
            deform = sf.deformation(dataset,
                                    ref_dataset,
                                    derived_data,
                                    options)
            
            S_ij_temp, abs_S_temp = sf.shear(deform)
            
            S_ij = 1/2*S_ij_temp
            S_ij.name = 'S_ij'
            abs_S = np.sqrt(abs_S_temp)
            abs_S.name = "abs_S"
            
            S_ij_filt = sf.filter_field(S_ij, filtered_data,
                                                  options, new_filter)
            
            abs_S_filt = sf.filter_field(abs_S, filtered_data,
                                                  options, new_filter)
            
            
            S_ij_abs_S = S_ij*abs_S
            S_ij_abs_S.name = 'S_ij_abs_S'
            
            S_ij_abs_S_hat_filt = sf.filter_field(S_ij_abs_S, filtered_data,
                                                  options, new_filter)
            
            #             filt_scale_list_copy = filt_scale_list.copy()
            #             filt_scale_list_copy.append("domain")
            #             filt_scale = xr.DataArray(data=filt_scale_list_copy[j], name = "filt_scale")
            #             filt_scale = sf.save_field(filtered_data, filt_scale)
            



        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()

if __name__ == "__main__":
    main()
