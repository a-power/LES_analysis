import os
import numpy as np
import xarray as xr

import subfilter as sf
import filters as filt
import dask


options = {
        'FFT_type': 'RFFT',
        'save_all': 'Yes',
        'th_ref': 300.0,
        'dx': 20.0,
        'dy': 20.0,
          }

set_time = '14400'
dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
outdir_og = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/'
odir = outdir_og + '20m_gauss_dyn_oldscript' +'/'



os.makedirs(odir, exist_ok = True)

file = 'BOMEX_m0020_g0800_all_14400.nc'
ref_file = None

def main():
    '''
	20m data to be filtered using xarray
    '''
#   Non-global variables that are set once
    domain_in = 16 #km
    dx = 20.0
    N = int((domain_in*(1000))/dx)
    filter_name =  'gaussian' #"wave_cutoff"
    width=-1
    cutoff=0.000001
    
    opgrid = 'p'

    sigma_list = np.array([20, 40])

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


        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()

if __name__ == "__main__":
    main()
