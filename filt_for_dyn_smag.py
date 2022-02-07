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
odir = '/storage/silver/scenario/si818415/phd/20mLES/'
odir = odir + '20m_dyn_s_filt_' + options['FFT_type']+'/'

os.makedirs(odir, exist_ok = True)

file = 'cbl_13200.nc' #'diagnostics_3d_ts_13200.nc'
ref_file = None

plot_dir = odir + 'plots/'
os.makedirs(plot_dir, exist_ok = True)

plot_type = '.png'
data_dir = '' # Directory containing data
figshow = True

def main():
    '''
	20m data to be filtered using xarray
    '''
#   Non-global variables that are set once
    sigma_list = [15]
    dx = 20.0
    dy = 20.0
    N = 240
    filter_name = 'gaussian'
    width=-1
    cutoff=0.000001


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

    print(f'nch={nch}')

    dataset.close()

    defn = 1

    dataset = xr.open_dataset(dir+file, chunks={timevar: defn,
                                                xvar:nch, yvar:nch,
                                                'z':'auto', 'zn':'auto'})
    print(dataset)
#    ref_dataset = Dataset(dir+ref_file, 'r')
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

    ilev = 15
    iy = 40

    opgrid = 'w'
    fname = 'gauss'

    derived_data, exists = \
        sf.setup_derived_data_file( dir+file, odir, fname,
                                   options, override=True)

    filter_list = list([])

    for i,sigma in enumerate(sigma_list):
        if filter_name == 'gaussian':
            filter_id = 'filter_ga{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                       filter_name,
                                       sigma=sigma, width=width,
                                       delta_x=dx, cutoff=cutoff)
        elif filter_name == 'wave_cutoff':
            filter_id = 'filter_wc{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                       filter_name, wavenumber=np.pi/(2*sigma),
                                       width=width,
                                       delta_x=dx)
        elif filter_name == 'running_mean':
            filter_id = 'filter_rm{:02d}'.format(i)
            twod_filter = filt.Filter(filter_id,
                                       filter_name,
                                       width=width,
                                       delta_x=dx)

        filter_list.append(twod_filter)

# Add whole domain filter
    filter_name = 'domain'
    #filter_id = 'filter_do'
    filter_id = 'filter_do{:02d}'.format(len(filter_list))
    twod_filter = filt.Filter(filter_id, filter_name, delta_x=dx)
    filter_list.append(twod_filter)

    print(filter_list)

    for twod_filter in filter_list:

        print("Processing using filter: ")
        print(twod_filter)

        filtered_data, exists = \
            sf.setup_filtered_data_file( dir+file, odir, fname,
                                       options, twod_filter, override=True)
        if exists :
            print('Derived data file exists' )
        else :

            var_list = [
                "u",
                "v",
                "w",
                "th",
                ]
            field_list = sf.filter_variable_list(dataset, ref_dataset,
                                                 derived_data, filtered_data,
                                                 options, twod_filter,
                                                 var_list=var_list,
                                                 grid = opgrid)

            var_list = [
                    ["w","th"], ["u","u"], ["u","v"], ["u","w"],
                ["v","v"], ["v","w"], ["w","w"]
                  ]
            quad_field_list = sf.filter_variable_pair_list(dataset,
                                                  ref_dataset,
                                                  derived_data, filtered_data,
                                                  options, twod_filter,
                                                  var_list=var_list,
                                                  grid = opgrid)


       # if twod_filter.attributes['filter_type']!='domain' :
       #     fig1 = plt.figure(1)
       #     plt.contourf(twod_filter.data,20)
       #     plt.savefig(plot_dir+'Filter_'+\
       #                 twod_filter.id+plot_type)
       #     plt.close()

       #     fig2 = plt.figure(2)
       #     plt.plot(twod_filter.data[np.shape(twod_filter.data)[0]//2,:])
       #     plt.savefig(plot_dir+'Filter_y_xsect_'+\
       #                 twod_filter.id+plot_type)
       #     plt.close()

        filtered_data['ds'].close()
    derived_data['ds'].close()
    dataset.close()

if __name__ == "__main__":
    main()
