import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn_6hrs/'
file20 = "BOMEX_m0020_g0800_all_21600_gaussian_filter_"

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')
data_8D = path20f+file20+str('ga02.nc')

dataset_name2 = [path20f+file20+'C_th_2D.nc', path20f+file20+'C_qt_2D.nc']
dataset_name4 = [path20f+file20+'C_th_4D.nc', path20f+file20+'C_qt_4D.nc']
dataset_name8 = [path20f+file20+'C_th_8D.nc', path20f+file20+'C_qt_8D.nc']

scalar = ['th', 'q_total']

DX_2D = {
    'indir': data_2D,
    'dx_hat': 40
}
DX_4D = {
    'indir': data_4D,
    'dx_hat': 80
}
DX_8D = {
    'indir': data_8D,
    'dx_hat': 160
}


for i, scalar_in in enumerate(scalar):

########################################################################

    C_sq_prof_2D, C_prof_2D = \
        dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all = 1, **DX_2D)

    ds_2 = xr.Dataset()
    ds_2.to_netcdf(dataset_name2[i], mode='w')
    ds_in2 = {'file':dataset_name2[i], 'ds': ds_2}

    save_field(ds_in2, C_sq_prof_2D)
    save_field(ds_in2, C_prof_2D)

    ds_2.close()

    C_q_sq_prof_2D = None
    C_q_prof_2D = None

    ##########################################

    C_q_sq_prof_4D, C_q_prof_4D = \
        dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all = 1, **DX_4D)


    ds_4 = xr.Dataset()
    ds_4.to_netcdf(dataset_name4[i], mode='w')
    ds_in4 = {'file':dataset_name4[i], 'ds': ds_4}

    save_field(ds_in4, C_q_sq_prof_4D)
    save_field(ds_in4, C_q_prof_4D)

    ds_4.close()

    C_q_sq_prof_4D = None
    C_q_prof_4D = None



    #########################################################################


    C_q_sq_prof_8D, C_q_prof_8D = \
        dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_8D)



    ds_8 = xr.Dataset()
    ds_8.to_netcdf(dataset_name8[i], mode='w')
    ds_in8 = {'file':dataset_name8[i], 'ds': ds_8}

    save_field(ds_in8, C_q_sq_prof_8D)
    save_field(ds_in8, C_q_prof_8D)

    ds_8.close()

    C_q_sq_prof_8D = None
    C_q_prof_8D = None
