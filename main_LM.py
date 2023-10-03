import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

case = 'BOMEX'

av_type = 'all'
mygrid = 'p'

if case == 'BOMEX':
    path20f = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/'
    file20 = 'BOMEX_m0020_g0800_all_'
    times_list = ['14400']
    dx = 20

elif case == 'ARM':
    path20f = '/work/scratch-pw3/apower/ARM/corrected_sigmas/'
    file20 = 'diagnostics_3d_ts_' #{}_gaussian_filter_'
    times_list = ['18000', '25200', '32400', '39600']
    dx = 25


# data_4D = path20f+file20+str('ga01.nc')
# data_8D = path20f+file20+str('ga02.nc')
# data_16D = path20f+file20+str('ga03.nc')
# data_32D = path20f+file20+str('ga04.nc')
# data_64D = path20f+file20+str('ga05.nc')


# dataset_name4 = [path20f+file20+'Cs_4D.nc', path20f+file20+'C_th_4D.nc', path20f+file20+'C_qt_4D.nc']
# dataset_name8 = [path20f+file20+'Cs_8D.nc', path20f+file20+'C_th_8D.nc', path20f+file20+'C_qt_8D.nc']
# dataset_name16 = [path20f+file20+'Cs_16D.nc', path20f+file20+'C_th_16D.nc', path20f+file20+'C_qt_16D.nc']
# dataset_name32 = [path20f+file20+'Cs_32D.nc', path20f+file20+'C_th_32D.nc', path20f+file20+'C_qt_32D.nc']
# dataset_name64 = [path20f+file20+'Cs_64D.nc', path20f+file20+'C_th_64D.nc', path20f+file20+'C_qt_64D.nc']

scalar = ['momentum', 'th', 'q_total']
set_save_all = 2


# DX_4D = {
#     'indir': data_4D,
#     'dx_hat': 80
# }
# DX_8D = {
#     'indir': data_8D,
#     'dx_hat': 160
# }
# DX_16D = {
#     'indir': data_16D,
#     'dx_hat': 320
# }
# DX_32D = {
#     'indir': data_32D,
#     'dx_hat': 640
# }
# DX_64D = {
#     'indir': data_64D,
#     'dx_hat': 1280
# }

for nt, time_in in enumerate(times_list):

    file_in = file20+f'{time_in}_gaussian_filter_'
    data_2D = path20f + file_in + str('ga00.nc')
    dataset_name2 = [path20f + file_in + 'Cs_2D.nc',
                     path20f + file_in + 'C_th_2D.nc',
                     path20f + file_in + 'C_qt_2D.nc']

    DX_2D = {
        'indir': data_2D,
        'dx_hat': 2 * dx
    }

    for i, scalar_in in enumerate(scalar):

    ########################################################################
        #  = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        if scalar_in == 'momentum':
            z_save, zn_save, C_sq_prof_2D, C_prof_2D, HR_prof_2D, RR_prof_2D, HR_field_2D, RR_field_2D = \
             dy_s.Cs(dx=20, ingrid = mygrid, save_all = set_save_all, **DX_2D)
            # = \ #, C_sq_field_2D, Hj_2D, Rj_2D = \
        else:
            z_save, zn_save, C_sq_prof_2D, C_prof_2D, HR_prof_2D, RR_prof_2D, HR_field_2D, RR_field_2D = \
                dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all = set_save_all, **DX_2D)

        ds_2 = xr.Dataset()
        ds_2.to_netcdf(dataset_name2[i], mode='w')
        ds_in2 = {'file':dataset_name2[i], 'ds': ds_2}

        save_field(ds_in2, HR_field_2D)
        save_field(ds_in2, RR_field_2D)
        save_field(ds_in2, z_save)
        save_field(ds_in2, zn_save)
        save_field(ds_in2, C_sq_prof_2D)
        save_field(ds_in2, C_prof_2D)
        save_field(ds_in2, HR_prof_2D)
        save_field(ds_in2, RR_prof_2D)
        #save_field(ds_in2 = \ #, C_sq_field_2D)
        # save_field(ds_in2, Hj_2D)
        # save_field(ds_in2, Rj_2D)

        ds_2.close()

        C_sq_prof_2D = None
        C_prof_2D = None
        HR_prof_2D = None
        RR_prof_2D = None
        HR_field_2D = None
        RR_field_2D = None
        #C_sq_field_2D = None
        # Hj_2D = None
        # Rj_2D = None

        ##########################################
        # = \ #, C_sq_field_4D, Hj_4D, Rj_4D = \
    #     if scalar_in == 'momentum':
    #         z_save, zn_save, C_sq_prof_4D, C_prof_4D, HR_prof_4D, RR_prof_4D, HR_field_4D, RR_field_4D = \
    #             dy_s.Cs(dx=20, ingrid = mygrid, save_all = set_save_all, **DX_4D)
    #         # = \ #, C_sq_field_4D, Hj_4D, Rj_4D = \
    #     else:
    #         z_save, zn_save, C_sq_prof_4D, C_prof_4D, HR_prof_4D, RR_prof_4D, HR_field_4D, RR_field_4D = \
    #             dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all = set_save_all, **DX_4D)
    #
    #
    #     ds_4 = xr.Dataset()
    #     ds_4.to_netcdf(dataset_name4[i], mode='w')
    #     ds_in4 = {'file':dataset_name4[i], 'ds': ds_4}
    #
    #     save_field(ds_in4, HR_field_4D)
    #     save_field(ds_in4, RR_field_4D)
    #     save_field(ds_in4, z_save)
    #     save_field(ds_in4, zn_save)
    #     save_field(ds_in4, C_sq_prof_4D)
    #     save_field(ds_in4, C_prof_4D)
    #     save_field(ds_in4, HR_prof_4D)
    #     save_field(ds_in4, RR_prof_4D)
    #     #save_field(ds_in4 = \ #, C_sq_field_4D)
    #     # save_field(ds_in4, Hj_4D)
    #     # save_field(ds_in4, Rj_4D)
    #
    #     ds_4.close()
    #
    #     C_sq_prof_4D = None
    #     C_prof_4D = None
    #     HR_prof_4D = None
    #     RR_prof_4D = None
    #     HR_field_4D = None
    #     RR_field_4D = None
    #     C_sq_field_4D = None
    #     # Hj_4D = None
    #     # Rj_4D = None
    #
    #
    #     #########################################################################
    #     #  = \ #, C_sq_field_8D, Hj_8D, Rj_8D = \
    #     if scalar_in == 'momentum':
    #         z_save, zn_save, C_sq_prof_8D, C_prof_8D, HR_prof_8D, RR_prof_8D, HR_field_8D, RR_field_8D = \
    #             dy_s.Cs(dx=20, ingrid=mygrid, save_all=set_save_all, **DX_8D)
    #         #  = \ #, C_sq_field_8D, Hj_8D, Rj_8D = \
    #     else:
    #         z_save, zn_save, C_sq_prof_8D, C_prof_8D, HR_prof_8D, RR_prof_8D, HR_field_8D, RR_field_8D = \
    #             dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=set_save_all, **DX_8D)
    #
    #
    #
    #     ds_8 = xr.Dataset()
    #     ds_8.to_netcdf(dataset_name8[i], mode='w')
    #     ds_in8 = {'file':dataset_name8[i], 'ds': ds_8}
    #
    #     save_field(ds_in8, HR_field_8D)
    #     save_field(ds_in8, RR_field_8D)
    #     save_field(ds_in8, z_save)
    #     save_field(ds_in8, zn_save)
    #     save_field(ds_in8, C_sq_prof_8D)
    #     save_field(ds_in8, C_prof_8D)
    #     save_field(ds_in8, HR_prof_8D)
    #     save_field(ds_in8, RR_prof_8D)
    #     #save_field(ds_in8 = \ #, C_sq_field_8D)
    #     # save_field(ds_in8, Hj_8D)
    #     # save_field(ds_in8, Rj_8D)
    #
    #     ds_8.close()
    #
    #     C_sq_prof_8D = None
    #     C_prof_8D = None
    #     HR_prof_8D = None
    #     RR_prof_8D = None
    #     HR_field_8D = None
    #     RR_field_8D = None
    #     C_sq_field_8D = None
    #     # Hj_8D = None
    #     # Rj_8D = None
    #
    #     ##########################################
    #     # = \ #, C_sq_field_16D, Hj_16D, Rj_16D = \
    #     if scalar_in == 'momentum':
    #         z_save, zn_save, C_sq_prof_16D, C_prof_16D, HR_prof_16D, RR_prof_16D, HR_field_16D, RR_field_16D = \
    #             dy_s.Cs(dx=20, ingrid=mygrid, save_all=set_save_all, **DX_16D)
    #         #  = \ #, C_sq_field_16D, Hj_16D, Rj_16D = \
    #     else:
    #         z_save, zn_save, C_sq_prof_16D, C_prof_16D, HR_prof_16D, RR_prof_16D, HR_field_16D, RR_field_16D = \
    #             dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=set_save_all, **DX_16D)
    #
    #     ds_16 = xr.Dataset()
    #     ds_16.to_netcdf(dataset_name16[i], mode='w')
    #     ds_in16 = {'file':dataset_name16[i], 'ds': ds_16}
    #
    #     save_field(ds_in16, HR_field_16D)
    #     save_field(ds_in16, RR_field_16D)
    #     save_field(ds_in16, z_save)
    #     save_field(ds_in16, zn_save)
    #     save_field(ds_in16, C_sq_prof_16D)
    #     save_field(ds_in16, C_prof_16D)
    #     save_field(ds_in16, HR_prof_16D)
    #     save_field(ds_in16, RR_prof_16D)
    #     #save_field(ds_in16 = \ #, C_sq_field_16D)
    #     # save_field(ds_in16, Hj_16D)
    #     # save_field(ds_in16, Rj_16D)
    #
    #     ds_16.close()
    #
    #     C_sq_prof_16D = None
    #     C_prof_16D = None
    #     HR_prof_16D = None
    #     RR_prof_16D = None
    #     HR_field_16D = None
    #     RR_field_16D = None
    #     #C_sq_field_16D = None
    #     # Hj_16D = None
    #     # Rj_16D = None
    #
    # ##########################################
    #     #  = \ #, C_sq_field_32D, Hj_32D, Rj_32D = \
    #     if scalar_in == 'momentum':
    #         z_save, zn_save, C_sq_prof_32D, C_prof_32D, HR_prof_32D, RR_prof_32D, HR_field_32D, RR_field_32D = \
    #             dy_s.Cs(dx=20, ingrid=mygrid, save_all=set_save_all, **DX_32D)
    #         #  = \ #, C_sq_field_32D, Hj_32D, Rj_32D = \
    #     else:
    #         z_save, zn_save, C_sq_prof_32D, C_prof_32D, HR_prof_32D, RR_prof_32D, HR_field_32D, RR_field_32D = \
    #             dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=set_save_all, **DX_32D)
    #
    #     ds_32 = xr.Dataset()
    #     ds_32.to_netcdf(dataset_name32[i], mode='w')
    #     ds_in32 = {'file': dataset_name32[i], 'ds': ds_32}
    #
    #     save_field(ds_in32, HR_field_32D)
    #     save_field(ds_in32, RR_field_32D)
    #     save_field(ds_in32, z_save)
    #     save_field(ds_in32, zn_save)
    #     save_field(ds_in32, C_sq_prof_32D)
    #     save_field(ds_in32, C_prof_32D)
    #     save_field(ds_in32, HR_prof_32D)
    #     save_field(ds_in32, RR_prof_32D)
    #     # save_field(ds_in32 = \ #, C_sq_field_32D)
    #     # save_field(ds_in32, Hj_32D)
    #     # save_field(ds_in32, Rj_32D)
    #
    #     ds_32.close()
    #
    #     C_sq_prof_32D = None
    #     C_prof_32D = None
    #     HR_prof_32D = None
    #     RR_prof_32D = None
    #     HR_field_32D = None
    #     RR_field_32D = None
    #     C_sq_field_32D = None
    #     # Hj_32D = None
    #     # Rj_32D = None
    #
    #     ##########################################
    #     #  = \ #, C_sq_field_64D, Hj_64D, Rj_64D = \
    #     if scalar_in == 'momentum':
    #         z_save, zn_save, C_sq_prof_64D, C_prof_64D, HR_prof_64D, RR_prof_64D, HR_field_64D, RR_field_64D = \
    #             dy_s.Cs(dx=20, ingrid=mygrid, save_all=set_save_all, **DX_64D)
    #         #  = \ #, C_sq_field_64D, Hj_64D, Rj_64D = \
    #     else:
    #         z_save, zn_save, C_sq_prof_64D, C_prof_64D, HR_prof_64D, RR_prof_64D, HR_field_64D, RR_field_64D = \
    #             dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=set_save_all, **DX_64D)
    #
    #     ds_64 = xr.Dataset()
    #     ds_64.to_netcdf(dataset_name64[i], mode='w')
    #     ds_in64 = {'file': dataset_name64[i], 'ds': ds_64}
    #
    #     save_field(ds_in64, HR_field_64D)
    #     save_field(ds_in64, RR_field_64D)
    #     save_field(ds_in64, z_save)
    #     save_field(ds_in64, zn_save)
    #     save_field(ds_in64, C_sq_prof_64D)
    #     save_field(ds_in64, C_prof_64D)
    #     save_field(ds_in64, HR_prof_64D)
    #     save_field(ds_in64, RR_prof_64D)
    #     #save_field(ds_in64 = \ #, C_sq_field_64D)
    #     # save_field(ds_in64, Hj_64D)
    #     # save_field(ds_in64, Rj_64D)
    #
    #     ds_64.close()
    #
    #     C_sq_prof_64D = None
    #     C_prof_64D = None
    #     HR_prof_64D = None
    #     RR_prof_64D = None
    #     HR_field_64D = None
    #     RR_field_64D = None
    #     C_sq_field_64D = None
    #     # Hj_64D = None
    #     # Rj_64D = None
