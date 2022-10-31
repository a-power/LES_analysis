import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

hrs = ['6', '13']
sec = ['21600', '48600']

for i in range(len(hrs)):

    path20f = f'/work/scratch-pw/apower/20m_gauss_dyn_{hrs[i]}hrs/'
    file20 = f'BOMEX_m0020_g0800_all_{sec[i]}_gaussian_filter_'

    data_2D = path20f+file20+str('ga00.nc')
    data_4D = path20f+file20+str('ga01.nc')
    data_8D = path20f+file20+str('ga02.nc')

    dataset_name2 = [path20f+file20+'Cs_2D.nc', path20f + file20 + 'C_th_2D.nc', path20f + file20 + 'C_qt_2D.nc']
    dataset_name4 = [path20f+file20+'Cs_4D.nc', path20f + file20 + 'C_th_4D.nc', path20f + file20 + 'C_qt_4D.nc']
    dataset_name8 = [path20f+file20+'Cs_8D.nc', path20f + file20 + 'C_th_8D.nc', path20f + file20 + 'C_qt_8D.nc']



    scalar = ['momentum', 'th', 'q_total']

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



    for j, scalar_in in enumerate(scalar):

        if scalar_in == 'momentum':
            C_sq_prof_2D, C_prof_2D = dy_s.Cs(dx=20, ingrid = mygrid, save_all=0, **DX_2D)
        else:

            C_sq_prof_2D, C_prof_2D = dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=1, **DX_2D)

        ds_2 = xr.Dataset()
        ds_2.to_netcdf(dataset_name2[j], mode='w')
        ds_in2 = {'file': dataset_name2[j], 'ds': ds_2}

        save_field(ds_in2, C_sq_prof_2D)
        save_field(ds_in2, C_prof_2D)

        ds_2.close()

        C_q_sq_prof_2D = None
        C_q_prof_2D = None

####################################################################################################################

        if scalar_in == 'momentum':
            C_sq_prof_4D, C_prof_4D = dy_s.Cs(dx=20, ingrid = mygrid, save_all=0, **DX_4D)
        else:

            C_sq_prof_4D, C_prof_4D = dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=1, **DX_4D)

        ds_4 = xr.Dataset()
        ds_4.to_netcdf(dataset_name4[j], mode='w')
        ds_in4 = {'file': dataset_name4[j], 'ds': ds_4}

        save_field(ds_in4, C_sq_prof_4D)
        save_field(ds_in4, C_prof_4D)

        ds_4.close()

        C_q_sq_prof_4D = None
        C_q_prof_4D = None

####################################################################################################################

        if scalar_in == 'momentum':
            C_sq_prof_8D, C_prof_8D = dy_s.Cs(dx=20, ingrid=mygrid, save_all=0, **DX_8D)
        else:

            C_sq_prof_8D, C_prof_8D = dy_s.C_scalar(scalar=scalar_in, dx=20, ingrid=mygrid, save_all=1, **DX_8D)

        ds_8 = xr.Dataset()
        ds_8.to_netcdf(dataset_name8[j], mode='w')
        ds_in8 = {'file': dataset_name8[j], 'ds': ds_8}

        save_field(ds_in8, C_sq_prof_8D)
        save_field(ds_in8, C_prof_8D)

        ds_8.close()

        C_q_sq_prof_8D = None
        C_q_prof_8D = None





