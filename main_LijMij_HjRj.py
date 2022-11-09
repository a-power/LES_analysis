import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'w'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')
data_8D = path20f+file20+str('ga02.nc')
data_16D = path20f+file20+str('ga03.nc')
data_32D = path20f+file20+str('ga04.nc')
data_64D = path20f+file20+str('ga05.nc')

dataset_name2 = path20f+file20+'LijMij_HjRj_2D.nc'
dataset_name4 = path20f+file20+'LijMij_HjRj_4D.nc'
dataset_name8 = path20f+file20+'LijMij_HjRj_8D.nc'
dataset_name16 = path20f+file20+'LijMij_HjRj_16D.nc'
dataset_name32 = path20f+file20+'LijMij_HjRj_32D.nc'
dataset_name64 = path20f+file20+'LijMij_HjRj_64D.nc'

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
DX_16D = {
    'indir': data_16D,
    'dx_hat': 320
}
DX_32D = {
    'indir': data_32D,
    'dx_hat': 640
}
DX_64D = {
    'indir': data_64D,
    'dx_hat': 1280
}

for i, scalar_in in enumerate(scalar):

    ########################################################################

    HR_field_2D, RR_field_2D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_2D)

    ds_2 = xr.Dataset()
    ds_2.to_netcdf(dataset_name2, mode='w')
    ds_in2 = {'file': dataset_name2, 'ds': ds_2}

    save_field(ds_in2, HR_field_2D)
    save_field(ds_in2, RR_field_2D)

    ds_2.close()

    HR_field_2D = None
    RR_field_2D = None

    ##########################################

    HR_field_4D, RR_field_4D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_4D)


    ds_4 = xr.Dataset()
    ds_4.to_netcdf(dataset_name4, mode='w')
    ds_in4 = {'file':dataset_name4, 'ds': ds_4}

    save_field(ds_in4, HR_field_4D)
    save_field(ds_in4, RR_field_4D)

    ds_4.close()

    HR_field_4D = None
    RR_field_4D = None


    #########################################################################

    HR_field_8D, RR_field_8D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_8D)

    ds_8 = xr.Dataset()
    ds_8.to_netcdf(dataset_name8, mode='w')
    ds_in8 = {'file':dataset_name8, 'ds': ds_8}

    save_field(ds_in8, HR_field_8D)
    save_field(ds_in8, RR_field_8D)

    ds_8.close()

    HR_field_8D = None
    RR_field_8D = None

    ##########################################

    HR_field_16D, RR_field_16D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_16D)

    ds_16 = xr.Dataset()
    ds_16.to_netcdf(dataset_name16, mode='w')
    ds_in16 = {'file':dataset_name16, 'ds': ds_16}

    save_field(ds_in16, HR_field_16D)
    save_field(ds_in16, RR_field_16D)


    ds_16.close()

    HR_field_16D = None
    RR_field_16D = None

##########################################

    HR_field_32D, RR_field_32D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_32D)

    ds_32 = xr.Dataset()
    ds_32.to_netcdf(dataset_name32, mode='w')
    ds_in32 = {'file': dataset_name32, 'ds': ds_32}

    save_field(ds_in32, HR_field_32D)
    save_field(ds_in32, RR_field_32D)

    ds_32.close()

    HR_field_32D = None
    RR_field_32D = None

    ##########################################

    HR_field_64D, RR_field_64D = dy_s.LijMij_fields(scalar=scalar_in, dx=20, ingrid=mygrid, **DX_64D)

    ds_64 = xr.Dataset()
    ds_64.to_netcdf(dataset_name64, mode='w')
    ds_in64 = {'file': dataset_name64, 'ds': ds_64}

    save_field(ds_in64, HR_field_64D)
    save_field(ds_in64, RR_field_64D)

    ds_64.close()

    HR_field_64D = None
    RR_field_64D = None
