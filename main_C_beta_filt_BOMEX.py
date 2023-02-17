import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr

av_type = 'all'
mygrid = 'p'

path20f = '/work/scratch-pw3/apower/20m_gauss_dyn/on_p_grid/beta_filtered_filters/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"

scalar = ['momentum', 'th', 'q_total_f']
set_save_all = 2

data_2D = path20f+file20+str('ga00_gaussian_filter_ga0')
data_4D = path20f+file20+str('ga01_gaussian_filter_ga0')
data_8D = path20f+file20+str('ga02_gaussian_filter_ga0')
data_16D = path20f+file20+str('ga03_gaussian_filter_ga0')
data_32D = path20f+file20+str('ga04_gaussian_filter_ga0')
data_64D = path20f+file20+str('ga05_gaussian_filter_ga0')

dataset_name0_0= [path20f+file20+'Cs_0_0.nc', path20f+file20+'C_th_0_0.nc', path20f+file20+'C_qt_0_0.nc']
dataset_name0_1= [path20f+file20+'Cs_0_1.nc', path20f+file20+'C_th_0_1.nc', path20f+file20+'C_qt_0_1.nc']

dataset_name1_0= [path20f+file20+'Cs_1_0.nc', path20f+file20+'C_th_1_0.nc', path20f+file20+'C_qt_1_0.nc']
dataset_name1_1= [path20f+file20+'Cs_1_1.nc', path20f+file20+'C_th_1_1.nc', path20f+file20+'C_qt_1_1.nc']

dataset_name2_0= [path20f+file20+'Cs_2_0.nc', path20f+file20+'C_th_2_0.nc', path20f+file20+'C_qt_2_0.nc']
dataset_name2_1= [path20f+file20+'Cs_2_1.nc', path20f+file20+'C_th_2_1.nc', path20f+file20+'C_qt_2_1.nc']

dataset_name3_0= [path20f+file20+'Cs_3_0.nc', path20f+file20+'C_th_3_0.nc', path20f+file20+'C_qt_3_0.nc']
dataset_name3_1= [path20f+file20+'Cs_3_1.nc', path20f+file20+'C_th_3_1.nc', path20f+file20+'C_qt_3_1.nc']

dataset_name4_0= [path20f+file20+'Cs_4_0.nc', path20f+file20+'C_th_4_0.nc', path20f+file20+'C_qt_4_0.nc']
dataset_name4_1= [path20f+file20+'Cs_4_1.nc', path20f+file20+'C_th_4_1.nc', path20f+file20+'C_qt_4_1.nc']

dataset_name5_0= [path20f+file20+'Cs_5_0.nc', path20f+file20+'C_th_5_0.nc', path20f+file20+'C_qt_5_1.nc']
dataset_name5_1= [path20f+file20+'Cs_5_1.nc', path20f+file20+'C_th_5_1.nc', path20f+file20+'C_qt_5_1.nc']

dataset_list = [dataset_name0_0, dataset_name0_1, dataset_name1_0, dataset_name1_1, dataset_name2_0, dataset_name2_1, \
                dataset_name3_0, dataset_name3_1, dataset_name4_0, dataset_name4_1, dataset_name5_0, dataset_name5_1]


DX_20_0 = {
    'indir': data_2D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 20,
    'dx_hat': 40
}
DX_20_1 = {
    'indir': data_2D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 20,
    'dx_hat': 80
}
DX_40_0 = {
    'indir': data_4D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 40,
    'dx_hat': 80
}
DX_40_1 = {
    'indir': data_4D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 40,
    'dx_hat': 160
}
DX_80_0 = {
    'indir': data_8D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 80,
    'dx_hat': 160
}
DX_80_1 = {
    'indir': data_8D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 80,
    'dx_hat': 320
}
DX_160_0 = {
    'indir': data_16D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 160,
    'dx_hat': 320
}
DX_160_1 = {
    'indir': data_16D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 160,
    'dx_hat': 640
}
DX_320_0 = {
    'indir': data_32D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 320,
    'dx_hat': 640
}
DX_320_1 = {
    'indir': data_32D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 320,
    'dx_hat': 1280
}
DX_640_0 = {
    'indir': data_64D+str('0.nc'),
    'ingrid': mygrid,
    'dx': 640,
    'dx_hat': 1280
}
DX_640_1 = {
    'indir': data_64D+str('1.nc'),
    'ingrid': mygrid,
    'dx': 640,
    'dx_hat': 2560
}

dict_list = [DX_20_0, DX_20_1, DX_40_0, DX_40_1, DX_80_0, DX_80_1, \
                DX_160_0, DX_160_1, DX_320_0, DX_320_1, DX_640_0, DX_640_1]


for j in range(len(dict_list)):
    for i, scalar_in in enumerate(scalar):

    ########################################################################
        z_save, zn_save, HR_field, RR_field = dy_s.LijMij_fields(scalar=scalar_in, **dict_list[j])

        ds = xr.Dataset()
        ds.to_netcdf(dataset_list[j][i], mode='w')
        ds_in = {'file':dataset_list[j][i], 'ds': ds}

        save_field(ds_in, z_save)
        save_field(ds_in, zn_save)
        save_field(ds_in, HR_field)
        save_field(ds_in, RR_field)

        ds.close()

        z_save = None
        zn_save = None
        HR_field = None
        RR_field = None

        print('ran for ', scalar, 'dictionary = ', dict_list[j])

