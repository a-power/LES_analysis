import dynamic_script as dy_s
from subfilter.io.dataout import save_field
import os
import xarray as xr
import mask_cloud_vs_env as clo

av_type = 'all'
mygrid = 'w'

path20f = '/work/scratch-pw/apower/20m_gauss_dyn/corrected_fields/'
file20 = "BOMEX_m0020_g0800_all_14400_gaussian_filter_"
data_path = path20f+file20


options = {
    'return_fields': True,

    'cloud_thres': 10 ** (-5),

    'other_var_choice': False,

    'other_var_thres': False,

    'less_greater': 'greater',

    'my_and_or': 'and',

    'return_all_masks': False,

    'dataset_in': path20f+file20
}


deltas = ['2D', '4D', '8D', '16D', '32D', '64D']


dataset_name = [data_path+'cloud_v_env_2D.nc', data_path+'cloud_v_env_4D.nc', data_path+'cloud_v_env_8D.nc', \
                 data_path+'cloud_v_env_16D.nc', data_path+'cloud_v_env_32D.nc', data_path+'cloud_v_env_64D.nc',]


for i, delta_in in enumerate(deltas):

    Cs_cloud_prof, Cs_env_prof, Cth_cloud_prof, Cth_env_prof, Cqt_cloud_prof, Cqt_env_prof, \
    LM_cloud_av, LM_env_av, MM_cloud_av, MM_env_av, \
    HR_th_cloud_av, HR_th_env_av, RR_th_cloud_av, RR_th_env_av, \
    HR_qt_cloud_av, HR_qt_env_av, RR_qt_cloud_av, RR_qt_env_av, \
    LijMij_cloud, LijMij_env, MijMij_cloud, MijMij_env, \
    HjRj_th_cloud, HjRj_th_env, RjRj_th_cloud, RjRj_th_env, \
    HjRj_qt_cloud, HjRj_qt_env, RjRj_qt_cloud, RjRj_qt_env = clo.get_masked_fields(delta = delta_in, res_count = i, **options)


    ds = xr.Dataset()
    ds.to_netcdf(dataset_name[i], mode='w')
    ds_in = {'file': dataset_name[i], 'ds':ds}


    save_field(ds_in, Cs_cloud_field)
    save_field(ds_in, Cs_env_field)
    save_field(ds_in, Cth_cloud_field)
    save_field(ds_in, Cth_env_field)
    save_field(ds_in, Cqt_cloud_field)
    save_field(ds_in, Cth_env_field)
    save_field(ds_in, Cs_cloud_prof)
    save_field(ds_in, Cs_env_prof)
    save_field(ds_in, Cth_cloud_prof)
    save_field(ds_in, Cth_env_prof)
    save_field(ds_in, Cqt_cloud_prof)
    save_field(ds_in, Cqt_env_prof)

    ds.close()

    Cs_cloud_field = None
    Cs_env_field = None
    Cth_cloud_field = None
    Cth_env_field = None
    Cqt_cloud_field = None
    Cth_env_field = None
    Cs_cloud_prof = None
    Cs_env_prof = None
    Cth_cloud_prof = None
    Cth_env_prof = None
    Cqt_cloud_prof = None
    Cqt_env_prof = None
