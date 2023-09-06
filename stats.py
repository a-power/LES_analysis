import numpy as np
import mask_cloud_vs_env as clo
import numpy.ma as ma
import dynamic_functions as dyn
import xarray as xr
import os
import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--case_in', type=str, default='ARM')
parser.add_argument('--time_it', type=int, default=0)

args = parser.parse_args()
case = args.case_in

mygrid = 'p'
Deltas = ['0', '1', '2', '3', '4', '5']
beta_filt_num = ['0']

if case == 'BOMEX':
    data_path = '/storage/silver/MONC_data/Alanna/BOMEX/beta_filtered_data/smoothed_LM_HR_fields/'
    times_analysed = [ '14400' ]
    set_time = times_analysed[args.time_it]
    file_name = f'BOMEX_m0020_g0800_all_{set_time}_gaussian_filter_'

    zn_set = np.arange(0, 3020, 20)

    # todd_dir = '/gws/nopw/j04/paracon_rdg/users/toddj/updates_suite/BOMEX_m0020_g0800/diagnostic_files/'
    # prof_file = todd_dir + 'BOMEX_m0020_g0800_all_14400.nc'

    z_cl_r_ind_set = [33, 77] # set
    z_cl_r_ind_calc = [22, 109]  # calc

    z_ML_r_ind = [10, 22] # 22 as calc but the profs


if case == 'ARM':
    data_path = '/work/scratch-pw3/apower/ARM/corrected_sigmas/filtering_filtered/smoothed_LM_HR_fields/'
    times_analysed = [ '18000', '25200', '32400', '39600' ]
    set_time = times_analysed[args.time_it]
    file_name = f'diagnostics_3d_ts_{set_time}_gaussian_filter_'

    zn_set = np.arange(0, 4410, 10)
    z_ML_bottom = 20

    z_cl_r_ind_set_list = [ [87, 110], [102, 150], [115, 200], [130, 230] ] #z_cl_range_calc
    z_ml_r_ind_list = [ [20, 75], [20, 80], [20, 85], [20, 90] ]

    z_cl_r_ind_set = z_cl_r_ind_set_list[args.time_it]
    z_ml_r_ind_set = z_ml_r_ind_list[args.time_it]

    profiles_dir = f'/work/scratch-pw3/apower/ARM/MONC_out/diagnostics_ts_{set_time}.nc'



w_field = f'f(f(w_on_{mygrid})_r_on_{mygrid})_r'
w2_field = f'f(f(w_on_{mygrid}.w_on_{mygrid})_r_on_{mygrid})_r'
th_v_field = f'f(f(th_v_on_{mygrid})_r_on_{mygrid})_r'


mask_options = {'cloud_thres': 1e-7,
                   'other_vars': [w_field, th_v_field],
                    'other_var_thres': [0.5, 0],
                    'less_greater_in': ['less', 'less'],
                    'and_or_in': ['and', 'and']
                    }



def calc_z_ML_and_CL(file_path, time_stamp=-1):

    prof_data = xr.open_dataset(file_path)

    wth_prof = prof_data['wtheta_cn_mean'].data[time_stamp, ...]
    wth_prof_list = wth_prof.tolist()
    z_ML = wth_prof_list.index(np.min(wth_prof))

    z_cloud = prof_data['total_cloud_fraction'].data[time_stamp, ...]
    z_cloud_where = np.where(z_cloud > 1e-7)
    z_ind = np.arange(0, len(z_cloud))
    z_cloud_ind = z_ind[z_cloud_where]
    #print('z_cloud_ind =', z_cloud_ind)
    z_min_CL = np.min(z_cloud_ind)
    z_max_CL = np.max(z_cloud_ind)
    z_CL = [ z_min_CL, z_max_CL ]

    zn_out = prof_data['zn'].data[...] # timeless parameter?

    return z_ML, z_CL, zn_out




if case == 'ARM':
    z_ML_top_index, z_cl_r_ind_calc, zn_arr = calc_z_ML_and_CL(profiles_dir)
    z_ML_r_ind = [z_ML_bottom, z_ML_top_index]
elif case== 'BOMEX':
    #z_cl_r_ind_calc set
    z_ML_top_index = z_ML_r_ind[1]
else:
    print('case ', case, ' is undefined')

z_ML = zn_set[z_ML_top_index]

z_cl_r_set_m = [ zn_set[z_cl_r_ind_set[0]],  zn_set[z_cl_r_ind_set[1]] ]
z_cl_range_calc_m = [ zn_set[z_cl_r_ind_calc[0]], zn_set[z_cl_r_ind_calc[1]] ]




def get_masks(dataset_path, file_name_in, Delta_in, beta_filt,
              cloud_thres, other_vars, other_var_thres,
                             less_greater_in, and_or_in, grid=mygrid):

    contour_field_in = dataset_path + file_name_in + \
        f'ga0{Delta_in}_gaussian_filter_ga0{beta_filt}_running_mean_filter_rm00.nc'

    cloud_only_mask, env_only_mask = \
        clo.cloud_vs_env_masks(contour_field_in, cloud_liquid_threshold=cloud_thres)


    cloud_up_mask, cloud_core_mask = clo.cloudy_and_or(contour_field_in, other_var=other_vars, var_thres=other_var_thres,
                                                                 less_greater=less_greater_in, and_or=and_or_in,
                                                                 cloud_liquid_threshold=cloud_thres, grid=grid)

    return cloud_only_mask, env_only_mask, cloud_up_mask, cloud_core_mask



def get_stats_for_C(dataset_path, file_name_in, Delta_in, beta_filt, param, ML_r_int, CL_depth_int_calc, CL_depth_int_set,
              cloud_only_mask_in, env_only_mask_in, cloud_up_mask_in, cloud_core_mask_in, grid=mygrid):

    csv_file_path = dataset_path + 'stats/'
    os.makedirs(csv_file_path, exist_ok=True)

    dataset_in = dataset_path + file_name_in + param + '_' + Delta_in + '_' + beta_filt + '_running_mean_filter_rm00.nc'


    if param == 'Cs':
        field_name = '$C_s$'
        field_name_sq = '$C_s^2$'

    elif param == 'C_th':
        field_name = '$C_{\\theta}$'
        field_name_sq = '$C_{\\theta}^2$'

    elif param == 'C_qt':
        field_name = '$C_{qt}$'
        field_name_sq = '$C_{qt}^2$'


    data_set = xr.open_dataset(dataset_in)

    time_data = data_set['time']
    times = time_data.data
    nt = len(times)
    x_data = data_set['x_p']
    x_s = x_data.data
    y_data = data_set['y_p']
    y_s = y_data.data
    z_data = data_set['z']
    z_s = z_data.data
    zn_data = data_set['zn']
    zn_s = zn_data.data


    if param == 'Cs':

        if f'f(LM_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(LM_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(MM_field_on_{grid})_r'].data[...]
        elif f'LM_field' in data_set:
            num_field = data_set[f'LM_field'].data[...]
            den_field = data_set[f'MM_field'].data[...]
        else:
            print('LM_field_on_{grid} not in file')

    elif param == 'C_th':

        if f'f(HR_th_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_th_field_on_{grid})_r'].data[...]
        elif f'HR_th_field' in data_set:
            num_field = data_set[f'HR_th_field'].data[...]
            den_field = data_set[f'RR_th_field'].data[...]
        else:
            print('HR_field_on_{grid} not in file')


    elif param == 'C_qt':


        if f'f(HR_q_total_f_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(HR_q_total_f_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_q_total_f_field_on_{grid})_r'].data[...]
        elif f'HR_q_total_f_field' in data_set:
            num_field = data_set[f'HR_q_total_f_field'].data[...]
            den_field = data_set[f'RR_q_total_f_field'].data[...]
        else:
            print('HR_field_on_{grid} not in file')


    #num_field is the entire domain dataset
    #den_field is the entire domain dataset

    num_field_env = ma.masked_array(num_field, mask=env_only_mask_in)
    den_field_env = ma.masked_array(den_field, mask=env_only_mask_in)

    num_field_IC = ma.masked_array(num_field, mask=cloud_only_mask_in)
    den_field_IC = ma.masked_array(den_field, mask=cloud_only_mask_in)

    num_field_CU = ma.masked_array(num_field, mask=cloud_up_mask_in)
    den_field_CU = ma.masked_array(den_field, mask=cloud_up_mask_in)

    num_field_CC = ma.masked_array(num_field, mask=cloud_core_mask_in)
    den_field_CC = ma.masked_array(den_field, mask=cloud_core_mask_in)




    dom_num = num_field[...]
    dom_den = den_field[...]
    C_sq_dom = 0.5*(dom_num / dom_den)
    C_dom = dyn.get_Cs( C_sq_dom )

    ML_num = num_field_env[:, :, :, ML_r_int[0] :: ML_r_int[1] ]
    ML_den = den_field_env[:, :, :, ML_r_int[0] :: ML_r_int[1] ]
    C_sq_ML = 0.5*(ML_num / ML_den)
    C_ML = dyn.get_Cs( C_sq_ML )

    CL_num_calc = num_field[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    CL_den_calc = den_field[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    C_sq_CL_calc = 0.5*(CL_num_calc / CL_den_calc)
    C_CL_calc = dyn.get_Cs( C_sq_CL_calc )

    CL_num_set = num_field[:, :, :, CL_depth_int_set[0] :: CL_depth_int_set[1] ]
    CL_den_set = den_field[:, :, :, CL_depth_int_set[0] :: CL_depth_int_set[1] ]
    C_sq_CL_set = 0.5*(CL_num_set / CL_den_set)
    C_CL_set = dyn.get_Cs( C_sq_CL_set )

    IC_num = num_field_IC[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    IC_den = den_field_IC[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    C_sq_IC = 0.5*(IC_num / IC_den)
    C_IC = dyn.get_Cs( C_sq_IC )

    CU_num = num_field_CU[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    CU_den = den_field_CU[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    C_sq_CU = 0.5*(CU_num / CU_den)
    C_CU = dyn.get_Cs( C_sq_CU )

    CC_num = num_field_CC[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    CC_den = den_field_CC[:, :, :, CL_depth_int_calc[0] :: CL_depth_int_calc[1] ]
    C_sq_CC = 0.5*(CC_num / CC_den)
    C_CC = dyn.get_Cs( C_sq_CC )
    


    C_partiton = [C_dom, C_ML, C_CL_calc, C_CL_set, C_IC, C_CU, C_CC,
                  C_sq_dom, C_sq_ML, C_sq_CL_calc, C_sq_CL_set, C_sq_IC, C_sq_CU, C_sq_CC]

    partition_name = ['domain', 'ML', 'CL_calc', 'CL_set', 'IC', 'CU', 'CC']
    header = ['Smagorinsky_Parameter', 'Partition_Condition', 'Layer_Range', 'Mean', 'Standard_Deviation',
              'Median', 'Lower_Quartile', 'Upper_Quartile', 'Minimum', 'Maximum', 'Number_of_Points']

    CL_range_name = str(CL_depth_int_set[0]) + '_' + str(CL_depth_int_set[1])



    file_csv = open(csv_file_path + f'{smag}_{times[-1]}_delta_{Delta_in}_CL_{CL_range_name}', 'w')
    C_stats = csv.writer(file_csv)

    for it, C_part_in in enumerate(C_partiton):

        if it < len(partition_name):
            param_name = param
        else:
            param_name = param+'_sq'


        if partition_name[it] == 'CL_set':
            range_str = str(CL_depth_int_set[0]) + '_' + str(CL_depth_int_set[1])

        elif partition_name[it] == 'IC' or partition_name[it] == 'CU'  or partition_name[it] == 'CC' \
                or partition_name[it] == 'CL_calc':
            range_str = str(CL_depth_int_calc[0]) + '_' + str(CL_depth_int_calc[1])

        elif partition_name[it] == 'ML':
            range_str = str(ML_r_int[0]) + '_' + str(ML_r_int[1])

        else:
            range_str = 'all_domain'


        C_mean = C_part_in.mean()
        C_st_dev = C_part_in.std()
        C_med = C_part_in.median()
        C_25 = np.percentile( C_part_in.compressed(), 25)
        C_75 = np.percentile( C_part_in.compressed(), 75)
        C_min = C_part_in.max()
        C_max = C_part_in.min()
        C_n = len( C_part_in.compressed() )

        row = [param_name, partition_name[it % 6], range_str, C_mean,
               C_st_dev, C_med, C_25, C_75, C_min, C_max, C_n]


        C_stats.writerow(row)

    file_csv.close()


for it_d, delta in enumerate(Deltas):
    for it_b, beta_in in enumerate(beta_filt_num):


        cloud_only_mask, env_only_mask, cloud_up_mask, cloud_core_mask = \
            get_masks(data_path, file_name, delta, beta_in, **mask_options)


        for smag, it_c in enumerate(['Cs', 'C_th', 'C_qt']):

            get_stats_for_C(data_path, file_name, delta, beta_in, smag, z_ML_r_ind, z_cl_r_ind_calc, z_cl_r_ind_set,
                            cloud_only_mask, env_only_mask, cloud_up_mask, cloud_core_mask)



