import numpy as np
import matplotlib.pyplot as plt
import dynamic_functions as dyn
import mask_cloud_vs_env as clo
import numpy.ma as ma
import dynamic_functions as dyn
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import xarray as xr
import os
import csv


set_CL_r = []
Deltas = []


def get_masks(contour_field_in, cloud_thres, other_vars, other_var_thres,
                             less_greater_in, and_or_in, grid):

    cloud_only_mask, env_only_mask = \
        clo.cloud_vs_env_masks(contour_field_in, cloud_liquid_threshold=cloud_thres)


    cloud_up_mask, cloud_core_mask = clo.cloudy_and_or(contour_field_in,
                                                                 other_var=other_vars, var_thres=other_var_thres,
                                                                 less_greater=less_greater_in, and_or=and_or_in,
                                                                 cloud_liquid_threshold=cloud_thres, grid=grid)

    return cloud_only_mask, env_only_mask, cloud_up_mask, cloud_core_mask


def get_stats_for_C(dataset_in, param, ML_int, CL_int, CL_depth_int, t_in, grid,
              cloud_only_mask_in, env_only_mask_in, cloud_up_mask_in, cloud_core_mask_in):

    csv_file_path = ''
    Delta_in = ''


    if param == 'Cs':
        field_name = '$C_s$'
        field_name_sq = '$C_s^2$'

    elif param == 'Cth':
        field_name = '$C_{\\theta}$'
        field_name_sq = '$C_{\\theta}^2$'

    elif param == 'Cqt':
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

    elif param == 'Cth':

        if f'f(HR_th_field_on_{grid})_r' in data_set:
            num_field = data_set[f'f(HR_th_field_on_{grid})_r'].data[...]
            den_field = data_set[f'f(RR_th_field_on_{grid})_r'].data[...]
        elif f'HR_th_field' in data_set:
            num_field = data_set[f'HR_th_field'].data[...]
            den_field = data_set[f'RR_th_field'].data[...]
        else:
            print('HR_field_on_{grid} not in file')


    elif param == 'Cqt':


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




    dom_num = num_field[t_in, ...]
    dom_den = den_field[t_in, ...]
    C_sq_dom = 0.5*(dom_num / dom_den)
    C_dom = dyn.get_Cs( C_sq_dom )

    ML_num = num_field_env[t_in, :, :, ML_int[0] :: ML_int[1] ]
    ML_den = den_field_env[t_in, :, :, ML_int[0] :: ML_int[1] ]
    C_sq_ML = 0.5*(ML_num / ML_den)
    C_ML = dyn.get_Cs( C_sq_ML )

    CL_num = num_field[t_in, :, :, CL_depth_int[0] :: CL_depth_int[1] ]
    CL_den = den_field[t_in, :, :, CL_depth_int[0] :: CL_depth_int[1] ]
    C_sq_CL = 0.5*(CL_num / CL_den)
    C_CL = dyn.get_Cs( C_sq_CL )

    IC_num = num_field_IC[t_in, :, :, CL_int[0] :: CL_int[1] ]
    IC_den = den_field_IC[t_in, :, :, CL_int[0] :: CL_int[1] ]
    C_sq_IC = 0.5*(IC_num / IC_den)
    C_IC = dyn.get_Cs( C_sq_IC )

    CU_num = num_field_CU[t_in, :, :, CL_int[0] :: CL_int[1] ]
    CU_den = den_field_CU[t_in, :, :, CL_int[0] :: CL_int[1] ]
    C_sq_CU = 0.5*(CU_num / CU_den)
    C_CU = dyn.get_Cs( C_sq_CU )

    CC_num = num_field_CC[t_in, :, :, CL_int[0] :: CL_int[1] ]
    CC_den = den_field_CC[t_in, :, :, CL_int[0] :: CL_int[1] ]
    C_sq_CC = 0.5*(CC_num / CC_den)
    C_CC = dyn.get_Cs( C_sq_CC )
    


    C_partiton = [C_dom, C_ML, C_CL, C_IC, C_CU, C_CC,
                  C_sq_dom, C_sq_ML, C_sq_CL, C_sq_IC, C_sq_CU, C_sq_CC]

    partition_name = ['domain', 'ML', 'CL', 'IC', 'CU', 'CC']
    header = ['Smagorinsky_Parameter', 'Partition_Condition', 'Layer_Range', 'Mean', 'Standard_Deviation',
              'Median', 'Lower_Quartile', 'Upper_Quartile', 'Minimum', 'Maximum', 'Number_of_Points']

    CL_range_name = str(CL_int[0]) + '_' + str(CL_int[1])



    file_csv = open(csv_file_path + f'{smag}_{times[-1]}_{t_in}_{Delta_in}_CL_{CL_range_name}', 'w')
    C_stats = csv.writer(file_csv)

    for it, C_part_in in enumerate(C_partiton):

        if it < len(partition_name):
            param_name = param
        else:
            param_name = param+'_sq'


        if C_part_in == 'CL':
            range_str = str(CL_depth_int[0]) + '_' + str(CL_depth_int[1])

        elif C_part_in == 'IC' or C_part_in == 'CU'  or C_part_in == 'CC':
            range_str = str(CL_int[0]) + '_' + str(CL_int[1])

        elif C_part_in == 'ML':
            range_str = str(ML_int[0]) + '_' + str(ML_int[1])
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


for delta, it_d in enumerate(Deltas):


    cloud_only_mask, env_only_mask, cloud_up_mask, cloud_core_mask = \
        get_masks(contour_field_in, cloud_thres, other_vars, other_var_thres, less_greater_in, and_or_in, grid)


    for smag, it_c in enumerate(['Cs', 'Cth', 'Cqt']):

        get_stats_for_C(dataset_in, param, ML_int, CL_int, CL_depth_int, t_in, grid,
                        cloud_only_mask_in, env_only_mask_in, cloud_up_mask_in, cloud_core_mask_in)


