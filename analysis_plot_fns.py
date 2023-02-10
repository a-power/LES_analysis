import numpy as np
import matplotlib.pyplot as plt
import dynamic_functions as dyn
import mask_cloud_vs_env as clo
import numpy.ma as ma


def negs_in_field(plotdir, field, data_field_list, data_cl_list):
    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']


    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_cl_list[i])

        data_field = data_field_list[i][f'{field}'].data[...]
        print(np.shape(data_field[0,...]))

        data_field_cloud = np.mean(ma.masked_array(data_field, mask=cloud_only_mask), axis=0)
        data_field_env = np.mean(ma.masked_array(data_field, mask=env_only_mask), axis=0)

        print(np.shape(data_field_env))

        counter_env = np.zeros(len(data_field_env[0, 0, :]))
        counter_cloud = np.zeros(len(data_field_cloud[0,0,:]))
        for j in range(len(data_field_cloud[0,0,:])):
            counter_cloud[j] = np.count_nonzero(data_field_cloud[:,:,j] < 0)
            counter_env[j] = np.count_nonzero(data_field_env[:, :, j] < 0)

        plt.figure(figsize=(7, 6))
        plt.hist([counter_env, counter_cloud], bins=12, histtype='bar', stacked=True, label=["environment", "in-cloud"])
        plt.legend()

        og_xtic = plt.xticks()
        plt.xticks(og_xtic[0],
                   np.round(np.linspace((0) * (20 / 480), (151) * (20 / 480), len(og_xtic[0])), 1))

        plt.xlabel("$z/z_{ML}$", fontsize=16)
        plt.ylabel("number of negative values", fontsize=16)
        plt.savefig(plotdir + f'neg_{field}_vs_z_{deltas[i]}.png', pad_inches=0)
        plt.clf()

        print(f'plotted neg vs z for {field}')

    plt.close('all')


def C_values(plotdir, field, data_field_list, data_cl_list, **kwargs):
    deltas = ['2D', '4D', '8D']#, '16D', '32D', '64D']


    for i in range(len(data_field_list)):

        cloud_only_mask, env_only_mask = clo.cloud_vs_env_masks(data_cl_list[i])

        data_field = data_field_list[i][f'{field}'].data[...]
        print(np.shape(data_field[...]))

        data_field_cloud = ma.masked_array(data_field, mask=cloud_only_mask)
        data_field_env = ma.masked_array(data_field, mask=env_only_mask)

        print(np.shape(data_field_env))

        if field=='Cs':
            name=field
            scalar='$C_{s}$'
        elif field == 'C_theta':
            name=field
            scalar = '$C_{\\theta}$'
        elif field == 'C_q':
            name=field
            scalar = '$C_{qt}$'

        elif field == 'f(LM_field_on_w)_r':
            scalar = '$L_{ij}M_{ij}$'
            name = 'LM'
        elif field == 'f(HR_th_field_on_w)_r':
            scalar = '$H_{j}R_{j \\theta}$'
            name = 'HR_th'
        elif field == 'f(HR_q_total_field_on_w)_r':
            scalar = '$H_{j}R_{j qt}$'
            name = 'HR_q_total'

        plt.figure(figsize=(7, 6))
        plt.hist([data_field_env[...,0:24].flatten(), data_field_env[...,24:151].flatten(), data_field_cloud[...].flatten()], \
                 bins=20, histtype='bar', stacked=True, label=["ML", "CL: clear sky", "CL: cloudy"])
        plt.legend()

        # og_xtic = plt.xticks()

        plt.xlabel(f"{scalar}", fontsize=16)
        plt.ylabel("number of value occurrences", fontsize=16)
        plt.savefig(plotdir + f'dist_of_{name}_values_{deltas[i]}.png', pad_inches=0)
        plt.clf()

        print(f'plotted for {field} {deltas[i]}')

    plt.close('all')



def plotfield(plotdir, field, x_or_y, axis_set, data_field_list, set_percentile, data_cl_list, t_av_or_not):

    deltas = ['2D', '4D', '8D', '16D', '32D', '64D']

    for i in range(len(data_field_list)):

        if field == 'Cs_field':
            print('length of time array for LM is ', len(data_field_list[i]['f(LM_field_on_w)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                LM_field = np.mean(data_field_list[i]['f(LM_field_on_w)_r'].data[...], axis=0)
                MM_field = np.mean(data_field_list[i]['f(MM_field_on_w)_r'].data[...], axis=0)
            else:
                LM_field = data_field_list[i]['f(LM_field_on_w)_r'].data[...]
                MM_field = data_field_list[i]['f(MM_field_on_w)_r'].data[...]

            data_field_sq = 0.5 * LM_field/MM_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cth_field':
            print('length of time array for HR_th is ', len(data_field_list[i]['f(HR_th_field_on_w)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_field_list[i]['f(HR_th_field_on_w)_r '].data[...], axis=0)
                RR_field = np.mean(data_field_list[i]['f(RR_th_field_on_w)_r'].data[...], axis=0)
            else:
                HR_field = data_field_list[i]['f(HR_th_field_on_w)_r'].data[...]
                RR_field = data_field_list[i]['f(RR_th_field_on_w)_r'].data[...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        elif field == 'Cqt_field':
            print('length of time array for HR_qt is ', len(data_field_list[i]['f(HR_q_total_field_on_w)_r'].data[:,0,0,0]))
            if t_av_or_not == 'yes':
                HR_field = np.mean(data_field_list[i]['f(HR_q_total_field_on_w)_r'].data[...], axis=0)
                RR_field = np.mean(data_field_list[i]['f(RR_q_total_field_on_w)_r'].data[...], axis=0)
            else:
                HR_field = data_field_list[i]['f(HR_q_total_field_on_w)_r'].data[...]
                RR_field = data_field_list[i]['f(RR_q_total_field_on_w)_r'].data[...]

            data_field_sq = 0.5*HR_field/RR_field
            data_field = dyn.get_Cs(data_field_sq)

        else:
            print(f'length of time array for {field} is ', len(data_field_list[i][f'{field}'].data[:, 0, 0, 0]))
            if t_av_or_not == 'yes':
                data_field = np.mean(data_field_list[i][f'{field}'].data[...], axis = 0)
            else:
                data_field = data_field_list[i][f'{field}'].data[...]


        print('length of time array for cloud field is ', len(data_cl_list[i]['f(f(q_cloud_liquid_mass_on_w)_r_on_w)_r'].data[:, 0, 0, 0]))
        if t_av_or_not == 'yes':
            cloud_field = np.mean(data_cl_list[i]['f(f(q_cloud_liquid_mass_on_w)_r)_on_w)_r'].data[...], axis = 0)
        else:
            cloud_field = data_cl_list[i]['f(f(q_cloud_liquid_mass_on_w)_r_on_w)_r'].data[...]


        if t_av_or_not == 'yes':

            plt.figure(figsize=(16,5))
            plt.title(f'{field}', fontsize=16)
            if x_or_y == 'x':

                if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                    myvmin = 0
                else:
                    myvmin = np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
                myvmax = np.percentile(data_field[axis_set, :, 5:120], set_percentile[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field[axis_set, :, :]), levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                #cb.set_label(f'{field}', size=16)

                plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='red', linewidths=2, levels=[1e-5])
                plt.xlabel(f'y (cross section with x = {axis_set}) (km)', fontsize=16)

            elif x_or_y == 'y':

                if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                    myvmin = 0
                else:
                    myvmin = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[0])
                myvmax = np.percentile(data_field[50:351, axis_set, 5:120], set_percentile[1])

                mylevels = np.linspace(myvmin, myvmax, 8)

                cf = plt.contourf(np.transpose(data_field[50:351, axis_set, 0:101]), levels=mylevels, extend='both')
                cb = plt.colorbar(cf, format='%.2f')
                cb.set_label(f'{field}', size=16)

                plt.contour(np.transpose(cloud_field[50:351, axis_set, 0:101]), colors='red', linewidths=2, levels=[1e-5])
                plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
            else:
                print("axis_set must be 'x' or'y'.")
            plt.ylabel("z (km)", fontsize=16)
            og_xtic = plt.xticks()
            plt.xticks(og_xtic[0], np.linspace(1, 7, len(og_xtic[0])))
            og_ytic = plt.yticks()
            plt.yticks(np.linspace(0, 101, 5) , np.linspace(0, 2, 5)) # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

            plt.savefig(plotdir+f'zoomed_{field}_{deltas[i]}_tav_{x_or_y}={axis_set}.png', pad_inches=0)
            plt.clf()

            if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
                plt.figure(figsize=(20, 5))
                plt.title(f'{field}$^2$', fontsize=16)
                if x_or_y == 'x':

                    myvmin = 0 #np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
                    myvmax = np.percentile(data_field_sq[axis_set, :, 5:120], set_percentile[1])

                    mylevels = np.linspace(myvmin, myvmax, 8)

                    cf = plt.contourf(np.transpose(data_field_sq[axis_set, :, :]), levels=mylevels, extend='both')
                    cb = plt.colorbar(cf, format='%.2f')
                    #cb.set_under('k')
                    cb.set_label(f'{field}$^2$', size=16)

                    plt.contour(np.transpose(cloud_field[axis_set, :, :]), colors='red', linewidths=2, levels=[1e-5])
                    plt.xlabel(f'y (cross section with x = {axis_set}) (km)')

                elif x_or_y == 'y':

                    myvmin = 0 #np.percentile(data_field[:, axis_set, 5:120], set_percentile[0])
                    myvmax = np.percentile(data_field_sq[:, axis_set, 5:120], set_percentile[1])

                    mylevels = np.linspace(myvmin, myvmax, 8)

                    cf = plt.contourf(np.transpose(data_field_sq[:, axis_set, :]), levels=mylevels, extend='both')
                    cb = plt.colorbar(cf, format='%.2f')
                    #cb.set_under('k')
                    cb.set_label(f'{field}$^2$', size=16)

                    plt.contour(np.transpose(cloud_field[:, axis_set, :]), colors='red', linewidths=2, levels=[1e-5])
                    plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
                else:
                    print("axis_set must be 'x' or 'y'.")

                og_xtic = plt.xticks()
                plt.xticks(og_xtic[0],np.linspace(0, 16, len(og_xtic[0])))
                og_ytic = plt.yticks()
                plt.yticks(np.linspace(0, 151, 7) ,np.linspace(0, 3, 7))
                plt.ylabel("z (km)")
                plt.savefig(plotdir + f'{field}_sq_{deltas[i]}_tav_{x_or_y}={axis_set}.png', pad_inches=0)
                plt.clf()

        else:
            print(t_av_or_not)
            for t in range(len(t_av_or_not)):

                plt.figure(figsize=(16, 5))
                plt.title(f'{field}', fontsize=16)
                if x_or_y == 'x':

                    if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                        myvmin = 0
                    else:
                        myvmin = np.percentile(data_field[t, axis_set, :, 5:120], set_percentile[0])
                    myvmax = np.percentile(data_field[t, axis_set, :, 5:120], set_percentile[1])

                    mylevels = np.linspace(myvmin, myvmax, 8)

                    cf = plt.contourf(np.transpose(data_field[t, axis_set, :, :]), levels=mylevels, extend='both')
                    cb = plt.colorbar(cf, format='%.2f')
                    # cb.set_label(f'{field}', size=16)

                    plt.contour(np.transpose(cloud_field[t, axis_set, :, :]), colors='red', linewidths=2, levels=[1e-5])
                    plt.xlabel(f'y (cross section with x = {axis_set}) (km)', fontsize=16)

                elif x_or_y == 'y':

                    if field == 'LM_field' or field == 'HR_th_field' or field == 'HR_qt_field':
                        myvmin = 0
                    else:
                        myvmin = np.percentile(data_field[t, 50:351, axis_set, 5:120], set_percentile[0])
                    myvmax = np.percentile(data_field[t, 50:351, axis_set, 5:120], set_percentile[1])

                    mylevels = np.linspace(myvmin, myvmax, 8)

                    cf = plt.contourf(np.transpose(data_field[t, 50:351, axis_set, 0:101]), levels=mylevels, extend='both')
                    cb = plt.colorbar(cf, format='%.2f')
                    cb.set_label(f'{field}', size=16)

                    plt.contour(np.transpose(cloud_field[t, 50:351, axis_set, 0:101]), colors='red', linewidths=2,
                                levels=[1e-5])
                    plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
                else:
                    print("axis_set must be 'x' or'y'.")
                plt.ylabel("z (km)", fontsize=16)
                og_xtic = plt.xticks()
                plt.xticks(og_xtic[0], np.linspace(1, 7, len(og_xtic[0])))
                og_ytic = plt.yticks()
                plt.yticks(np.linspace(0, 101, 5),
                           np.linspace(0, 2, 5))  # plt.yticks(np.linspace(0, 151, 7) , np.linspace(0, 3, 7))

                plt.savefig(plotdir + f'zoomed_{field}_{deltas[i]}_t{t}_{x_or_y}={axis_set}.png', pad_inches=0)
                plt.clf()

                if field == 'Cqt_field' or field == 'Cth_field' or field == 'Cs_field':
                    plt.figure(figsize=(20, 5))
                    plt.title(f'{field}$^2$', fontsize=16)
                    if x_or_y == 'x':

                        myvmin = 0  # np.percentile(data_field[axis_set, :, 5:120], set_percentile[0])
                        myvmax = np.percentile(data_field_sq[t, axis_set, :, 5:120], set_percentile[1])

                        mylevels = np.linspace(myvmin, myvmax, 8)

                        cf = plt.contourf(np.transpose(data_field_sq[t, axis_set, :, :]), levels=mylevels, extend='both')
                        cb = plt.colorbar(cf, format='%.2f')
                        # cb.set_under('k')
                        cb.set_label(f'{field}$^2$', size=16)

                        plt.contour(np.transpose(cloud_field[t, axis_set, :, :]), colors='red', linewidths=2,
                                    levels=[1e-5])
                        plt.xlabel(f'y (cross section with x = {axis_set}) (km)')

                    elif x_or_y == 'y':

                        myvmin = 0  # np.percentile(data_field[:, axis_set, 5:120], set_percentile[0])
                        myvmax = np.percentile(data_field_sq[t, :, axis_set, 5:120], set_percentile[1])

                        mylevels = np.linspace(myvmin, myvmax, 8)

                        cf = plt.contourf(np.transpose(data_field_sq[t, :, axis_set, :]), levels=mylevels, extend='both')
                        cb = plt.colorbar(cf, format='%.2f')
                        # cb.set_under('k')
                        cb.set_label(f'{field}$^2$', size=16)

                        plt.contour(np.transpose(cloud_field[t, :, axis_set, :]), colors='red', linewidths=2,
                                    levels=[1e-5])
                        plt.xlabel(f'x (cross section with y = {axis_set}) (km)')
                    else:
                        print("axis_set must be 'x' or 'y'.")

                    og_xtic = plt.xticks()
                    plt.xticks(og_xtic[0], np.linspace(0, 16, len(og_xtic[0])))
                    og_ytic = plt.yticks()
                    plt.yticks(np.linspace(0, 151, 7), np.linspace(0, 3, 7))
                    plt.ylabel("z (km)")
                    plt.savefig(plotdir + f'{field}_sq_{deltas[i]}_t{t}_{x_or_y}={axis_set}.png', pad_inches=0)
                    plt.clf()


        print(f'plotted fields for {field}')

    plt.close('all')
