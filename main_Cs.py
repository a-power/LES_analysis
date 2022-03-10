import time_av_dynamic as t_dy
import matplotlib.pyplot as plt
import numpy as np

av_type = 'all'

plotdir = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/plots/dyn/'
path20f = '/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn_w/'
file20 = "BOMEX_m0020_g0800_all_14400_filter_"

data_2D = path20f+file20+str('ga00.nc')
data_4D = path20f+file20+str('ga01.nc')

Cs_2D_prof_t0 = indiv_Cs(data_2D, dx=20, dx_hat=40, t_in=0)
Cs_2D_prof_t1 = indiv_Cs(data_2D, dx=20, dx_hat=40, t_in=1)
Cs_2D_prof_t2 = indiv_Cs(data_2D, dx=20, dx_hat=40, t_in=2)
Cs_2D_av, times_2D = t_dy.time_av_Cs(data_2D, dx=20, dx_hat=40,  Cs_av_method = av_type)
Cs_4D_prof_t0 = indiv_Cs(data_4D, dx=20, dx_hat=80, t_in=0)
Cs_4D_prof_t1 = indiv_Cs(data_4D, dx=20, dx_hat=80, t_in=1)
Cs_4D_prof_t2 = indiv_Cs(data_4D, dx=20, dx_hat=80, t_in=2)
Cs_4D_av, times_4D = t_dy.time_av_Cs(data_4D, dx=20, dx_hat=80,  Cs_av_method = av_type)


#########################plots#########################

if times_2D.all() == times_4D.all():

    z = np.arange(0,3020,20)


    y = 200
    levels1 = np.linspace(0, 0.4, 6)

    # cm1 = plt.contourf(np.transpose(Cs_2D_av_field[:, y, :]), levels1, extend='both')
    # cb1 = plt.colorbar(cm1)
    # plt.title(f'Cs 2D averaged over times: {times_2D}')
    # plt.xlabel("x")
    # plt.ylabel("z")
    # cb1.set_label("$C_{s}$", size=12)
    # plt.savefig(plotdir + "Cs_2D_cross_sec_y=" + str(y) + "_t_av.png", pad_inches=0)
    #
    #
    # cm2 = plt.contourf(np.transpose(Cs_4D_av_field[:, y, :]), levels1, extend='both')
    # #cb2 = plt.colorbar(cm2)
    # plt.title(f'Cs 4D averaged over times: {times_4D}')
    # plt.xlabel("x")
    # plt.ylabel("z")
    # #cb2.set_label("$C_{s}$", size=12)
    # plt.savefig(plotdir + "Cs_4D_cross_sec_y=" + str(y) + "_t_av.png", pad_inches=0)


    fig = plt.figure(figsize=(10, 8))
    plt.plot(Cs_2D_prof_t0, z / 500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t0, z / 500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_av, z, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_av, z, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
    plt.xlim(-0.01, 0.2)
    plt.title(f'Cs averaged over times: {times_2D}')
    plt.ylabel('z (m)')
    plt.xlabel('$ C_{s} $', fontsize=14)
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig(plotdir + "Cs_profiles_t_av_line_" + str(av_type) + ".png", pad_inches=0)

    fig2 = plt.figure(figsize=(10, 8))
    plt.plot(Cs_2D_prof_t0, z/500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t0, z/500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
    plt.xlim(-0.01, 0.2)
    plt.title(f'Cs averaged over times: {times_2D}')
    plt.ylabel('$z/z_i')
    plt.xlabel('$ C_{s} $', fontsize=14)
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig(plotdir + "Cs_profiles_t_av_line_scaled_" + str(av_type) + ".png", pad_inches=0)

    fig3 = plt.figure(figsize=(10, 8))
    plt.plot(Cs_2D_prof_t0, z / 500, 'r-', markersize=6, label='$t0 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t0, z / 500, 'r-*', markersize=6, label='$t0 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t1, z / 500, 'g-', markersize=6, label='$t1 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t1, z / 500, 'g-*', markersize=6, label='$t1 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_prof_t2, z / 500, 'b-', markersize=6, label='$t2 C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_prof_t2, z / 500, 'b-*', markersize=6, label='$t2 C_{s 4 \\Delta} $')
    plt.plot(Cs_2D_av, z / 500, 'k-', markersize=6, label='$Average C_{s 2 \\Delta} $')
    plt.plot(Cs_4D_av, z / 500, 'k-*', markersize=6, label='$Average C_{s 4 \\Delta} $')
    plt.xlim(-0.01, 0.2)
    plt.ylim(0, 2)
    plt.title(f'Cs averaged over times: {times_2D}')
    plt.ylabel('$z/z_i$')
    plt.xlabel('$ C_{s} $', fontsize=14)
    plt.legend(fontsize=16, loc='upper right')
    plt.savefig(plotdir + "Cs_profiles_t_av_line_scaled_zoomed_" + str(av_type) + ".png", pad_inches=0)

else:
    print('times for 2Delta data = ', times_2D, 'times for 4Delta data = ', times_4D)