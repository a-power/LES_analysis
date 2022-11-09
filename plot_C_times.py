import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import dynamic_functions as dyn

np.seterr(divide='ignore') #ignore divide by zero errors in beta calcs
np.seterr(invalid='ignore')


hrs = ['6', '13']
sec = ['21600', '48600']

for i in range(len(hrs)):

    path20f = f'/work/scratch-pw/apower/20m_gauss_dyn_{hrs[i]}hrs/'
    file20 = f'BOMEX_m0020_g0800_all_{sec[i]}_gaussian_filter_'

    dir_data_Cs = path20f + file20 + 'Cs_'
    dir_data_C_th = path20f + file20 + 'C_th_'
    dir_data_Cq_tot = path20f + file20 + 'C_qt_'
    plotdir = f'/gws/nopw/j04/paracon_rdg/users/apower/LES_analysis/20m_gauss_dyn/plots/{hrs[i]}hrs/'
    os.makedirs(plotdir, exist_ok = True)

    data_2D_s = xr.open_dataset(dir_data_Cs + '2D.nc')
    data_4D_s = xr.open_dataset(dir_data_Cs + '4D.nc')
    data_8D_s = xr.open_dataset(dir_data_Cs + '8D.nc')

    data_2D_th = xr.open_dataset(dir_data_C_th + '2D.nc')
    data_4D_th = xr.open_dataset(dir_data_C_th + '4D.nc')
    data_8D_th = xr.open_dataset(dir_data_C_th + '8D.nc')

    data_2D_qtot = xr.open_dataset(dir_data_Cq_tot + '2D.nc')
    data_4D_qtot = xr.open_dataset(dir_data_Cq_tot + '4D.nc')
    data_8D_qtot = xr.open_dataset(dir_data_Cq_tot + '8D.nc')


    z = np.arange(0, 3020, 20)
    z_i = 490

    #index of 0 at the start is to get rid of the dummy time index thats required to save the files

    Cs_sq_2 = data_2D_s['Cs_sq_prof'].data[0, ...]
    Cs_sq_4 = data_4D_s['Cs_sq_prof'].data[0, ...]
    Cs_sq_8 = data_8D_s['Cs_sq_prof'].data[0, ...]

    Cth_sq_2 = data_2D_th['C_th_sq_prof'].data[0, ...]
    Cth_sq_4 = data_4D_th['C_th_sq_prof'].data[0, ...]
    Cth_sq_8 = data_8D_th['C_th_sq_prof'].data[0, ...]

    Cq_sq_2 = data_2D_qtot['C_q_total_sq_prof'].data[0, ...]
    Cq_sq_4 = data_4D_qtot['C_q_total_sq_prof'].data[0, ...]
    Cq_sq_8 = data_8D_qtot['C_q_total_sq_prof'].data[0, ...]



    Cs_2 = dyn.get_Cs(Cs_sq_2)
    Cs_4 = dyn.get_Cs(Cs_sq_4)
    Cs_8 = dyn.get_Cs(Cs_sq_8)

    Cth_2 = data_2D_th['C_th_prof'].data[0, ...]
    Cth_4 = data_4D_th['C_th_prof'].data[0, ...]
    Cth_8 = data_8D_th['C_th_prof'].data[0, ...]

    Cq_2 = data_2D_qtot['C_q_total_prof'].data[0, ...]
    Cq_4 = data_4D_qtot['C_q_total_prof'].data[0, ...]
    Cq_8 = data_8D_qtot['C_q_total_prof'].data[0, ...]


    ########################################################################################################################


    beta_s = dyn.beta_calc(Cs_sq_2, Cs_sq_4)
    beta_th = dyn.beta_calc(Cth_sq_2, Cth_sq_4)
    beta_q = dyn.beta_calc(Cq_sq_2, Cq_sq_4)

    # plt.figure(figsize=(6,7))
    # plt.plot(beta_s, z)
    # plt.xlabel('$\\beta_s$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.xlim(0, 1)
    # plt.savefig(plotdir+'Cs_beta_prof.png', pad_inches=0)
    #
    # plt.figure(figsize=(6,7))
    # plt.plot(beta_th, z)
    # plt.xlabel('$\\beta_{\\theta}$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.xlim(0, 1)
    # plt.savefig(plotdir+'Cth_beta_prof.png', pad_inches=0)
    #
    # plt.figure(figsize=(6,7))
    # plt.plot(beta_q, z)
    # plt.xlabel('$\\beta_{qt}$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.xlim(0.19, 0.21)
    # plt.savefig(plotdir+'Cq_beta_prof.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(beta_s, z, label='$\\beta_s$')
    plt.plot(beta_th, z, label='$\\beta_{\\theta}$')
    plt.plot(beta_q, z, label='$\\beta_{qt}$')
    plt.legend(fontsize=16, loc='upper right')
    plt.xlabel('$\\beta$', fontsize=14)
    plt.ylabel("z (m)")
    #plt.xlim(0, 1)
    plt.savefig(plotdir+'all_beta_profs.png', pad_inches=0)
    plt.close()



    # plt.figure(figsize=(6,7))
    # plt.plot(beta_s, z/z_i)
    # plt.xlabel('$\\beta_s$', fontsize=14)
    # plt.ylabel("z/z$_{ML}$")
    # plt.xlim(0, 1)
    # plt.savefig(plotdir+'Cs_beta_prof_scaled.png', pad_inches=0)
    #
    # plt.figure(figsize=(6,7))
    # plt.plot(beta_th, z/z_i)
    # plt.xlabel('$\\beta_{\\theta}$', fontsize=14)
    # plt.ylabel("z/z$_{ML}$")
    # plt.xlim(0, 1)
    # plt.savefig(plotdir+'Cth_beta_prof_scaled.png', pad_inches=0)
    #
    # plt.figure(figsize=(6,7))
    # plt.plot(beta_q, z/z_i)
    # plt.xlabel('$\\beta_{qt}$', fontsize=14)
    # plt.ylabel("z/z$_{ML}$")
    # plt.xlim(0.19, 0.21)
    # plt.savefig(plotdir+'Cq_beta_prof_scaled.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(beta_s, z/z_i, label='$\\beta_s$')
    plt.plot(beta_th, z/z_i, label='$\\beta_{\\theta}$')
    plt.plot(beta_q, z/z_i, label='$\\beta_{qt}$')
    plt.legend(fontsize=16, loc='upper right')
    plt.xlabel('$\\beta$', fontsize=14)
    plt.ylabel("z/z$_{ML}$")
    plt.xlim(0, 1)
    plt.savefig(plotdir+'all_beta_profs_scaled.png', pad_inches=0)
    plt.close()


    ###########################################################################################################


    Cs_beta_sq = Cs_sq_2/beta_s
    Cth_beta_sq = Cth_sq_2/beta_th
    Cq_beta_sq = Cq_sq_2/beta_q

    Cs_beta = dyn.get_Cs(Cs_beta_sq)
    Cth_beta = dyn.get_Cs(Cth_beta_sq)
    Cq_beta = dyn.get_Cs(Cq_beta_sq)


    # plt.figure(figsize=(6,7))
    # plt.plot(Cs_beta, z, label = '$\\Delta = 20$m')
    # plt.plot(Cs_2, z, label = '$\\Delta = 40}m$')
    # plt.plot(Cs_4, z, label = '$\\Delta = 80}m$')
    # plt.xlabel('$C_{s}$', fontsize=16)
    # plt.ylabel("z (m)")
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'Cs_prof.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(Cs_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Cs_2, z/z_i, label = '$\\Delta = 40}m$')
    plt.plot(Cs_4, z/z_i, label = '$\\Delta = 80}m$')
    plt.plot(Cs_8, z/z_i, label = '$\\Delta = 160}m$')
    plt.xlabel('$C_{s}$', fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'Cs_prof_scaled.png', pad_inches=0)
    plt.close()

    #
    # plt.figure(figsize=(6,7))
    # plt.plot(Cq_beta, z, label = '$\\Delta = 20$m')
    # plt.plot(Cq_2, z, label = '$\\Delta = 40$m')
    # plt.plot(Cq_4, z, label = '$\\Delta = 80$m')
    # plt.xlabel('$C_{qt}$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'Cqt_prof.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(Cq_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Cq_2, z/z_i, label = '$\\Delta = 40$m')
    plt.plot(Cq_4, z/z_i, label = '$\\Delta = 80$m')
    plt.plot(Cq_8, z/z_i, label = '$\\Delta = 160}m$')
    plt.xlabel('$C_{qt}$', fontsize=14)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'Cqt_prof_scaled.png', pad_inches=0)
    plt.close()

    #
    # plt.figure(figsize=(6,7))
    # plt.plot(Cth_beta, z, label = '$\\Delta = 20$m')
    # plt.plot(Cth_2, z, label = '$\\Delta = 40$m')
    # plt.plot(Cth_4, z, label = '$\\Delta = 80$m')
    # plt.xlabel('$C_{\\theta}$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'Cth_prof.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(Cth_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Cth_2, z/z_i, label = '$\\Delta = 40$m')
    plt.plot(Cth_4, z/z_i, label = '$\\Delta = 80$m')
    plt.plot(Cth_8, z/z_i, label = '$\\Delta = 160}m$')
    plt.xlabel('$C_{\\theta}$', fontsize=14)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'Cth_prof_scaled.png', pad_inches=0)
    plt.close


    #########################################################################################################################

    monc_l_20 = dyn.l_mix_MONC(0.23, 20, z, k=0.4)
    monc_l_40 = dyn.l_mix_MONC(0.23, 40, z, k=0.4)
    monc_l_80 = dyn.l_mix_MONC(0.23, 80, z, k=0.4)
    monc_l_160 = dyn.l_mix_MONC(0.23, 160, z, k=0.4)

    # plt.figure(figsize=(6,7))
    # plt.plot(Cs_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
    # plt.plot(Cs_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
    # plt.plot(Cs_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
    # plt.plot(monc_l_20, z, color ='tab:blue')
    # plt.plot(monc_l_40, z, color ='tab:orange')
    # plt.plot(monc_l_80, z, color ='tab:green')
    # plt.xlabel('$l_{mix}$', fontsize=16)
    # plt.ylabel("z (m)", fontsize=16)
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'l_mix_w_MONC.png', pad_inches=0)

    plt.figure(figsize=(6,7))

    plt.plot(Cs_beta*(20), z/z_i, '.', color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
    plt.plot(Cs_2*(40), z/z_i, '.', color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
    plt.plot(Cs_4*(80), z/z_i, '.', color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
    plt.plot(Cs_8*(160), z/z_i, '.', color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')

    plt.plot(monc_l_20, z/z_i, color ='tab:blue')
    plt.plot(monc_l_40, z/z_i, color ='tab:orange')
    plt.plot(monc_l_80, z/z_i, color ='tab:green')
    plt.plot(monc_l_160, z/z_i, color ='tab:red')

    plt.xlabel('$l_{mix}$', fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'l_mix_w_MONC_scaled.png', pad_inches=0)
    plt.close()

    print('plotted l mix')

    #########################################################################################################################

    #scalars

    #########################################################################################################################

    C_scalar = np.sqrt((0.23*0.23)/0.7)

    monc_l_scalar_20 = dyn.l_mix_MONC(C_scalar, 20, z, k=0.4)
    monc_l_scalar_40 = dyn.l_mix_MONC(C_scalar, 40, z, k=0.4)
    monc_l_scalar_80 = dyn.l_mix_MONC(C_scalar, 80, z, k=0.4)
    monc_l_scalar_160 = dyn.l_mix_MONC(C_scalar, 160, z, k=0.4)

    # plt.figure(figsize=(6,7))
    # plt.plot(Cth_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
    # plt.plot(Cth_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
    # plt.plot(Cth_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
    # plt.plot(monc_l_scalar_20, z, color ='tab:blue')
    # plt.plot(monc_l_scalar_40, z, color ='tab:orange')
    # plt.plot(monc_l_scalar_80, z, color ='tab:green')
    # plt.xlabel('$l_{\\theta}$', fontsize=16)
    # plt.ylabel("z (m)", fontsize=16)
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'l_th_w_MONC.png', pad_inches=0)

    plt.figure(figsize=(6,7))
    plt.plot(Cth_beta*(20), z/z_i, '.', color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
    plt.plot(Cth_2*(40), z/z_i, '.', color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
    plt.plot(Cth_4*(80), z/z_i, '.', color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
    plt.plot(Cth_8*(160), z/z_i, '.', color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')

    plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
    plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
    plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
    plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')


    plt.xlabel('$l_{\\theta}$', fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'l_th_w_MONC_scaled.png', pad_inches=0)
    plt.close()

    print('plotted l_th')

    #########################################################################################################################



    #########################################################################################################################

    #
    # plt.figure(figsize=(6,7))
    # plt.plot(Cq_beta*(20), z, '.', markersize = 10, label = '$\\Delta = 20$m')
    # plt.plot(Cq_2*(40), z, '.', markersize = 10, label = '$\\Delta = 40$m')
    # plt.plot(Cq_4*(80), z, '.', markersize = 10, label = '$\\Delta = 80$m')
    # plt.plot(monc_l_scalar_20, z, color ='tab:blue')
    # plt.plot(monc_l_scalar_40, z, color ='tab:orange')
    # plt.plot(monc_l_scalar_80, z, color ='tab:green')
    # plt.xlabel('$l_{qt}$', fontsize=16)
    # plt.ylabel("z (m)", fontsize=16)
    # plt.legend(fontsize=16, loc='upper right')
    # #plt.xlim(1, 3)
    # plt.savefig(plotdir+'l_qt_w_MONC.png', pad_inches=0)

    plt.figure(figsize=(6,7))

    plt.plot(Cq_beta*(20), z/z_i, '.', color ='tab:blue', markersize = 10, label = '$\\Delta = 20$m')
    plt.plot(Cq_2*(40), z/z_i, '.', color ='tab:orange', markersize = 10, label = '$\\Delta = 40$m')
    plt.plot(Cq_4*(80), z/z_i, '.', color ='tab:green', markersize = 10, label = '$\\Delta = 80$m')
    plt.plot(Cq_8*(160), z/z_i, '.', color ='tab:red', markersize = 10, label = '$\\Delta = 160$m')

    plt.plot(monc_l_scalar_20, z/z_i, color ='tab:blue')
    plt.plot(monc_l_scalar_40, z/z_i, color ='tab:orange')
    plt.plot(monc_l_scalar_80, z/z_i, color ='tab:green')
    plt.plot(monc_l_scalar_160, z/z_i, color ='tab:red')

    plt.xlabel('$l_{qt}$', fontsize=16)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    #plt.xlim(1, 3)
    plt.savefig(plotdir+'l_qt_w_MONC_scaled.png', pad_inches=0)
    plt.close()

    print('plotted l_qt')

    #########################################################################################################################


    Pr_th_beta = dyn.Pr(Cs_beta_sq, Cth_beta_sq)
    Pr_th_2D = dyn.Pr(Cs_sq_2, Cth_sq_2)
    Pr_th_4D = dyn.Pr(Cs_sq_4, Cth_sq_4)
    Pr_th_8D = dyn.Pr(Cs_sq_8, Cth_sq_8)

    plt.figure(figsize=(6,7))
    plt.plot(Pr_th_beta, z, label = '$\\Delta = 20$m')
    plt.plot(Pr_th_2D, z, label = '$\\Delta = 40$m')
    plt.plot(Pr_th_4D, z, label = '$\\Delta = 80$m')
    plt.plot(Pr_th_8D, z, label = '$\\Delta = 160$m')

    plt.xlabel('$Pr_{\\theta}$', fontsize=14)
    plt.ylabel("z (m)")
    plt.legend(fontsize=16, loc='upper right')
    plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
    #plt.xlim(-3, 7)
    plt.savefig(plotdir+'Pr_th_prof.png', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6,7))
    plt.plot(Pr_th_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Pr_th_2D, z/z_i, label = '$\\Delta = 40$m')
    plt.plot(Pr_th_4D, z/z_i, label = '$\\Delta = 80$m')
    plt.plot(Pr_th_8D, z/z_i, label = '$\\Delta = 160$m')

    plt.xlabel('$Pr_{\\theta}$', fontsize=14)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
    plt.xlim(-3, 7)
    plt.savefig(plotdir+'Pr_th_prof_scaled1.png', pad_inches=0)
    plt.close()


    plt.figure(figsize=(6,7))
    plt.plot(Pr_th_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Pr_th_2D, z/z_i, label = '$\\Delta = 40$m')
    plt.plot(Pr_th_4D, z/z_i, label = '$\\Delta = 80$m')
    plt.plot(Pr_th_8D, z/z_i, label = '$\\Delta = 160$m')

    plt.xlabel('$Pr_{\\theta}$', fontsize=14)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
    plt.xlim(-3, 31)
    plt.savefig(plotdir+'Pr_th_prof_scaled2.png', pad_inches=0)
    plt.close()

    ###############################################################################################

    Pr_q_beta = dyn.Pr(Cs_beta_sq, Cq_beta_sq)
    Pr_q_2D = dyn.Pr(Cs_sq_2, Cq_sq_2)
    Pr_q_4D = dyn.Pr(Cs_sq_4, Cq_sq_4)
    Pr_q_8D = dyn.Pr(Cs_sq_8, Cq_sq_8)


    plt.figure(figsize=(6,7))
    plt.plot(Pr_q_beta, z, label = '$\\Delta = 20$m')
    plt.plot(Pr_q_2D, z, label = '$\\Delta = 40$m')
    plt.plot(Pr_q_4D, z, label = '$\\Delta = 80$m')
    plt.plot(Pr_q_8D, z, label = '$\\Delta = 160$m')

    plt.xlabel('$Pr_{qt}$', fontsize=14)
    plt.ylabel("z (m)")
    plt.legend(fontsize=16, loc='upper right')
    plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
    #plt.xlim(-3, 7)
    plt.savefig(plotdir+'Pr_qt_prof.png', pad_inches=0)
    plt.close()

    plt.figure(figsize=(6,7))
    plt.plot(Pr_q_beta, z/z_i, label = '$\\Delta = 20$m')
    plt.plot(Pr_q_2D, z/z_i, label = '$\\Delta = 40$m')
    plt.plot(Pr_q_4D, z/z_i, label = '$\\Delta = 80$m')
    plt.plot(Pr_q_8D, z/z_i, label = '$\\Delta = 160$m')

    plt.xlabel('$Pr_{qt}$', fontsize=14)
    plt.ylabel("z/z$_{ML}$", fontsize=16)
    plt.legend(fontsize=16, loc='upper right')
    plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
    plt.xlim(-3, 7)
    plt.savefig(plotdir+'Pr_qt_prof_scaled.png', pad_inches=0)
    plt.close()



    # plt.figure(figsize=(6,7))
    # plt.plot(Pr_th_beta, z, label = '$Pr_{\\theta}:$ $\\Delta = 20$m')
    # plt.plot(Pr_th_2D, z, label = '$Pr_{\\theta}:$ $\\Delta = 40$m')
    # plt.plot(Pr_th_4D, z, label = '$Pr_{\\theta}:$ $\\Delta = 80$m')
    # plt.plot(Pr_q_beta, z, label = '$Pr_{qt}:$ $\\Delta = 20$m', color ='tab:blue', linestyle='dashdot')
    # plt.plot(Pr_q_2D, z, label = '$Pr_{qt}:$ $\\Delta = 40$m', color ='tab:orange', linestyle='dashdot')
    # plt.plot(Pr_q_4D, z, label = '$Pr_{qt}:$ $\\Delta = 80$m', color ='tab:green', linestyle='dashdot')
    # plt.xlabel('$Pr$', fontsize=14)
    # plt.ylabel("z (m)")
    # plt.legend(fontsize=16, loc='upper right')
    # plt.vlines(0.7, 0, 3020, 'k', linestyles='dashed', label='')
    # plt.xlim(-3, 7)
    # plt.savefig(plotdir+'Pr_all_prof.png', pad_inches=0)
    #
    # plt.figure(figsize=(6,7))
    # plt.plot(Pr_th_beta, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 20$m')
    # plt.plot(Pr_th_2D, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 40$m')
    # plt.plot(Pr_th_4D, z/z_i, label = '$Pr_{\\theta}:$ $\\Delta = 80$m')
    # plt.plot(Pr_q_beta, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 20$m', color ='tab:blue', linestyle='dashdot')
    # plt.plot(Pr_q_2D, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 40$m', color ='tab:orange', linestyle='dashdot')
    # plt.plot(Pr_q_4D, z/z_i, label = '$Pr_{qt}:$ $\\Delta = 80$m', color ='tab:green', linestyle='dashdot')
    # plt.xlabel('$Pr$', fontsize=14)
    # plt.ylabel("z/$_{ML}$")
    # plt.legend(fontsize=16, loc='upper right')
    # plt.vlines(0.7, 0, 3020/z_i, 'k', linestyles='dashed', label='')
    # plt.xlim(-3, 7)
    # plt.savefig(plotdir+'Pr_all_prof_scaled.png', pad_inches=0)

