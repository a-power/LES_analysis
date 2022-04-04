import numpy as np

def k_cut_find(delta):
    return np.pi/(delta)

def sigma_find(delta):
    return delta/2


def my_L_ij_sym(u, v, w, uu, uv, uw, vv, vw, ww):
    
    L_ij = np.array([ (u*u-uu), (u*v-uv), (u*w-uw),
                                (v*v-vv), (v*w-vw),
                                          (w*w-ww) ] )
    
    return L_ij


def L_ij_sym_xarray(uu, uv, uw, vv, vw, ww):
    L_ij = np.array([-(uu), -(uv), -(uw),
                           -(vv), -(vw),
                                 -(ww)])

    return L_ij


def S_ij(u, v, w, dx):
    
    surf_BC_0s = np.zeros_like(u)
    
    dudx = np.diff(np.insert(u, 0, [u[-1, :, :]], axis=0), axis=0)/dx
    dudy = np.diff(np.insert(u, 0, [u[:, -1, :]], axis=1), axis=1)/dx
    dudz = np.diff(np.insert(u, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2)/dx
    
    dvdx = np.diff(np.insert(v, 0, [v[-1, :, :]], axis=0), axis=0)/dx
    dvdy = np.diff(np.insert(v, 0, [v[:, -1, :]], axis=1), axis=1)/dx
    dvdz = np.diff(np.insert(v, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2)/dx
    
    dwdx = np.diff(np.insert(w, 0, [w[-1, :, :]], axis=0), axis=0)/dx
    dwdy = np.diff(np.insert(w, 0, [w[:, -1, :]], axis=1), axis=1)/dx
    dwdz = np.diff(np.insert(w, 0, [surf_BC_0s[:, :, 0]], axis=2), axis=2)/dx
    
    
    S_ij = 0.5*np.array([ [(2*dudx), (dudy+dvdx), (dudz+dwdx)],
                          [(dvdx+dudy), (2*dvdy), (dvdz+dwdy)],
                          [(dwdx+dudz), (dwdy+dvdz), (2*dwdz)] ])
    
    
    return S_ij



def abs_S(s):
    
    abs_S = np.sqrt(2*( s[0,0]*s[0,0] + s[0,1]*s[0,1] + s[0,2]*s[0,2] +
                       s[1,0]*s[1,0] + s[1,1]*s[1,1] + s[1,2]*s[1,2] +
                       s[2,0]*s[2,0] + s[2,1]*s[2,1] + s[2,2]*s[2,2]
                    ))
    return abs_S



def abs_S_hat(S_filt_in):
    
    temp_S_sum = np.zeros_like(S_filt_in[0,:,:,:])
    S_sum = np.zeros_like(S_filt_in)
    
    for i in range(6):
        if i in [1,2,4]:
            S_sum += 2*S_filt_in[i,:,:,:]*S_filt_in[i,:,:,:]
        else:
            S_sum += S_filt_in[i,:,:,:]*S_filt_in[i,:,:,:]
    

    abs_S_hat_out = np.sqrt(2*S_sum)
    
    return abs_S_hat_out


def abs_S_hat_S_ij_hat(S_filt):
    
    abs_S_filt = abs_S_hat(S_filt)
    abs_S_hat_S_ij_hat = abs_S_filt*S_filt
    
    return abs_S_hat_S_ij_hat




def M_ij(dx, dx_filt, S_filt, HAT_abs_S_Sij, beta=1):
    
    alpha = dx_filt/dx
    power = alpha/2
    
    M_ij = dx_filt**2 * beta**power * abs_S_hat_S_ij_hat(S_filt) - dx**2 * HAT_abs_S_Sij
    
    return M_ij



def index_sym(i,j, i_j_in):
    
    if j < i:
        i_new = j
        j_new = i
        
    else:
        i_new = i
        j_new = j
        
    index_str = '{}_{}'.format(i_new,j_new)

    index = np.where(i_j_in == index_str)[0][0]
    
    return index


def C_s_sq(L_ij, M_ij):
    
    C_s_sq = np.zeros_like(L_ij[0,:,:,:])
    C_s_num = np.zeros_like(L_ij[0,:,:,:])
    C_s_den = np.zeros_like(L_ij[0,:,:,:])
                        
    for it in range(0,6):
        if it in [0,3,5]:
            
            C_s_num += L_ij[it,:,:,:] * M_ij[it,:,:,:]
            C_s_den += M_ij[it,:,:,:]**2            
    
        else:        
            C_s_num += 2*(L_ij[it,:,:,:] * M_ij[it,:,:,:])
            C_s_den += 2*M_ij[it,:,:,:]**2
       
    C_s_sq = 0.5 * C_s_num / C_s_den
                        
    
    return C_s_sq


def get_Cs(Cs_sq):
    """ calculates C_s from C_s^2 by setting neg values to zero
    and sq rooting"""

    Cs_sq_copy = Cs_sq.copy()
    Cs_sq_copy[Cs_sq < 0] = 0
    Cs = np.sqrt(Cs_sq_copy)

    return Cs


def Cs_av_levels(L_ij, M_ij, av_method = 'all', return_all=0):
    """ Calculates the horizontal average Cs value at each level 
    using the Lij and Mij fields as input.
    
    av_method can equal:
    'all': all Cs_squared values are used in the calculation, 
    neg_to_zero: the negative Cs_squared values are set to zero, 
    or 'no_neg': the negative Cs_squared values are not included in the calculation
    
    """
    
    C_s_num = np.zeros_like(L_ij[0,:,:,:])
    C_s_den = np.zeros_like(L_ij[0,:,:,:])
  
                        
    for it in range(0,6):
        if it in [0,3,5]:
            
            C_s_num += L_ij[it,:,:,:] * M_ij[it,:,:,:]
            C_s_den += M_ij[it,:,:,:]**2

        else:        
            C_s_num += 2*(L_ij[it,:,:,:] * M_ij[it,:,:,:])
            C_s_den += 2*M_ij[it,:,:,:]**2
            
            
    z_num = len(C_s_num[0,0,:])
    horiz_num_temp = len(C_s_num[0,:,0])
    horiz_num = horiz_num_temp**2

    LM_flat = C_s_num.reshape(horiz_num,z_num)
    LM_av = np.zeros(z_num)

    MM_flat = C_s_den.reshape(horiz_num,z_num)
    MM_av = np.zeros(z_num)
    
    #########################################
    if av_method == 'no_neg':

        for k in range(z_num):    
            n = 0
            LM_pos_sum = 0
            MM_pos_sum = 0

            for ij in range(horiz_num): 
                if LM_flat[ij,k] > 0:
                    LM_pos_sum += LM_flat[ij,k]
                    MM_pos_sum += MM_flat[ij,k]
                    n = n+1
            if n != 0:
                LM_av[k] = LM_pos_sum/n
                MM_av[k] = MM_pos_sum/n
        Cs_av_sq = (0.5*(LM_av / MM_av))
        Cs_av = Cs_av_sq**(1/2)
                
    ##########################################
    elif av_method == 'neg_to_zero':

        for k in range(z_num):    
            n = 0
            LM_pos_sum = 0
            MM_pos_sum = 0

            for ij in range(horiz_num): 
                if LM_flat[ij,k] > 0:
                    LM_pos_sum += LM_flat[ij,k]
                MM_pos_sum += MM_flat[ij,k]
                n = n+1
            if n != 0:
                LM_av[k] = LM_pos_sum/n
                MM_av[k] = MM_pos_sum/n
        Cs_av_sq = (0.5*(LM_av / MM_av))
        Cs_av = Cs_av_sq**(1/2)
                
    ##########################################
    elif av_method == 'all':

        for k in range(z_num):    
            
            LM_av[k] = np.sum(LM_flat[:,k])/horiz_num
            MM_av[k] = np.sum(MM_flat[:,k])/horiz_num

       
        Cs_av_sq = (0.5*(LM_av / MM_av))
        Cs_av_temp = Cs_av_sq.copy()
        Cs_av_temp[Cs_av_sq < 0] = 0
        Cs_av = Cs_av_temp**(1/2)

    if return_all == 1:
        return Cs_av, LM_av, MM_av, Cs_av_sq
    else:
        return Cs_av
      
    
    
def beta_calc(C_2D_sq_in, C_4D_sq_in):
    
    Cs_2D_sq_copy1 = C_2D_sq_in.copy()
    Cs_2D_sq_copy2 = C_2D_sq_in.copy()
    Cs_4D_sq_copy = C_4D_sq_in.copy()
    
    Cs_2D_sq_copy2[Cs_2D_sq_copy1==0.00000] = 1
    Cs_4D_sq_copy[Cs_2D_sq_copy1==0.00000] = 100000 #go to inf
    
    beta = Cs_4D_sq_copy/Cs_2D_sq_copy2
    beta[beta < 0.125] = 0.125
    
    return beta
    
    
def Cs_sq_beta_dep(C_s2_sq, beta):    
    
    """ calculates Cs_sq using C_s = C_s2_sq/beta """
    
    return C_s2_sq/beta
    

def w_therm_field(w, t_in, return_all=False):
    w_95th = np.zeros_like(w[t_in, 0, 0, :])
    w_therm = np.zeros_like(w[t_in, ...])

    for k in range(len(w[t_in, 0, 0, :])):
        w_95th[k] = np.percentile(w[t_in, :, :, k], 95)
        w_therm[:, :, k] = (w[t_in, :, :, k] >= w_95th[k])

    if return_all == True:
        return w_therm, w_95th
    else:
        return w_therm


def cloud_field_ind(cloud_field, t_in, cloud_liquid_threshold = 10**(-5)):

    clouds = np.zeros_like(cloud_field[t_in, ...])

    for k in range(len(cloud_field[t_in, 0, 0, :])):
        clouds[:, :, k] = (cloud_field[t_in, :, :, k] >= cloud_liquid_threshold)

    return clouds


def Cs_therm_cloud_field(Cs, t_in, i_ind, j_ind, k_ind):
    Cs_therm_cloud = np.zeros_like(Cs[t_in, ...])

    for i in range(len(i_ind)):
        Cs_therm_cloud[i_ind[i], j_ind[i], k_ind[i]] = Cs[t_in, i_ind[i], j_ind[i], k_ind[i]]

    return Cs_therm_cloud


#### check time indices for Cs (Lij and Mij)