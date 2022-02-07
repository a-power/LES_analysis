import numpy as np
from scipy import ndimage
# Ignore divide-by-zero warnings that occur when converting between wavenumbers/frequency/wavelength
np.seterr(divide='ignore')


# def sig_to_dx_filt(sig_smag, sig_filt):
#     dx = (np.pi/(1.5174))*(np.sqrt((sig_smag**2)+(sig_filt**2))) #10% method
#     return round(dx, 2)

def sig_to_dx(sig_filt):
    dx = np.sqrt(6)*(sig_filt) #(np.pi/(2))
    return round(dx, 2)

# def dx_to_sig_filt(dx_want, dw_orig, sig_smag):
#     sig_filt = np.sqrt((4/np.pi**2)*(dx_want - dx_orig)**2 - sig_smag)
#     return sig_filt

def dx_to_sig_filt(dx_eff, sig_smag):
    sig_filt = np.sqrt((2.3/np.pi**2)*(dx_eff)**2 - sig_smag**2)
    return sig_filt

def S_smag(u, v, w):
    "calculates S for each point, therefore at each point you haev an Sij matrix/tensor"

    S = 0.5*[[2*np.diff(u, axis=0), np.diff(u, axis=1)+np.diff(v, axis=0), np.diff(u, axis=2)+np.diff(x, axis=0)],
        [np.diff(v, axis=0)+np.diff(u, axis=1), 2*np.diff(v, axis=1), np.diff(v, axis=2)+np.diff(w, axis=1)],
        [np.diff(w, axis=0)+np.diff(u, axis=2), np.diff(w, axis=1)+np.diff(v, axis=2), 2*np.diff(w, axis=2)]]
    return S

def L(u,v,w,uu,vv,ww,uv,uw,vw):
    "inputs must all be filtered versions"
    

def C_dyn(L, M): # set_z=None):
    "calculates the local dynamic Smagorinsky constant"
    
    if np.shape(L) != np.shape(M):
        print("Arrays dimentions do not match")
        
    
    else:
        C = 0
        C = (1/2)*(L*M/M**2)
           
    return C
 

def interp(var_in, grid_to, grid_from, underground=False):
    """ interpolates a variable (var_in) from one grid (grid_from) to another grid (grid_to)
    using a simple linear interpolation and returns the interpolated variable (var_out)"""
    
    iters = len(grid_to)
    
    var_out = np.zeros(iters)
    
    if underground == False:
        for k in range(iters-1):

            var_out[k] = var_in[k] - \
                (grid_from[k] - grid_to[k])*((var_in[k+1] - var_in[k])/(grid_from[k+1] - grid_from[k]))
            
    else:
        for k in range(iters-1):
            var_out[k] = var_in[k] + \
                (grid_to[k] - grid_from[k])*((var_in[k+1] - var_in[k])/(grid_from[k+1] - grid_from[k]))
        var_out[0] = var_in[0]
        
    return var_out




def interp_3d(var_in, grid_to, grid_from):
    """ Linearly interpolates a given 3d dataset  (var_in)
    from its origional grid (grid from) to the desired
    grid (grid_to)"""
    
    var_out = np.zeros_like(var_in)
    
    for k in range(len(var_in[0,0,:])-1):
        for i in range(len(var_in[:,0,0])):
            for j in range(len(var_in[0,:,0])):

                var_out[i,j,k] = var_in[i,j,k] + \
                    (grid_to[k] - grid_from[k])*((var_in[i,j,k+1] - var_in[i,j,k])/(grid_from[k+1] - grid_from[k]))
    return var_out




def filt_cg_2d(var_in):
    """ horizontally filters a given 3D variable in using 
    a coursegraining method where the 4 points around the 
    given point being filtered are averaged. the point itself 
    is not included in this average """
    
    z_len = len(var_in[0,0,:])
    x_len = len(var_in[:,0,0])
    y_len = len(var_in[0,:,0]) 
    
    var_out = np.zeros((int(x_len/2), int(y_len/2), int(z_len)))
    
    for i in np.arange(0, z_len):
        for j in np.arange(0, x_len, 2):
            for k in np.arange(0, y_len, 2):

                j_new = int(j/2)
                k_new = int(k/2)


                var_out[j_new,k_new,i] = (var_in[j%x_len,(k+1)%x_len,i] + var_in[j%x_len, (k-1)%x_len, i] \
                                        + var_in[(j+1)%x_len, k%x_len, i] + var_in[(j-1)%x_len, k%x_len, i])/4
                        
    return var_out          




def filt_ma5_2d(var_in):
    """ 5 point stencil:
    horizontally filters a given 3D variable in using 
    a moving average method where the 4 points around the 
    given point, and the point itself, are being averaged. """
    
    z_len = len(var_in[0,0,:])
    x_len = len(var_in[:,0,0])
    y_len = len(var_in[0,:,0]) 
    
    var_out = np.zeros((int(x_len), int(y_len), int(z_len)))
    
    for i in np.arange(0, z_len):
        for j in np.arange(0, x_len):
            for k in np.arange(0, y_len):

                var_out[j,k,i] = (var_in[j%x_len, k%x_len, i] + var_in[j%x_len,(k+1)%x_len,i] \
                                  + var_in[j%x_len, (k-1)%x_len, i] + var_in[(j+1)%x_len, k%x_len, i] \
                                  + var_in[(j-1)%x_len, k%x_len, i])/5
                        
    return var_out   




def filt_ma9_2d(var_in):
    """ horizontally filters a given 3D variable in using 
    a moving average method where the 8 points around the 
    given point, along with the point itself, are averaged."""
    
    z_len = len(var_in[0,0,:])
    x_len = len(var_in[:,0,0])
    y_len = len(var_in[0,:,0]) 
    
    var_out = np.zeros((int(x_len), int(y_len), int(z_len)))
    
    for k in np.arange(0, z_len):
        for i in np.arange(0, x_len):
            for j in np.arange(0, y_len):
                var_add = 0
                for ii in range(-1,2):
                    for jj in range(-1,2):

                        var_add += var_in[(i+ii)%x_len, (j+jj)%y_len, k]
                        
                var_out[i,j,k] = var_add/9 
                
    return var_out   




def mean_prof(var1_in, var2_in = None):
    """ takes in variable 1 and an optional variable 2. 
    if only one variable the function gets the mean 
    vertical profile of var1, if a second variable is
    also given then var1 is multiplied by var2 and the
    mean vertical profile of their product is computed. """

    vert_points = len(var1_in[0,0,:])
    
    var_out = np.zeros(vert_points)
    var_mult = np.zeros_like(var1_in)

    for i in range(vert_points):
    
        if type(var2_in) is np.ndarray:
            for j in range(len(var1_in[:,0,0])):
                for k in range(len(var1_in[0,:,0])):
                      
                    var_mult[j,k,i] = var1_in[j,k,i]*var2_in[j,k,i]
                    
            var_out[i] = np.mean(var_mult[:,:,i])
                      
        else:
            var_out[i] = np.mean(var1_in[:,:,i])
                      
                      
    return var_out                 
                      
                      
                      

def new_sub(var_og, var_filt):   
    """this function takes in the unfiltered data (var_og) and the 
    filtered data (var_filt) and find the new subgrid"""
    
    new_subgrid = var_og - var_filt
#     subgrid_end = int(np.where(new_subgrid < 0.01)[0][1])
    
#     print(subgrid_end)
    
#     only_subgrid = np.zeros(len(var_og))

#     for i in range(subgrid_end):

#         only_subgrid[i] = new_subgrid[i]
        
    return new_subgrid





def hor_av(var_in, av_dir, z_level):
    """function takes in a 3d variable and the direction (x or y)
    accross which the variable is to be averaged on a given z level:
    
    av_dir = x_dir or y_dir """
    
    av_dir_str = str(av_dir)
    z_in = int(z_level)
    
    if av_dir_str == 'x_dir':
        len_domain = len(var_in[0,:,0])
        av_var = np.zeros_like(var_in[0,:,0])
        
        for j in range(len_domain):
            
            av_var[j] = np.mean(var_in[:, j, z_in])
          
        
    
    if av_dir_str == 'y_dir':
        len_domain = len(var_in[:,0,0])
        av_var = np.zeros_like(var_in[:,0,0])
        
        for i in range(len_domain):
            
            av_var[i] = np.mean(var_in[i, :, z_in])
        
    
    return av_var






def GetPSD1D(psd2D):
    """
    Get PSD 1D (total radial power spectrum) 
    For use with option spec_method: ndimage

    Args:
        psd2D    : 2D numpy array containing 2D spectra values

    Returns:
        psd1D    : 1D numpy array containing 1D spectra ordered from 
                   wavenumber 0 to highest wavenumber of shortest
                   dimension

    @author:  https://gist.github.com/TangibitStudios/47beaf24690329ac7fecddde70835ce9

    """

    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))

    return psd1D


def forward(n):
    return 2*np.pi/n # l = 2*np.pi/n

def inverse(l):
    return 2*np.pi/l # n = 2*np.pi/l



def spectra_2d(var_in, dx, dy, options, z=1):
    
    #indexing the co-ords
    
    xx = 0
    yx = 1
    zx = 2
    
    # Prepare map flag: need for durran
    prepare_map = True

    # Set up helpful parameters
    
    nt = 0
    if z == 1:
        nx = len(var_in[:,0,0])
        ny = len(var_in[0,:,0])
        nz = len(var_in[0,0,:])
    else:
        nx = len(var_in[:,0])
        ny = len(var_in[0,:])
        

    # Horizontal domain lengths
    L_x = dx*nx
    L_y = dy*ny

    # NOTE: Wavenumber zero is neither considered nor reported - mean removed from data.

    fx = np.fft.fftfreq(nx, d=dx)[1:]   # frequencies (1/m)
    fy = np.fft.fftfreq(ny, d=dy)[1:]
    kx = fx*2*np.pi                     # wavenumbers (radians/m)
    ky = fy*2*np.pi
    kh = np.sqrt(kx**2 + ky**2)         # total wavenumber
    dkx = 2*np.pi/L_x                   # delta wavenumbers
    dky = 2*np.pi/L_y

    # discretize the 2D wavenumber in multiples of the maximum one-dimensional wavenumber
    dkh = np.max((dkx,dky))
    Nmax = np.int(np.ceil(np.sqrt(2)*np.max((nx/2,ny/2))))  # maximum number of points
    kp = (np.arange(Nmax-1)+1)*dkh                          # for eqn 18
    dkmin = np.min((dkx,dky))

    # Reliable Nyquist frequency (highest reliable wavenumber, shortest reliable wavelength)
    NYQwavel = 2*np.max((dy,dx))
    NYQwaven = 2*np.pi/NYQwavel
    kp_keep = kp <= NYQwaven   # wavenumbers to retain if using "restrict" option

    fkx = np.fft.fftfreq(nx, d=dx)*2*np.pi
    fky = np.fft.fftfreq(ny, d=dy)*2*np.pi
    norm = (dx*dy*dkmin)/(8*(np.pi**2)*nx*ny) # normalization factor (see eqn 24)

    # Compute the 2D fft for the full [t,y,x,z] dataset, over y and x, at once.
    temp = np.fft.fft2((var_in - np.mean(var_in, axis=(yx,xx), keepdims=True, dtype='float64')),axes=(yx,xx))  

    Ek = (temp*np.conj(temp)).real

    # Populate the labels, counts, and means for each kp [for radial summation]
    # basically, this is "if this is the first time working on data with the same horizontal dimensions"
    if prepare_map:
        #print("Preparing map")
        # Prepare wavnumber grid (rmap) based on kx and ky
        # To be used in radial sum and/or applying the Tanguay/Durran correction term
        
        gkx=np.tile(fkx, (ny, 1))                      # x wavenumber array, repeated ny times 
        gky=np.tile(np.array([fky]).T, (1, nx))        # y wavenumber array, repeated nx times
        rmap=np.sqrt(gkx**2 + gky**2,dtype='float64')  # wavenumber grid
        rlab=(rmap*0).astype(np.int)                   # grid of labels denoting index of wavenumbers (kp)
        kcount=kp*0                                    # to hold count of points in kp; sum(kcount)=(nx*ny)-1
        kpbar=kp*0                                     # to hold mean of wavenumber grid values at kp
        rindex=np.arange(0,Nmax-1)                     # list of labels to sum (all - to start with...)

        for knc,kpl in enumerate(kp):
            keep=((kpl-dkh/2 <= rmap) & (rmap < kpl+dkh/2))  # see eqn 18
            rlab[keep] = knc
            kcount[knc] = np.count_nonzero(keep)
            kpbar[knc] = np.mean(rmap[keep])
            
    prepare_map = False     #ALWAYS ASSUMES UNIFORMITY IN FILE    # When enabled, only compute the map once.

    # Calculate the fft2 with requested computation method supplied via options
    method = options['spec_method']  # [Durran | ndimage]
    compensation = options['spec_compensation'] # [True | False]
    restrict = options['spec_restrict'] # [True | False]


    if method.upper() == 'DURRAN':
        # Prepare result array
        kpo=kp
        if restrict:   # Keep only values below Nyquist frequency
            kpo=kp[kp_keep]
            rindexo=rindex[kp_keep]
            kcounto=kcount[kp_keep]
            kpbaro=kpbar[kp_keep]
        else:
            kpo=kp
            rindexo=rindex
            kcounto=kcount
            kpbaro=kpbar

        kplen=kpo.size
        
        if z==1:
            Ekp=np.zeros((kplen,nz))
            for znc in np.arange(nz):
                    # Sum points
                Ekp[:,znc] = norm * ndimage.sum(Ek[...,znc], rlab, index=rindexo)
                    # Apply compensation  (See eqn 29)
                if compensation:
                    Ekp[:,znc] *= (2*np.pi*kpbaro[:]/(dkmin*kcounto[:]))
        else:
            Ekp = norm * ndimage.sum(Ek, rlab, index=rindexo)
                    # Apply compensation  (See eqn 29)
            if compensation:
                Ekp *= (2*np.pi*kpbaro[:]/(dkmin*kcounto[:]))

    elif (method.upper() == 'NDIMAGE') and (nx == ny):
        # Prepare result array
        if z==1:
            if nt == 0:
                kplen=Ek[...,0].shape[0]//2
                Ekp=np.zeros((kplen,nz))
                kpo=fkx[0:kplen]

                for znc in np.arange(nz):
                        # Sum points
                    Ekp[:,znc] = norm * GetPSD1D(np.fft.fftshift(Ek[...,znc]))

            else:
                kplen=Ek[0,...,0].shape[0]//2
                Ekp=np.zeros((kplen,nz))
                kpo=fkx[0:kplen]
                for znc in np.arange(nz):
                        # Sum points
                    Ekp[:,znc] = norm * GetPSD1D(np.fft.fftshift(Ek[...,znc]))
        else:
            if nt == 0:
                kplen=Ek.shape[0]//2
                kpo=fkx[0:kplen]
                Ekp = norm * GetPSD1D(np.fft.fftshift(Ek))

            else:
                kplen=Ek[0,...].shape[0]//2
                kpo=fkx[0:kplen]
                Ekp = norm * GetPSD1D(np.fft.fftshift(Ek))
    
    else:
        print("Must supply 2D FFT spec_method: [DURRAN | NDIMAGE] and must have nx == ny for NDIMAGE")
        print("FAIL - spec_method")
        sys.exit()

    # Useful plotting values
    #kpo is radian wavenumber
    freq = kpo/(2*np.pi)
    wavel = 1/freq

    return Ekp, kpo



def cospec_2d(var_a, var_b, dx, dy, options):
    
    #indexing the co-ords
    
    
    xx = 0
    yx = 1
    zx = 2
    
    # Prepare map flag: need for durran
    prepare_map = True

    # Set up helpful parameters
    
    nt = 0
    
    nx = len(var_a[:,0,0]) #assumes variables a and b aare on the same grid
    ny = len(var_a[0,:,0]) #assumes variables a and b aare on the same grid
    nz = len(var_a[0,0,:]) #assumes variables a and b aare on the same grid

    # Horizontal domain lengths
    L_x = dx*nx
    L_y = dy*ny

    # NOTE: Wavenumber zero is neither considered nor reported - mean removed from data.

    fx = np.fft.fftfreq(nx, d=dx)[1:]   # frequencies (1/m)
    fy = np.fft.fftfreq(ny, d=dy)[1:]
    kx = fx*2*np.pi                     # wavenumbers (radians/m)
    ky = fy*2*np.pi
    kh = np.sqrt(kx**2 + ky**2)         # total wavenumber
    dkx = 2*np.pi/L_x                   # delta wavenumbers
    dky = 2*np.pi/L_y

    # discretize the 2D wavenumber in multiples of the maximum one-dimensional wavenumber
    dkh = np.max((dkx,dky))
    Nmax = np.int(np.ceil(np.sqrt(2)*np.max((nx/2,ny/2))))  # maximum number of points
    kp = (np.arange(Nmax-1)+1)*dkh                          # for eqn 18
    dkmin = np.min((dkx,dky))

    # Reliable Nyquist frequency (highest reliable wavenumber, shortest reliable wavelength)
    NYQwavel = 2*np.max((dy,dx))
    NYQwaven = 2*np.pi/NYQwavel
    kp_keep = kp <= NYQwaven   # wavenumbers to retain if using "restrict" option

    fkx = np.fft.fftfreq(nx, d=dx)*2*np.pi
    fky = np.fft.fftfreq(ny, d=dy)*2*np.pi
    norm = (dx*dy*dkmin)/(8*(np.pi**2)*nx*ny) # normalization factor (see eqn 24)

    # Compute the 2D fft for the full [t,y,x,z] dataset, over y and x, at once.
    temp_a = np.fft.fft2((var_a - np.mean(var_a, axis=(yx,xx), keepdims=True, dtype='float64')),axes=(yx,xx))  
    temp_b = np.fft.fft2((var_b - np.mean(var_b, axis=(yx,xx), keepdims=True, dtype='float64')),axes=(yx,xx))

    co = (temp_a*np.conj(temp_b)).real

    # Populate the labels, counts, and means for each kp [for radial summation]
    # basically, this is "if this is the first time working on data with the same horizontal dimensions"
    if prepare_map:
        print("Preparing map")
        # Prepare wavnumber grid (rmap) based on kx and ky
        # To be used in radial sum and/or applying the Tanguay/Durran correction term
        
        gkx=np.tile(fkx, (ny, 1))                      # x wavenumber array, repeated ny times 
        gky=np.tile(np.array([fky]).T, (1, nx))        # y wavenumber array, repeated nx times
        rmap=np.sqrt(gkx**2 + gky**2,dtype='float64')  # wavenumber grid
        rlab=(rmap*0).astype(np.int)                   # grid of labels denoting index of wavenumbers (kp)
        kcount=kp*0                                    # to hold count of points in kp; sum(kcount)=(nx*ny)-1
        kpbar=kp*0                                     # to hold mean of wavenumber grid values at kp
        rindex=np.arange(0,Nmax-1)                     # list of labels to sum (all - to start with...)

        for knc,kpl in enumerate(kp):
            keep=((kpl-dkh/2 <= rmap) & (rmap < kpl+dkh/2))  # see eqn 18
            rlab[keep] = knc
            kcount[knc] = np.count_nonzero(keep)
            kpbar[knc] = np.mean(rmap[keep])
            
    prepare_map = False     #ALWAYS ASSUMES UNIFORMITY IN FILE    # When enabled, only compute the map once.


    # Calculate the fft2 with requested computation method supplied via options
    method = options['spec_method']  # [Durran | ndimage]
    compensation = options['spec_compensation'] # [True | False]
    restrict = options['spec_restrict'] # [True | False]



    if method.upper() == 'DURRAN':
        # Prepare result array
        kpo=kp
        if restrict:   # Keep only values below Nyquist frequency
            kpo=kp[kp_keep]
            rindexo=rindex[kp_keep]
            kcounto=kcount[kp_keep]
            kpbaro=kpbar[kp_keep]
        else:
            kpo=kp
            rindexo=rindex
            kcounto=kcount
            kpbaro=kpbar

        kplen=kpo.size
        cospec=np.zeros((kplen,nz))
        
        
        for znc in np.arange(nz):
                # Sum points
            cospec[:,znc] = norm * ndimage.sum(co[...,znc], rlab, index=rindexo)
                # Apply compensation  (See eqn 29)
            if compensation:
                cospec[:,znc] *= (2*np.pi*kpbaro[:]/(dkmin*kcounto[:]))

    elif (method.upper() == 'NDIMAGE') and (nx == ny):
        # Prepare result array
        if nt == 0:
            kplen=co[...,0].shape[0]//2
            cospec=np.zeros((kplen,nz))
            kpo=fkx[0:kplen]
            
            for znc in np.arange(nz):
                    # Sum points
                cospec[:,znc] = norm * GetPSD1D(np.fft.fftshift(co[...,znc]))
        
        else:
            kplen=co[0,...,0].shape[0]//2
            cospec=np.zeros((kplen,nz))
            kpo=fkx[0:kplen]
            for znc in np.arange(nz):
                    # Sum points
                cospec[:,znc] = norm * GetPSD1D(np.fft.fftshift(co[...,znc]))
    else:
        print("Must supply 2D FFT spec_method: [DURRAN | NDIMAGE] and must have nx == ny for NDIMAGE")
        print("FAIL - spec_method")
        sys.exit()

    # Useful plotting values
    #kpo is radian wavenumber
    freq = kpo/(2*np.pi)
    wavel = 1/freq

    return co, cospec, kpo