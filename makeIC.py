import numpy as np
import h5py as h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import scipy.special
import pdb


def makeIC_box(DIMS=2, N_1D=64, fname='gasgrain_2d_64.hdf5', Pressure_Bump_Amplitude=1.0, Pressure_Bump_Width=1., BoxSize=6., forcedEOS=False):

    Lbox = BoxSize*Pressure_Bump_Width;
    rho_desired = 1.; Ngrains_Ngas = 1; dustgas_massratio=0.01;

    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    x0=np.arange(-0.5,0.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);
    N_1D_dust = np.round((1.*N_1D)*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')
    if(Ngrains_Ngas <= 0): N_1D_dust=10
    x0d=np.arange(-0.5,0.5,1./N_1D_dust); x0d+=0.5*(0.5-x0d[-1]);
    x0d+=0.5/(1.*N_1D_dust)+0.5*(1./(1.*N_1D)-1./(1.*N_1D_dust)); x0d[x0d>0.5]-=1.

    if(DIMS==3):
        xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
        xv_d, yv_d, zv_d = np.meshgrid(x0d,x0d,x0d, sparse=False, indexing='xy')
    elif(DIMS==2):
        xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
        xv_d, yv_d = np.meshgrid(x0d,x0d, sparse=False, indexing='xy'); zv_d = 0.0*xv_d
 
    Ngas=xv_g.size; Ngrains=xv_d.size;
    xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lbox;
    xv_d=xv_d.flatten()*Lbox; yv_d=yv_d.flatten()*Lbox; zv_d=zv_d.flatten()*Lbox;
    m_target_gas = (1.*rho_desired) * ((1.*Lbox)**DIMS) / (1.*Ngas)
    m_target_gas *= (1. + Pressure_Bump_Amplitude*(Pressure_Bump_Width/Lbox)*np.sqrt(2.*np.pi)*0.9973); # account for mass 'in bump'
    if(Ngrains_Ngas<=0): Ngrains=0
    
    print(xv_g)
    ## 1-D compression to give desired density ratio:
    dx=0.01/N_1D*Lbox/2.; xt=np.arange(0.,Lbox/2.,dx);
    xt_c=np.cumsum(dx/(1. + Pressure_Bump_Amplitude*np.exp(-xt*xt/(2.*Pressure_Bump_Width*Pressure_Bump_Width))))
    xt=np.append(xt,-xt[1:]); xt_c=np.append(xt_c,-xt_c[1:]); xt=np.sort(xt); xt_c=np.sort(xt_c); 
    xv_g_new = np.interp(xv_g,xt,xt_c)
    
    s0=np.sqrt(2.)*Pressure_Bump_Width/Lbox; # length units of Lbox
    p0=Pressure_Bump_Amplitude*np.sqrt(np.pi)*s0; 
    dx=1.e-4/N_1D; xt=np.arange(-0.5,0.5,dx); 
    ft = 0.5*(1. + (2.*xt + p0*scipy.special.erf(xt/s0)) / (1. + p0*scipy.special.erf(0.5*Lbox/s0)));
    
    f0 = np.random.rand(Ngas); xv_g = np.interp(f0,ft,xt) * Lbox
    yv_g=(np.random.rand(Ngas)-0.5)*Lbox; 
    if DIMS==3:
        zv_g=(np.random.rand(Ngas)-0.5)*Lbox; 
        
    xv_g+=0.5*Lbox; yv_g+=0.5*Lbox; zv_g+=0.5*Lbox; 
    xv_d+=0.5*Lbox; yv_d+=0.5*Lbox; zv_d+=0.5*Lbox; 
    
    ok=np.where(xv_d >= np.max(xv_d)-0.01*Lbox/N_1D_dust)[0];
    xv_d=xv_d[ok]-0.08*Lbox; yv_d=yv_d[ok]; zv_d=zv_d[ok]; Ngrains=xv_d.size
        
    pylab.close('all'); pylab.plot(xv_g,yv_g,linestyle='',marker=',',color='black')

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    # Compensate for pressure bump shear
    if forcedEOS:
        p["Velocities"][:,1] -= -0.05+Pressure_Bump_Amplitude * ((xv_g-0.5*Lbox) / Pressure_Bump_Width**2) * np.exp(-(xv_g-0.5*Lbox)**2/2/Pressure_Bump_Width**2)/2/(1+Pressure_Bump_Amplitude*np.exp(-(xv_g-0.5*Lbox)**2/2/Pressure_Bump_Width**2))

    if(Ngrains > 0):
        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xv_d+dustgas_massratio*m_target_gas/(1.*Ngrains_Ngas)))
    file.close()

def makeIC_rectangular_box(DIMS=3, N_1D=64, fname='gasgrain_3d_64.hdf5',
        Pressure_Bump_Amplitude=1.0, Pressure_Bump_Width=1.,xz=1,yz=0):

    Lbox = 6.*Pressure_Bump_Width;
    Lheight = 2.*Pressure_Bump_Width;
    rho_desired = 1.; Ngrains_Ngas = 1; dustgas_massratio=0.01;

    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    x0=np.arange(-0.5,0.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);
    N_1D_dust = np.round((1.*N_1D)*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')
    if(Ngrains_Ngas <= 0): N_1D_dust=10
    x0d=np.arange(-0.5,0.5,1./N_1D_dust); x0d+=0.5*(0.5-x0d[-1]);
    x0d+=0.5/(1.*N_1D_dust)+0.5*(1./(1.*N_1D)-1./(1.*N_1D_dust)); x0d[x0d>0.5]-=1.

    #likewise for the z-direction
    z0=np.arange(-0.5,0.5,1./(N_1D*(Lheight/Lbox))); z0+=0.5*(0.5-z0[-1]);
    N_1D_dust_z = np.round((1.*(N_1D*Lheight/Lbox))*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')
    if(Ngrains_Ngas <= 0): N_1D_dust_z=10
    z0d=np.arange(-0.5,0.5,1./N_1D_dust_z); z0d+=0.5*(0.5-z0d[-1]);
    z0d+=0.5/(1.*N_1D_dust_z)+0.5*(1./(1.*N_1D*Lheight/Lbox)-1./(1.*N_1D_dust_z)); z0d[z0d>0.5]-=1.

    if(DIMS==3):
        xv_g, yv_g, zv_g = np.meshgrid(x0,x0,z0, sparse=False, indexing='xy')
        xv_d, yv_d, zv_d = np.meshgrid(x0d,x0d,z0d, sparse=False, indexing='xy')
    elif (DIMS==2) and (xz == 0) and (yz == 0):
        xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
        xv_d, yv_d = np.meshgrid(x0d,x0d, sparse=False, indexing='xy'); zv_d = 0.0*xv_d
    elif (DIMS==2) and (xz == 1):
        xv_g, yv_g = np.meshgrid(z0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
        xv_d, yv_d = np.meshgrid(z0d,x0d, sparse=False, indexing='xy'); zv_d = 0.0*xv_d
 
    Ngas=xv_g.size; Ngrains=xv_d.size;
    if xz == 0:
        xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lheight;
        xv_d=xv_d.flatten()*Lbox; yv_d=yv_d.flatten()*Lbox; zv_d=zv_d.flatten()*Lheight;
    elif xz == 1:
        xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lheight; zv_g=zv_g.flatten()*Lbox;
        xv_d=xv_d.flatten()*Lbox; yv_d=yv_d.flatten()*Lheight; zv_d=zv_d.flatten()*Lbox;

    if DIMS < 3:
        m_target_gas = (1.*rho_desired) * ((1.*Lbox)*(1.*Lheight)) / (1.*Ngas)
    else:
        m_target_gas = (1.*rho_desired) * ((1.*Lbox)**(DIMS-1)*(1.*Lheight)) / (1.*Ngas)

    m_target_gas *= (1. + Pressure_Bump_Amplitude*(Pressure_Bump_Width/Lbox)*np.sqrt(2.*np.pi)*0.9973); # account for mass 'in bump'
    if(Ngrains_Ngas<=0): Ngrains=0
    
    print(xv_g)
    ## 1-D compression to give desired density ratio:
    dx=0.01/N_1D*Lbox/2.; xt=np.arange(0.,Lbox/2.,dx);
    xt_c=np.cumsum(dx/(1. + Pressure_Bump_Amplitude*np.exp(-xt*xt/(2.*Pressure_Bump_Width*Pressure_Bump_Width))))
    xt=np.append(xt,-xt[1:]); xt_c=np.append(xt_c,-xt_c[1:]); xt=np.sort(xt); xt_c=np.sort(xt_c); 
    xv_g_new = np.interp(xv_g,xt,xt_c)
    
    s0=np.sqrt(2.)*Pressure_Bump_Width/Lbox; # length units of Lbox
    p0=Pressure_Bump_Amplitude*np.sqrt(np.pi)*s0; 
    dx=1.e-4/N_1D; xt=np.arange(-0.5,0.5,dx); 
    ft = 0.5*(1. + (2.*xt + p0*scipy.special.erf(xt/s0)) / (1. + p0*scipy.special.erf(0.5*Lbox/s0)));
    
    f0 = np.random.rand(Ngas); xv_g = np.interp(f0,ft,xt) * Lbox + 0.5*Lbox
    if xz == 0:
        yv_g=(np.random.rand(Ngas)-0.5)*Lbox + 0.5*Lbox; 
        zv_g=(np.random.rand(Ngas)-0.5)*Lheight + 0.5*Lheight;
        xv_d+=0.5*Lbox; yv_d+=0.5*Lbox; zv_d+=0.5*Lheight; 
    elif xz == 1:
        zv_g += 0.5*Lbox; 
        yv_g=(np.random.rand(Ngas)-0.5)*Lheight + 0.5 * Lheight; 
        xv_d+=0.5*Lbox; yv_d+=0.5*Lheight; zv_d+=0.5*Lbox; 
    
    ok=np.where(xv_d >= np.max(xv_d)-0.01*Lbox/N_1D_dust)[0];
    xv_d=xv_d[ok]-0.08*Lbox; yv_d=yv_d[ok]; zv_d=zv_d[ok]; Ngrains=xv_d.size
        
    pylab.close('all'); pylab.plot(xv_g,yv_g,linestyle='',marker=',',color='black')

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    if(Ngrains > 0):
        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xv_d+dustgas_massratio*m_target_gas/(1.*Ngrains_Ngas)))
    file.close()

def makeIC_box_uniform_gas(DIMS=2, N_1D=64, fname='gasgrain_2d_64_unifmu.hdf5', BoxSize=6.):

    Lbox = BoxSize;
    rho_desired = 1.; Ngrains_Ngas = 1; dustgas_massratio=0.01;

    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    x0=np.arange(-0.5,0.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);
    N_1D_dust = np.round((1.*N_1D)*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')
    if(Ngrains_Ngas <= 0): N_1D_dust=10
    x0d=np.arange(-0.5,0.5,1./N_1D_dust); x0d+=0.5*(0.5-x0d[-1]);
    x0d+=0.5/(1.*N_1D_dust)+0.5*(1./(1.*N_1D)-1./(1.*N_1D_dust)); x0d[x0d>0.5]-=1.


    if(DIMS==3):
        xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
        xv_d, yv_d, zv_d = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
    else:
        xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
        xv_d, yv_d = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_d = 0.0*xv_d
    Ngas=xv_g.size; Ngrains = xv_d.size; 
    xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lbox;
    xv_d=xv_d.flatten()*Lbox; yv_d=yv_d.flatten()*Lbox; zv_d=zv_d.flatten()*Lbox;
    m_target_gas = (1.*rho_desired) * ((1.*Lbox)**DIMS) / (1.*Ngas)
    if(Ngrains_Ngas<=0): Ngrains=0
   
    xv_g+=0.5*Lbox; yv_g+=0.5*Lbox; zv_g+=0.5*Lbox; 
    xv_d+=0.5*Lbox; yv_d+=0.5*Lbox; zv_d+=0.5*Lbox;

    #ok=np.where(xv_d >= np.max(xv_d)-0.01*Lbox/N_1D_dust)[0];
    #xv_d=xv_d[ok]-0.08*Lbox; yv_d=yv_d[ok]; zv_d=zv_d[ok]; Ngrains=xv_d.size
    ok=np.where(xv_d >= np.max(xv_d)-0.01*Lbox/N_1D_dust)[0];
    xv_d=xv_d[ok]-0.08*Lbox; yv_d=yv_d[ok]; zv_d=zv_d[ok]; Ngrains=xv_d.size
        
    pylab.close('all'); pylab.plot(xv_g,yv_g,linestyle='',marker=',',color='black')

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    print("Ngrains = %i, chk (%i)"%(Ngrains,xv_d.size))

    if(Ngrains > 0):
        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xv_d+dustgas_massratio*m_target_gas/(1.*Ngrains_Ngas)))
    file.close()


def makeIC_box_unifmu(DIMS=2, N_1D=64, fname='gasgrain_2d_64_unifmu.hdf5',
        Pressure_Bump_Amplitude=1.0, Pressure_Bump_Width=1., BoxSize=6., Ngrains=100000):

    Lbox = BoxSize * Pressure_Bump_Width;
    rho_desired = 1.; Ngrains_Ngas = 1; dustgas_massratio=0.01;

    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    x0=np.arange(-0.5,0.5,1./N_1D); x0+=0.5*(0.5-x0[-1]);
    #N_1D_dust = np.round((1.*N_1D)*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')

    if(DIMS==3):
        xv_g, yv_g, zv_g = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
        xv_d, yv_d, zv_d = np.meshgrid(x0,x0,x0, sparse=False, indexing='xy')
    else:
        xv_g, yv_g = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_g = 0.0*xv_g
        xv_d, yv_d = np.meshgrid(x0,x0, sparse=False, indexing='xy'); zv_d = 0.0*xv_d
    Ngas=xv_g.size; 
    xv_g=xv_g.flatten()*Lbox; yv_g=yv_g.flatten()*Lbox; zv_g=zv_g.flatten()*Lbox;
    xv_d=xv_d.flatten()*Lbox; yv_d=yv_d.flatten()*Lbox; zv_d=zv_d.flatten()*Lbox;
    m_target_gas = (1.*rho_desired) * ((1.*Lbox)**DIMS) / (1.*Ngas)
    m_target_gas *= (1. + Pressure_Bump_Amplitude*(Pressure_Bump_Width/Lbox)*np.sqrt(2.*np.pi)*0.9973); # account for mass 'in bump'
    if(Ngrains_Ngas<=0): Ngrains=0
   
    ## 1-D compression to give desired density ratio:
    dx=0.01/N_1D*Lbox/2.; xt=np.arange(0.,Lbox/2.,dx);
    xt_c=np.cumsum(dx/(1. + Pressure_Bump_Amplitude*np.exp(-xt*xt/(2.*Pressure_Bump_Width*Pressure_Bump_Width))))
    xt=np.append(xt,-xt[1:]); xt_c=np.append(xt_c,-xt_c[1:]); xt=np.sort(xt); xt_c=np.sort(xt_c); 
    xv_g_new = np.interp(xv_g,xt,xt_c)
    
    s0=np.sqrt(2.)*Pressure_Bump_Width/Lbox; # length units of Lbox
    p0=Pressure_Bump_Amplitude*np.sqrt(np.pi)*s0; 
    dx=1.e-4/N_1D; xt=np.arange(-0.5,0.5,dx); 
    ft = 0.5*(1. + (2.*xt + p0*scipy.special.erf(xt/s0)) / (1. + p0*scipy.special.erf(0.5*Lbox/s0)));
    
    f0 = np.random.rand(Ngas); xv_g = np.interp(f0,ft,xt) * Lbox
    yv_g=(np.random.rand(Ngas)-0.5)*Lbox; 
    if(DIMS==3):
        zv_g=(np.random.rand(Ngas)-0.5)*Lbox; 
        
    xv_g+=0.5*Lbox; yv_g+=0.5*Lbox; zv_g+=0.5*Lbox; 
    xv_d+=0.5*Lbox; yv_d+=0.5*Lbox; zv_d+=0.5*Lbox;

    #ok=np.where(xv_d >= np.max(xv_d)-0.01*Lbox/N_1D_dust)[0];
    #xv_d=xv_d[ok]-0.08*Lbox; yv_d=yv_d[ok]; zv_d=zv_d[ok]; Ngrains=xv_d.size
    if xv_d.size > Ngrains:
        ok = np.random.randint(0, xv_d.size, size=Ngrains)
        xv_d = xv_d[ok]
        yv_d = yv_d[ok]
        zv_d = zv_d[ok]
        
    pylab.close('all'); pylab.plot(xv_g,yv_g,linestyle='',marker=',',color='black')

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    print("Ngrains = %i, chk (%i)"%(Ngrains,xv_d.size))

    if(Ngrains > 0):
        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xv_d+dustgas_massratio*m_target_gas/(1.*Ngrains_Ngas)))
    file.close()



def makeIC_stratified(DIMS=2, Nbase=1.0e4, Ngrains_Ngas=1,
        fname='stratbox_2d_N100.hdf5', Lbox_xy=1., Lbox_z=20., rho_target=1.):

    xv_g=np.zeros(0);yv_g=np.zeros(0);zv_g=np.zeros(0);
    xv_d=np.zeros(0);yv_d=np.zeros(0);zv_d=np.zeros(0);
    z0=1.e-10; iter=0; dz=0.; shift=0.;
    while(z0 < Lbox_z):
        print(iter,z0,dz)
        rho=np.exp(-z0) #check exponent for disk
        dz=((Lbox_xy**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS)); N_1D=np.around(Lbox_xy/dz).astype('int');
        x0=np.arange(0.,1.,1./N_1D) + shift;
        while(x0.max() > 1.): x0[(x0 > 1.)]-=1.;
        while(x0.min() < 0.): x0[(x0 < 0.)]+=1.;
            
        #x0+=0.5*(0.5-x0[-1]);
        N_1D_dust = np.round((1.*N_1D)*((1.*Ngrains_Ngas)**(1./(1.*DIMS)))).astype('int')
        x0d=np.arange(0.,1.,1./N_1D_dust) + shift + dz/2.;
        while(x0d.max() > 1.): x0d[(x0d > 1.)]-=1.;
        while(x0d.min() < 0.): x0d[(x0d < 0.)]+=1.;
        if(DIMS==3):
            x, y = np.meshgrid(x0,x0, sparse=False, indexing='xy'); x=x.flatten(); y=y.flatten();
            xd, yd = np.meshgrid(x0d,x0d, sparse=False, indexing='xy'); xd=xd.flatten(); yd=yd.flatten();
        else:
            x=x0; xd=x0d; y=0.*x; yd=y;
        z=z0+dz/2.+np.zeros(x.size); zd=z0+dz/2.+np.zeros(xd.size);
        print(x.shape,y.shape,z.shape)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z)
        xv_d = np.append(xv_d,xd)
        yv_d = np.append(yv_d,yd)
        zv_d = np.append(zv_d,zd)
        z0 += dz; iter += 1;
        shift += dz/2.;
    if(DIMS<3):
        yv_d=1.*zv_d; zv_d=0.*zv_d;
        yv_g=1.*zv_g; zv_g=0.*zv_g;
        
    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    Ngas=xv_g.size; Ngrains=xv_d.size;
    m_target_gas = (1.-np.exp(-Lbox_z))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)

    print(Ngas,Ngrains,m_target_gas)
    pylab.close('all')
    #pylab.axis([0.,1.,0.,1.])
    pylab.plot(xv_g,yv_g,marker='.',color='black',linestyle='',rasterized=True);
    pylab.plot(xv_d,yv_d,marker='.',color='red',linestyle='',rasterized=True);

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,Ngrains,0,0]) # we have gas and particles we will set for type 3 here, zero for all others
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 1.0; # box size
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 1.0; # z=0 Omega_matter
    h.attrs['OmegaLambda'] = 0.0; # z=0 Omega_Lambda
    h.attrs['HubbleParam'] = 1.0; # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
    h.attrs['Flag_Sfr'] = 0; # flag indicating whether star formation is on or off
    h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
    h.attrs['Flag_StellarAge'] = 0; # flag indicating whether stellar ages are to be saved
    h.attrs['Flag_Metals'] = 0; # flag indicating whether metallicity are to be saved
    h.attrs['Flag_Feedback'] = 0; # flag indicating whether some parts of springel-hernquist model are active
    h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
    h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

    # start with particle type zero. first (assuming we have any gas particles) create the group
    p = file.create_group("PartType0")
    q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    p = file.create_group("PartType3")
    q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
    p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_d+m_target_gas/(1.*Ngrains_Ngas)))
    file.close()

makeIC_box_uniform_gas(DIMS=3, N_1D=128, fname='gasgrain_3d_128_unifmu.hdf5', BoxSize=6.)
makeIC_stratified(DIMS=3, Nbase=1.0e4, Ngrains_Ngas=1,
        fname='stratbox_3d_N100.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)
