import numpy as np
import h5py as h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as pylab
import scipy.special
import pdb
import matplotlib.pyplot as plt
import random
import scipy.integrate as integrate

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

def makeIC_keplerian_disk_2d_old_1(Nbase=1.0e4, fname='keplerian_disk_2d.hdf5', p=0., r_in=0.1, r_out=2., rho_target=1.):
    #Note: even though the disk only starts at r=0.1, the Nbase refers to the center of the disk (so N < Nbase at the start of the actual disk)
    DIMS=2
    G=1.0; M=1.0;

    rv_g=np.zeros(0);phiv_g=np.zeros(0);
    rv_d=np.zeros(0);phiv_d=np.zeros(0);

    phi_disk=2.*np.pi
    r0=r_in; iter=0; dr=0.; shift=0.;
    while(r0 < r_out):
        #Here rho is the surface density of the disk in the radial direction
        rho=surf_rho_profile(r=r0,p=p) #TODO: need radial distribution of the disk density
        dr=((phi_disk**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS))
        N_1D=np.around(phi_disk/dr).astype('int');
        phi0=np.arange(0.,phi_disk,1./N_1D) + shift;
        while(phi0.max() > phi_disk): phi0[(phi0 > phi_disk)]-=phi_disk;
        while(phi0.min() < 0.): phi0[(phi0 < 0.)]+=phi_disk;

        phi=phi0; r=r0+dr/2.+np.zeros(phi.size);
        phiv_g = np.append(phiv_g,phi)
        rv_g = np.append(rv_g,r)
        r0 += dr; iter += 1;
        shift += dr/2.;

    Ngas=phiv_g.size
    m_target_gas = (phi_disk/(p+2.))*(surf_rho_profile(r=r_out,p=p+2.)-surf_rho_profile(r=r_in,p=p+2.))/ (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed
    #m_target_gas = (surf_rho_profile(r=r_in,p=p)-surf_rho_profile(r=r_out,p=p))*(phi_disk**(DIMS-1))*rho_target / (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed
    m_target_gas_test = (1.-surf_rho_profile(r=r_out,p=p))*(phi_disk**(DIMS-1))*rho_target / (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed

    #Convert polar coordinates to cartesian:
    # xv_g = rv_g*np.cos(phiv_g)+r_out
    # yv_g = rv_g*np.sin(phiv_g)+r_out

    #TEMPORARILY MAKING COORDS THE SAME AS PHIL'S
    xv_g = rv_g*np.cos(phiv_g)+4.
    yv_g = rv_g*np.sin(phiv_g)+4.

    zv_g=0.*yv_g

    print(Ngas,m_target_gas, m_target_gas_test)

    #Velocity components
    kep_velocity_x = -np.sqrt(G*M/rv_g)*np.sin(phiv_g)
    kep_velocity_y = np.sqrt(G*M/rv_g)*np.cos(phiv_g)
    kep_velocity_z = 0.*kep_velocity_y

    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,0,0,0]) # we have gas we will set for type 3 here, zero for all others (including particles bc there is only gas)
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    #h.attrs['BoxSize'] = 1.0; # box size #TODO: Check if its okay to just remove this
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
    vels = np.zeros((Ngas,3)); vels[:,0]=kep_velocity_x; vels[:,1]=kep_velocity_y; vels[:,2]=kep_velocity_z;
    p.create_dataset("Coordinates",data=q)
    #p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("Velocities",data=vels)
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))
    file.close()

    exit()

    plt.plot(xv_g,yv_g, marker = '.', markersize=0.5, linestyle='None')
    plt.show()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(phiv_g, rv_g, s=1.)
    ax.set_rmax(2)
    ax.set_rticks([0.1,0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    #ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()
    return;

######################################################################################################################################################
def makeIC_keplerian_disk_2d_old_2(N_1D_base=128, fname='keplerian_disk_2d.hdf5', p=0., r_in=0.1, r_out=2., rho_target=1., m_target_gas=2e-4):

    DIMS=2
    G=1.0; M=1.0; #in code units

    all_dr=np.zeros(0); all_rho=np.zeros(0); all_N_1D=np.zeros(0); all_r=np.zeros(0);

    rv_g=np.zeros(0);phiv_g=np.zeros(0);
    rv_d=np.zeros(0);phiv_d=np.zeros(0);

    phi_disk=2.*np.pi

    #bc the disk does not start at the center but at r_in, so we want rho=1 at r_in, same for N_1D
    rho0 = rho_target/surf_rho_profile(r=r_in,p=p)
    # N_1D_0 = N_1D_base/surf_rho_profile(r=r_in,p=p+1)
    # dist_bw_particles = phi_disk*r_in/N_1D_base
    dr = N_1D_base*m_target_gas/(phi_disk*r_in*rho_target)
    print(dr);

    r_cur=r_in; iter=0; shift=0.; #dr=0.;
    while(r_cur < r_out):
        #Here rho is the surface density of the disk in the radial direction
        rho_cur=rho0*surf_rho_profile(r=r_cur,p=p) #Constant if p=0, otherwise p=-1 or p=-3/2
        # N_1D_cur = N_1D_0*surf_rho_profile(r=r_cur,p=p+1)
        N_1D_cur = phi_disk*r_cur*rho_cur*dr/m_target_gas
        #print(N_1D_cur);
        phi_cur=np.arange(0.,phi_disk,phi_disk/N_1D_cur) #+ shift #TMP add of shift
        # dr = N_1D_cur*m_target_gas/(phi_disk*r_cur*rho_cur)

        #TMP add f while loops here
        # while(phi_cur.max() > phi_disk): phi_cur[(phi_cur > phi_disk)]-=phi_disk;
        # while(phi_cur.min() < 0.): phi_cur[(phi_cur < 0.)]+=phi_disk;
        # print(phi_cur);

        all_dr = np.append(all_dr,dr); all_rho = np.append(all_rho,rho_cur); all_N_1D = np.append(all_N_1D,N_1D_cur); all_r=np.append(all_r,r_cur);

        #Add phi,r coords of new ring of particles
        phi=phi_cur; r=r_cur+np.zeros(phi.size);
        phiv_g = np.append(phiv_g,phi)
        rv_g = np.append(rv_g,r)
        r_cur += dr; iter += 1;
        # shift += dr/2.; #TMP

    #FOR TESTING:
    # print(all_rho); print(all_N_1D); print(all_N_1D/all_r); print(1./all_dr); print(all_r**p);
################################################################################
        # dr=((phi_disk**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS))
        # N_1D=np.around(phi_disk/dr).astype('int');
        # phi0=np.arange(0.,phi_disk,1./N_1D) + shift;
        # while(phi0.max() > phi_disk): phi0[(phi0 > phi_disk)]-=phi_disk;
        # while(phi0.min() < 0.): phi0[(phi0 < 0.)]+=phi_disk;

        # phi=phi0; r=r_cur+dr/2.+np.zeros(phi.size);
        # phiv_g = np.append(phiv_g,phi)
        # rv_g = np.append(rv_g,r)
        # r_cur += dr; iter += 1;
        # shift += dr/2.;
################################################################################

    Ngas=phiv_g.size
    # m_target_gas = (phi_disk/(p+2.))*(surf_rho_profile(r=r_out,p=p+2.)-surf_rho_profile(r=r_in,p=p+2.))/ (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed
    # #m_target_gas = (surf_rho_profile(r=r_in,p=p)-surf_rho_profile(r=r_out,p=p))*(phi_disk**(DIMS-1))*rho_target / (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed
    # m_target_gas_test = (1.-surf_rho_profile(r=r_out,p=p))*(phi_disk**(DIMS-1))*rho_target / (1.*Ngas) #TODO: fix the exp term to be for a disk and think about inner radius of disk being removed

    #Convert polar coordinates to cartesian:
    xv_g = rv_g*np.cos(phiv_g)+r_out
    yv_g = rv_g*np.sin(phiv_g)+r_out

    #TEMPORARILY MAKING COORDS THE SAME AS PHIL'S
    # xv_g = rv_g*np.cos(phiv_g)+4.
    # yv_g = rv_g*np.sin(phiv_g)+4.

    zv_g=0.*yv_g

    # print(rv_g[0:int(all_N_1D[0])+1])
    # print(phiv_g[0:int(all_N_1D[0])+1])

    #Velocity components
    kep_velocity_x = -np.sqrt(G*M/rv_g)*np.sin(phiv_g)
    kep_velocity_y = np.sqrt(G*M/rv_g)*np.cos(phiv_g)
    kep_velocity_z = 0.*kep_velocity_y

    kep_velocity_mag = np.sqrt(kep_velocity_x**2 + kep_velocity_y**2)

    print(xv_g[0:int(all_N_1D[0])+1])
    print(yv_g[0:int(all_N_1D[0])+1])

    print(kep_velocity_x[0:int(all_N_1D[0])+1])
    print(kep_velocity_y[0:int(all_N_1D[0])+1])
    #print(kep_velocity_mag[0:int(all_N_1D[0])+1])
    # exit()
    file = h5py.File(fname,'w')
    npart = np.array([Ngas,0,0,0,0,0]) # we have gas we will set for type 3 here, zero for all others (including particles bc there is only gas)
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 8.0; # box size #TODO: Check if its okay to just remove this
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 0.0; # z=0 Omega_matter
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
    vels = np.zeros((Ngas,3)); vels[:,0]=kep_velocity_x; vels[:,1]=kep_velocity_y; vels[:,2]=kep_velocity_z;
    p.create_dataset("Coordinates",data=q)
    #p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("Velocities",data=vels)
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+9.e-5))
    file.close()

    exit()

    #For testing purposes (remove exit command above to use)

    #Plot using Cartesian coords
    plt.plot(xv_g,yv_g, marker = '.', markersize=0.5, linestyle='None')
    plt.show()

    #Plot using polar coords
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(phiv_g, rv_g, s=1.)
    ax.set_rmax(2)
    ax.set_rticks([0.1,0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    #ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()
    return;

######################################################################################################################################################

def surf_rho_profile(r,p=0.):
    return r**p

def omega_kep(r):
    return r**(-3./2.)

def makeIC_keplerian_disk_2d(fname='keplerian_disk_2d.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5,
                                p=0., r_in=0.1, r_out=2., width_ghost_in=0.0, width_ghost_out=0.0, rho_target=1., m_target_gas=2e-4, 
                                vary_particle_mass=False, num_particle_r_in = 500, include_dust=False):
    #For testing
    all_dr=np.zeros(0); all_rho=np.zeros(0); all_N_1D=np.zeros(0); all_r=np.zeros(0);

    #Some initial calculations --ONLY INCLUDE ONCE USING TEMP GRADIENT
    r_ref = 1.
    c_s_ref = 0.05
    internal_energy_ref = c_s_ref**2/(gamma-1.)
    T_ref = c_s_ref**2
    #Here, the reference point is r = 1
    T_0 = T_ref*r_ref**0.5
    print("T_0: ", T_0)

    #bc the disk does not start at the center but at r_in, so we want rho=1 at r_in, same for N_1D
    #Here, the reference point is r_in
    rho0 = rho_target/surf_rho_profile(r=r_in,p=p) #TODO: decide what you want to value of rho, although maybe doesn't really matter
    
    #Variables to be set
    DIMS=2
    G=1.0; M=1.0; #in code units
    phi_disk=2.*np.pi
    dr_factor=dr_factor #for 10% of scale height
    # c_s = np.sqrt((gamma-1)*internal_energy) #sound_speed (will be in loop when using TEMP GRADIENT)

    if(vary_particle_mass == True):
        #m_particle = m_target_gas_0 * r**(5/4) / ln(r)
        n_0 = num_particle_r_in*np.log(r_in+1)
        print(n_0)
        m_target_gas_0 = (phi_disk * dr_factor * rho0 * T_0**0.5 / n_0) * np.log(r_in+1)
        print(m_target_gas_0)
        # exit()
    #Initialize coord arrays for gas and dust particles (here in polar coords)
    rv_g=np.zeros(0);phiv_g=np.zeros(0);
    rv_d=np.zeros(0);phiv_d=np.zeros(0);
    internal_energy_g = np.zeros(0);
    m_target_gas_array = np.zeros(0);

    Nghost_in=0; Nghost_out=0;

    print("dr factor: ", dr_factor)
    print("Gamma: ", gamma)
    print("Internal energy: ", internal_energy)
    # print("Sound speed: ", c_s)
    print("p: ", p)
    print("Inner radius: ", r_in)
    print("Outer radius: ", r_out)
    print("Target density: ", rho_target)
    print("Particle mass: ", m_target_gas)

    r_cur=1.0*r_in-width_ghost_in; iter=0; dr=0.;
    while(r_cur <= r_out+width_ghost_out):
        #--ONLY INCLUDE ONCE USING TEMP GRADIENT
        T_cur = T_0*r_cur**-0.5
        c_s = T_cur**0.5
        print("r_cur: ", r_cur)
        print("c_s(r_in): ", c_s)
        print("H(r_in): ", c_s/omega_kep(r_cur))
        # exit()
        internal_energy_cur = c_s**2/(gamma-1.)

        #Here rho is the surface density of the disk in the radial direction
        rho_cur=rho0*surf_rho_profile(r=r_cur,p=p) #Constant if p=0, otherwise p=-1 or p=-3/2
        dr = dr_factor*c_s/omega_kep(r_cur)
        print('LOOK HERE: ', dr)

        #Check if dr < 0.0027: This is for trying to spread the particles rings out more at the inner radius.
        
        # if (dr < 0.0027):
        #     dr = 0.0027; print('dr < 0.0027, iter: ', iter);                
        #     if(vary_particle_mass==True):
        #         #the 1.85 was determined a by just seeing the difference bw expected vs output, so a little random
        #         #need to figure out exactly where its coming from
        #         m_particle_cur = m_target_gas_0 / 1.85 / np.log(r_cur+1)
        #         m_target_gas = m_particle_cur
        # elif(vary_particle_mass==True):
        #     m_particle_cur = m_target_gas_0 * r_cur**(5/4) / np.log(r_cur+1)
        #     m_target_gas = m_particle_cur
        if(vary_particle_mass==True):
            m_particle_cur = m_target_gas_0 * r_cur**(5/4) / np.log(r_cur+1)
            m_target_gas = m_particle_cur

        N_1D_cur = np.round(phi_disk*r_cur*rho_cur*dr/m_target_gas)
        print("LOOK AT THIS: ", N_1D_cur)
        if (N_1D_cur<1.0):
            r_cur += dr; iter += 1;
            continue

        dphi_cur = phi_disk/N_1D_cur
        phi_cur=np.arange(0.,phi_disk,dphi_cur)
        
        random.seed(iter); shift = random.randint(0, 11);
        phi_cur = phi_cur + (shift*np.pi/6)

        if(iter<100):
            print("Phi: ", phi_cur)
        #For testing
        all_dr = np.append(all_dr,dr); all_rho = np.append(all_rho,rho_cur); all_N_1D = np.append(all_N_1D,N_1D_cur); all_r=np.append(all_r,r_cur);

        #Add phi,r coords of new ring of particles
        phi=1.0*phi_cur; r=r_cur+np.zeros(phi.size);
        phiv_g = np.append(phiv_g,phi)
        rv_g = np.append(rv_g,r)

        #determine number of ghost particles
        if(r_cur<r_in):
            Nghost_in+=phi.size;
        if(r_cur>r_out):
            Nghost_out+=phi.size;

        #--ONLY INCLUDE ONCE USING TEMP GRADIENT
        internal_energy=internal_energy_cur+np.zeros(phi.size)
        internal_energy_g = np.append(internal_energy_g, internal_energy)

        mass_gas_cur_array = m_target_gas+np.zeros(phi.size)
        m_target_gas_array = np.append(m_target_gas_array, mass_gas_cur_array)

        r_cur += dr; iter += 1;

    print("iterations: ", iter)
    print("dr: ", all_dr)
    print("rho: ", all_rho)
    print("N_1D: ", all_N_1D[0:500])
    print("length of N_1D: ", len(all_N_1D))
    print("total number of perticles: ", np.sum(all_N_1D))
    # exit()
    Ngas=phiv_g.size

    #Convert polar coordinates to cartesian:
    xv_g = rv_g*np.cos(phiv_g)+r_out+width_ghost_out
    yv_g = rv_g*np.sin(phiv_g)+r_out+width_ghost_out

    #TEMPORARILY MAKING COORDS THE SAME AS PHIL'S
    # xv_g = rv_g*np.cos(phiv_g)+4.
    # yv_g = rv_g*np.sin(phiv_g)+4.

    zv_g=0.*yv_g

    print("x_min, x_max: ", np.min(xv_g), np.max(xv_g))
    print("y_min, y_max: ", np.min(yv_g), np.max(yv_g))

    if(include_dust):
        print("Including Dust")
        ok = np.where(rv_g==rv_g[-1])
        phiv_d = phiv_g[ok]
        rv_d = rv_g[-1] + 0.01 + np.zeros(phiv_d.size)
        Ngrains = phiv_d.size
        #Convert polar coordinates to cartesian:
        xv_d = rv_d*np.cos(phiv_d)+r_out+width_ghost_out
        yv_d = rv_d*np.sin(phiv_d)+r_out+width_ghost_out
        zv_d=0.*yv_d

    #Velocity components with keplerian angular velocity
    # kep_velocity_x = -np.sqrt(G*M/rv_g)*np.sin(phiv_g)
    # kep_velocity_y = np.sqrt(G*M/rv_g)*np.cos(phiv_g)
    # kep_velocity_z = 0.*kep_velocity_y

    #Velocity components with sub-keplerian angular velocity (only density gradient)
    # kep_velocity_x = -np.sqrt(omega_kep(rv_g)**2 * rv_g**2 + (p-3/2)*c_s**2)*np.sin(phiv_g)
    # kep_velocity_y = np.sqrt(omega_kep(rv_g)**2 * rv_g**2 + (p-3/2)*c_s**2)*np.cos(phiv_g)
    # kep_velocity_z = 0.*kep_velocity_y


    #Velocity components with sub-keplerian angular velocity (density gradient and TEMP GRADIENT)
    kep_velocity_x = -np.sqrt(omega_kep(rv_g)**2 * rv_g**2 + (p-7/4)*T_0/rv_g**0.5) * np.sin(phiv_g)
    kep_velocity_y = np.sqrt(omega_kep(rv_g)**2 * rv_g**2 + (p-7/4)*T_0/rv_g**0.5) * np.cos(phiv_g)
    kep_velocity_z = 0.*kep_velocity_y

    #Put ghost particles with keplerian velocity since they will not feel any forces: -- actually don't do this
    # kep_velocity_x[0:Nghost_in] = -omega_kep(rv_g[0:Nghost_in]) * rv_g[0:Nghost_in] * np.sin(phiv_g[0:Nghost_in])
    # kep_velocity_y[0:Nghost_in] = omega_kep(rv_g[0:Nghost_in]) * rv_g[0:Nghost_in] * np.cos(phiv_g[0:Nghost_in])

    # kep_velocity_x[Ngas-Nghost_out:] = -omega_kep(rv_g[Ngas-Nghost_out:]) * rv_g[Ngas-Nghost_out:] * np.sin(phiv_g[Ngas-Nghost_out:])
    # kep_velocity_y[Ngas-Nghost_out:] = omega_kep(rv_g[Ngas-Nghost_out:]) * rv_g[Ngas-Nghost_out:] * np.cos(phiv_g[Ngas-Nghost_out:])

    #Give ghost particles v=0:
    # kep_velocity_x[0:Nghost_in] = 0.0
    # kep_velocity_y[0:Nghost_in] = 0.0
    # kep_velocity_x[Ngas-Nghost_out:] = 0.0
    # kep_velocity_y[Ngas-Nghost_out:] = 0.0

    kep_velocity_mag = np.sqrt(kep_velocity_x**2 + kep_velocity_y**2)

    #FOR TESTING:
    v_phi_theoretical = np.sqrt(omega_kep(all_r)**2 * all_r**2 + (p-7/4)*T_0/all_r**0.5)

    centrifugal_theoretical = v_phi_theoretical*v_phi_theoretical/all_r
    phi_dot = ((xv_g-(r_out+width_ghost_out))*kep_velocity_y - (yv_g-(r_out+width_ghost_out))*kep_velocity_x) / ((xv_g-(r_out+width_ghost_out))**2 + (yv_g-(r_out+width_ghost_out))**2)

    phi_dot_by_radius = np.zeros(0)

    for i in range(len(all_r)):
        r_cur = all_r[i]
        ok = np.where(rv_g == r_cur)
        print(len(ok[0]))
        phi_dot_cur = np.average(phi_dot[ok])
        phi_dot_by_radius = np.append(phi_dot_by_radius,phi_dot_cur)

    centrifugal = phi_dot_by_radius*phi_dot_by_radius*all_r
    grav = 1/(all_r*all_r)
    sub_kep_diff_theoretical = grav - centrifugal_theoretical
    sub_kep_diff_IC = grav - centrifugal
    diff = centrifugal - centrifugal_theoretical
    plt.figure()
    # plt.plot(all_r, centrifugal_theoretical, marker='.', label='theoretical centrifugal')
    # plt.plot(all_r, phi_dot_by_radius,  marker='.', label='IC phi_dot')
    # plt.plot(all_r, centrifugal, marker='.', label='IC centrifugal')
    # plt.plot(all_r,grav, label='gravity')
    plt.plot(all_r, diff, marker='.', label='cent_IC - cent_thr')
    # plt.plot(all_r, sub_kep_diff_theoretical, marker='.', label='kep - sub_kep theoretical')
    # plt.plot(all_r, sub_kep_diff_IC, marker='.', label='kep - sub_kep IC')
    plt.title("Direct from makeIC")
    plt.xlabel('r')
    plt.ylabel('radial accel')
    plt.legend()
    plt.show()

    p_fit = np.polyfit(all_r, all_N_1D, 1)
    print(p_fit)

    print(num_particle_r_in*np.log(all_r+1)/all_N_1D)
    plt.figure()
    plt.plot(all_r, all_N_1D, marker='.', label='num particles from IC')
    # plt.plot(all_r, p_fit[0]*np.log(all_r) + p_fit[1], label='best fit')
    plt.plot(all_r, num_particle_r_in*np.log(all_r+1), label='log(r)')
    plt.title("Direct from makeIC")
    plt.xlabel('r')
    plt.ylabel('Number of particles')
    plt.legend()
    plt.show()
    # exit()

    #END TESTING.

    if(include_dust):
        #Velocity components of the dust particles(keplerian in phi direction, but TODO: also need some radial drift)
        kep_velocity_x_dust = -np.sqrt(G*M/rv_d)*np.sin(phiv_d)
        kep_velocity_y_dust = np.sqrt(G*M/rv_d)*np.cos(phiv_d)

        v_hw = 0. #TODO:set this
        tau = 0. #TODO:set this
        radial_velocity = 2*v_hw*tau/(tau**2+1.)
        kep_velocity_x_dust += radial_velocity*np.cos(phiv_d)
        kep_velocity_y_dust += radial_velocity*np.sin(phiv_d)
        
        kep_velocity_z_dust = 0.*kep_velocity_y_dust
        print("Num dust particles: ", len(xv_d))

    print("Num gas particles: ", len(xv_g))
    # exit()
    file = h5py.File(fname,'w')
    if(include_dust):
        npart = np.array([Ngas,0,0,Ngrains,0,0])
    else:
        npart = np.array([Ngas,0,0,0,0,0]) # we have gas we will set for type 3 here, zero for all others (including particles bc there is only gas)
    h = file.create_group("Header");
    h.attrs['NumPart_ThisFile'] = npart; # npart set as above - this in general should be the same as NumPart_Total, it only differs
    h.attrs['NumPart_Total'] = npart; # npart set as above
    h.attrs['NumPart_Total_HighWord'] = 0*npart; # this will be set automatically in-code (for GIZMO, at least)
    h.attrs['MassTable'] = np.zeros(6); # these can be set if all particles will have constant masses for the entire run. however since
    h.attrs['Time'] = 0.0;  # initial time
    h.attrs['Redshift'] = 0.0; # initial redshift
    h.attrs['BoxSize'] = 8.0; # box size #TODO: Check if its okay to just remove this
    h.attrs['NumFilesPerSnapshot'] = 1; # number of files for multi-part snapshots
    h.attrs['Omega0'] = 0.0; # z=0 Omega_matter
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
    vels = np.zeros((Ngas,3)); vels[:,0]=kep_velocity_x; vels[:,1]=kep_velocity_y; vels[:,2]=kep_velocity_z;
    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=vels)
    particle_IDs = np.append(np.append(np.zeros(Nghost_in),np.arange(1,Ngas-Nghost_in-Nghost_out+1)), np.zeros(Nghost_out))
    p.create_dataset("ParticleIDs",data=particle_IDs)
    # p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("Masses",data=m_target_gas_array)
    # p.create_dataset("InternalEnergy",data=(0.*xv_g+internal_energy))
    #Use this once including TEMP GRADIENT
    p.create_dataset("InternalEnergy",data=internal_energy_g)
    print(particle_IDs[Nghost_in-10:Nghost_in+500])
    print(" Nghost in and out: ", Nghost_in, Nghost_out)
    print("Ngas no ghost: ", Ngas-Nghost_in-Nghost_out)
    print("Ngas: ", Ngas,len(particle_IDs))

    if(include_dust):
        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3)); q[:,0]=xv_d; q[:,1]=yv_d; q[:,2]=zv_d;
        vels = np.zeros((Ngrains,3)); vels[:,0]=kep_velocity_x_dust; vels[:,1]=kep_velocity_y_dust; vels[:,2]=kep_velocity_z_dust;
        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=vels)
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xv_g+0.1*m_target_gas))
        # p.create_dataset("InternalEnergy",data=(0.*xv_g+internal_energy))

    file.close()
######################################################################################################################################################

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
        print(x0)
        while(x0.max() > 1.): x0[(x0 > 1.)]-=1.;
        while(x0.min() < 0.): x0[(x0 < 0.)]+=1.;
        print(x0)

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

def makeIC_disk_stratified_no_dust(DIMS=2, Nbase=1.0e4, dustgas_massratio=0.01,
        fname='stratbox_2d_N100.hdf5', Lbox_xy=1., Lbox_z=20., rho_target=1., include_dust=False):

    #For testing purposes:
    N_1D_all = []
    z0_all = []
    dz_all = []
    counter = 0.0;
    z_up_all = []
    z_low_all = []

    xv_g=np.zeros(0);yv_g=np.zeros(0);zv_g=np.zeros(0);
    xd=np.zeros(0);yd=np.zeros(0);zd=np.zeros(0);

    z0=1.e-10; iter=0; dz=0.; shift=0.;
    while(z0 < Lbox_z/2.): #here dividing box z length by two to get density porfile for just one side of midplane, then just copy for other half (above/below)
        print(iter,z0,dz)
        rho=np.exp(-z0**2) #check exponent for disk
        #Testing dz and N_1D to determine problem with profile density
        #dz=((Lbox_xy) / (rho*Nbase))**(1./(1.*DIMS)); #Tester
        dz=((Lbox_xy**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS)); #Original
        #N_1D=np.around(Lbox_xy**(DIMS-1.)/dz).astype('int'); print('N_1D: ' + str(N_1D)); counter+=N_1D**2; #Tester
        N_1D=np.around(Lbox_xy/dz).astype('int'); print('N_1D: ' + str(N_1D)); counter+=N_1D**2; #Original
        x0=np.arange(0.,1.,1./N_1D) + shift;
        #print("Initial x0: ", x0)
        while(x0.max() > 1.): x0[(x0 > 1.)]-=1.;
        while(x0.min() < 0.): x0[(x0 < 0.)]+=1.;
        #print("Final x0: ", x0)

        #Save N_1D at each z0 value to plot later:
        N_1D_all.append(N_1D)
        z0_all.append(z0)
        dz_all.append(dz)

        #x0+=0.5*(0.5-x0[-1]);
        if(DIMS==3):
            x, y = np.meshgrid(x0,x0, sparse=False, indexing='xy'); x=x.flatten(); y=y.flatten();
            print(np.max(x), np.max(y))
        else:
            x=x0; y=0.*x;

        z=z0+dz/2.+np.zeros(x.size); #Original
        z_upper = z + (Lbox_z/2.); #for upper disk
        z_lower = (Lbox_z/2.) - z; #for lower disk
        z_lower[(z_lower < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
        z_up_all.append(z_upper[0]); z_low_all.append(z_lower[0]);

        #print(x.shape,y.shape,z.shape)
        print(x.shape,y.shape,z_upper.shape,z_lower.shape)

        #For upper disk (above midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_upper) #fix z for upper

        #For lower disk (below midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_lower) #fix z for lower

        #dust
        if(include_dust):
            z_full_d=z0+dz/2.+np.zeros(x0.size);
            z_upper_d = z_full_d + (Lbox_z/2.); #for upper disk
            z_lower_d = (Lbox_z/2.) - z_full_d; #for lower disk
            z_lower_d[(z_lower_d < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
            yd = np.append(yd, x0) #upper disk
            zd = np.append(zd, z_upper_d)
            yd = np.append(yd, x0) #lower disk
            zd = np.append(zd, z_lower_d)

        z0 += dz; iter += 1;
        shift += dz/2.;

    Ngrains = 0;
    if(include_dust):
        #dust needs to be xv_d = 1.0 (or use above value)
        #yv_d = 0-1, zv_d = 0-Lbox_z 
        #-> uniform distribution, start by trying separation of 0.1 in z direction and 0.0035 in y direction
        #need to determine this from proper dust mass ratio of a disk

        #this is being removed because it is being done in the while loop instead to have uniform gas-dust ratio
        # y0 = np.arange(0., 1., 1./N_1D_all[0])
        # z0 = np.arange(0., Lbox_z, 0.1)
        # yd, zd = np.meshgrid(y0,z0, sparse=False, indexing='xy'); yd=yd.flatten(); zd=zd.flatten();

        xd_max_coord = 1.0
        xd = xd_max_coord + 0.0*yd
        Ngrains = xd.size

    print("##################Checking lower plane coords:################## ");
    #print(z_lower[0], z[0], dz);
    #print(dz_all);
    print(z_up_all); print(z_low_all);

    if(DIMS<3):
        yv_g=1.*zv_g; zv_g=0.*zv_g;
        
    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    Ngas=xv_g.size;
    m_target_gas = (1.-np.exp(-Lbox_z))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)
    m_target_gas_test = (1.-np.exp(-Lbox_z**2))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)

    print("###########HERE#########")
    print(Ngas,m_target_gas, m_target_gas_test)
    print("###########DONE#########")

    exit()
    #Calculate gas density at midplane
    density_at_all_z = np.zeros(len(N_1D_all))
    for i in range(len(N_1D_all)):
        density_at_all_z[i] = (N_1D_all[i]**2)*m_target_gas / (Lbox_xy*Lbox_xy*dz_all[i])

    density_overall = integrate.trapezoid(density_at_all_z, dz_all)
    print("Density overall: ", density_overall)
    
    density_at_miplane = (N_1D_all[0]**2)*m_target_gas / (Lbox_xy*Lbox_xy*dz_all[0])
    print("Density at the midplane: ", density_at_miplane)

    #For testing dust-to-gas mass ratio:
    time = 1200
    num_Pgas_outer_edge = len(np.where(xv_g*12.0>=(12 - time*12/1200))[0])
    print("Num gas particles in outer edge: ", num_Pgas_outer_edge)
    total_gas_mass_outer_edge = num_Pgas_outer_edge*m_target_gas

    total_dust_mass_outer_edge_thr = int(time/4.208)*15606*m_target_gas*dustgas_massratio

    print("dust-to-gas ratio: ", total_dust_mass_outer_edge_thr/total_gas_mass_outer_edge)

    pylab.close('all')
    #pylab.axis([0.,1.,0.,1.])
    # pylab.plot(xv_g,yv_g,marker='.',color='black',linestyle='',rasterized=True);

    # exit()
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

    #q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g; print(xv_g); print(yv_g); print(zv_g);
    #n_part_gas = 128**3
    #print(1 + (np.random.random(size=n_part_gas)-.5)*1e-3); exit();
    q=np.zeros((Ngas,3)); 
    q[:,0]=(xv_g * (1 + (np.random.random(size=len(xv_g))-.5)*1e-3))*Lbox_xy; 
    q[:,1]=(yv_g * (1 + (np.random.random(size=len(yv_g))-.5)*1e-3))*Lbox_xy; 
    q[:,2]=zv_g * (1 + (np.random.random(size=len(zv_g))-.5)*1e-3);

    plt.figure()
    plt.plot(q[:,1],q[:,2], marker='.', linestyle='None')
    plt.show()

    print(xv_g); print(q[:,0]);
    print(yv_g); print(q[:,1]);
    print(zv_g); print(q[:,2]);

    print(np.min(q[:,0]), np.max(q[:,0])); print(np.min(q[:,1]),  np.max(q[:,1])); print(np.min(q[:,2]),  np.max(q[:,2]));

    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=(0.*xv_g+m_target_gas))
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    if(include_dust):

        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3));

        x_temp = xd * (1 + (np.random.random(size=len(xd))-.5)*1e-3) * Lbox_xy;
        x_temp[np.where(x_temp>12.0)] = 11.999;

        y_temp = yd * (1 + (np.random.random(size=len(yd))-.5)*1e-3) * Lbox_xy;
        y_temp[np.where(y_temp>12.0)] = 11.999;

        print(len(np.where(x_temp>12.0)[0]))

        #to ensure no particles are at x > 1.0 -- doesn't work well when I run it with this IC
        # ok = np.where(x_temp > 1.0)
        # print("Checking if any dust particles are at x>1 before: ", ok)
        # max_off = np.max(x_temp[ok]-1.0)
        # x_temp -= max_off

        q[:,0]=x_temp
        q[:,1]=y_temp

        q[:,2]=zd * (1 + (np.random.random(size=len(zd))-.5)*1e-3);

        print("Checking if any dust particles are at x>1 after: ", np.where(q[:,0] > 1.0*Lbox_xy))
        print(np.min(q[:,0]), np.max(q[:,0]))
        print(np.min(q[:,1]), np.max(q[:,1]))
        print(np.min(q[:,2]), np.max(q[:,2]))
        print(xd); print(q[:,0]);
        print(yd); print(q[:,1]);
        print(zd); print(q[:,2]);

        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=(0.*xd + dustgas_massratio*m_target_gas))

    print(len(np.arange(1,Ngas+1))); print(counter); print(len(q));
    file.close()

    #plot dust:
    if(include_dust):
        print("xd: ", q[:,0][0:500])
        plt.figure()
        plt.plot(q[:,0],q[:,1], marker='.', linestyle='None')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    exit()

    #Save csv file of N_1D and z0:
    N_1D_all = np.asarray(N_1D_all); z0_all = np.asarray(z0_all); z_up_all=np.asarray(z_up_all); z_low_all=np.asarray(z_low_all);
    np.savetxt("density_IC_disk.csv", N_1D_all, delimiter=",")
    np.savetxt("z0_IC_disk.csv", z0_all, delimiter=",")
    np.savetxt("dz_IC_disk.csv", dz_all, delimiter=",")
    np.savetxt("z_up_IC_disk.csv", z_up_all, delimiter=",")
    np.savetxt("z_low_IC_disk.csv", z_low_all, delimiter=",")


def makeIC_disk_stratified_high_res(DIMS=3, N_1D=256, m_gas_0=4e-5, dustgas_massratio=0.01,
        fname='stratbox_2d_N100.hdf5', Lbox_xy=12., Lbox_z=8., include_dust=False):

    #For testing purposes:
    N_1D_all = []
    m_gas_all = []
    z0_all = []
    dz_all = []
    counter = 0.0;
    z_up_all = []
    z_low_all = []

    xv_g=np.zeros(0);yv_g=np.zeros(0);zv_g=np.zeros(0);
    xd=np.zeros(0);yd=np.zeros(0);zd=np.zeros(0);

    m_gas_array=np.zeros(0); m_dust_array=np.zeros(0);

    z0=1.e-10; iter=0; dz=0.; shift=0.;
    Nbase = N_1D**3 / (np.exp(-z0**2) * Lbox_xy)

    dz=(Lbox_z/2.0) / 40.0
    # rho_0 = m_gas_0*N_1D**2 / (Lbox_xy**2 * dz)

    while(z0 < Lbox_z/2.): #here dividing box z length by two to get density porfile for just one side of midplane, then just copy for other half (above/below)
        print(iter,z0,dz)
        
        #This was the original approach where the dz spacing changed to account for density gradient
        # rho=np.exp(-z0**2) #check exponent for disk  
        # dz=((Lbox_xy**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS)); #Original
        # m_gas_cur=m_gas_0 * (Lbox_xy/dz)**2 / N_1D**2; print('m_gas: ' + str(m_gas_cur)); counter+=N_1D**2; #Original
        # x0=np.arange(0.,1.,1./N_1D) + shift;

        #This is the new approach where everything is held constant except for particle mass, which will account for the density gradient alone
        m_gas_cur=m_gas_0 * np.exp(-z0**2); print('m_gas: ' + str(m_gas_cur)); counter+=N_1D**2; #Original
        x0=np.arange(0.,1.,1./N_1D) + shift;

        #print("Initial x0: ", x0)
        while(x0.max() > 1.): x0[(x0 > 1.)]-=1.;
        while(x0.min() < 0.): x0[(x0 < 0.)]+=1.;
        #print("Final x0: ", x0)

        #Save N_1D at each z0 value to plot later:
        N_1D_all.append(N_1D)
        m_gas_all.append(m_gas_cur)
        z0_all.append(z0)
        dz_all.append(dz)

        #x0+=0.5*(0.5-x0[-1]);
        if(DIMS==3):
            x, y = np.meshgrid(x0,x0, sparse=False, indexing='xy'); x=x.flatten(); y=y.flatten();
            print(np.max(x), np.max(y))
        else:
            x=x0; y=0.*x;

        z=z0+dz/2.+np.zeros(x.size); #Original
        z_upper = z + (Lbox_z/2.); #for upper disk
        z_lower = (Lbox_z/2.) - z; #for lower disk
        z_lower[(z_lower < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
        z_up_all.append(z_upper[0]); z_low_all.append(z_lower[0]);

        m_gas = m_gas_cur+np.zeros(x.size);
        #print(x.shape,y.shape,z.shape)
        print(x.shape,y.shape,z_upper.shape,z_lower.shape,m_gas.shape)

        #For upper disk (above midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_upper) #fix z for upper
        m_gas_array = np.append(m_gas_array,m_gas)

        #For lower disk (below midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_lower) #fix z for lower
        m_gas_array = np.append(m_gas_array,m_gas)

        #dust
        if(include_dust):
            m_dust = dustgas_massratio*m_gas_cur + np.zeros(x0.size);

            z_full_d=z0+dz/2.+np.zeros(x0.size);
            z_upper_d = z_full_d + (Lbox_z/2.); #for upper disk
            z_lower_d = (Lbox_z/2.) - z_full_d; #for lower disk
            z_lower_d[(z_lower_d < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
            yd = np.append(yd, x0) #upper disk
            zd = np.append(zd, z_upper_d)
            m_dust_array = np.append(m_dust_array,m_dust)

            yd = np.append(yd, x0) #lower disk
            zd = np.append(zd, z_lower_d)
            m_dust_array = np.append(m_dust_array,m_dust)

        z0 += dz; iter += 1;
        shift += dz/2.;

    Ngrains = 0;
    if(include_dust):
        #dust needs to be xv_d = 1.0 (or use above value)
        #yv_d = 0-1, zv_d = 0-Lbox_z 
        #-> uniform distribution, start by trying separation of 0.1 in z direction and 0.0035 in y direction
        #need to determine this from proper dust mass ratio of a disk

        #this is being removed because it is being done in the while loop instead to have uniform gas-dust ratio
        # y0 = np.arange(0., 1., 1./N_1D_all[0])
        # z0 = np.arange(0., Lbox_z, 0.1)
        # yd, zd = np.meshgrid(y0,z0, sparse=False, indexing='xy'); yd=yd.flatten(); zd=zd.flatten();

        xd_max_coord = 1.0
        xd = xd_max_coord + 0.0*yd
        Ngrains = xd.size

    volume=(Lbox_xy**2)*dz
    density=np.array(m_gas_all)*N_1D**2 / volume
    plt.figure()
    plt.plot(z0_all, density)
    plt.show()
    # exit()
    print("##################Checking lower plane coords:################## ");
    #print(z_lower[0], z[0], dz);
    #print(dz_all);
    print(z_up_all); print(z_low_all);

    if(DIMS<3):
        yv_g=1.*zv_g; zv_g=0.*zv_g;
        
    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    Ngas=xv_g.size;
    # m_target_gas = (1.-np.exp(-Lbox_z))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)
    # m_target_gas_test = (1.-np.exp(-Lbox_z**2))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)

    print("###########HERE#########")
    print(Ngas)
    print("###########DONE#########")

    #Calculate gas density at midplane
    # density_at_all_z = np.zeros(len(N_1D_all))
    # for i in range(len(N_1D_all)):
    #     density_at_all_z[i] = (N_1D_all[i]**2)*m_gas_all[i] / (Lbox_xy*Lbox_xy*dz_all[i])

    # density_overall = integrate.trapezoid(density_at_all_z, dz_all)
    # print("Density overall: ", density_overall)
    
    density_at_miplane = (N_1D_all[0]**2)*m_gas_all[0] / (Lbox_xy*Lbox_xy*dz_all[0])
    print("Density at the midplane: ", density_at_miplane)        

    # #For testing dust-to-gas mass ratio:
    # time = 1200
    # num_Pgas_outer_edge = len(np.where(xv_g*12.0>=(12 - time*12/1200))[0])
    # print("Num gas particles in outer edge: ", num_Pgas_outer_edge)
    # total_gas_mass_outer_edge = num_Pgas_outer_edge*m_target_gas

    # total_dust_mass_outer_edge_thr = int(time/4.208)*15606*m_target_gas*dustgas_massratio

    # print("dust-to-gas ratio: ", total_dust_mass_outer_edge_thr/total_gas_mass_outer_edge)

    pylab.close('all')
    #pylab.axis([0.,1.,0.,1.])
    # pylab.plot(xv_g,yv_g,marker='.',color='black',linestyle='',rasterized=True);

    # exit()
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

    #q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g; print(xv_g); print(yv_g); print(zv_g);
    #n_part_gas = 128**3
    #print(1 + (np.random.random(size=n_part_gas)-.5)*1e-3); exit();
    q=np.zeros((Ngas,3)); 
    q[:,0]=(xv_g * (1 + (np.random.random(size=len(xv_g))-.5)*1e-3))*Lbox_xy; 
    q[:,1]=(yv_g * (1 + (np.random.random(size=len(yv_g))-.5)*1e-3))*Lbox_xy; 
    q[:,2]=zv_g * (1 + (np.random.random(size=len(zv_g))-.5)*1e-3);

    plt.figure()
    plt.plot(q[:,1],q[:,2], marker='.', linestyle='None')
    plt.show()

    print(xv_g); print(q[:,0]);
    print(yv_g); print(q[:,1]);
    print(zv_g); print(q[:,2]);

    print(np.min(q[:,0]), np.max(q[:,0])); print(np.min(q[:,1]),  np.max(q[:,1])); print(np.min(q[:,2]),  np.max(q[:,2]));

    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=m_gas_array)
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    if(include_dust):

        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3));

        x_temp = xd * (1 + (np.random.random(size=len(xd))-.5)*1e-3) * Lbox_xy;
        x_temp[np.where(x_temp>12.0)] = 11.999;

        y_temp = yd * (1 + (np.random.random(size=len(yd))-.5)*1e-3) * Lbox_xy;
        y_temp[np.where(y_temp>12.0)] = 11.999;

        print(len(np.where(x_temp>12.0)[0]))

        #to ensure no particles are at x > 1.0 -- doesn't work well when I run it with this IC
        # ok = np.where(x_temp > 1.0)
        # print("Checking if any dust particles are at x>1 before: ", ok)
        # max_off = np.max(x_temp[ok]-1.0)
        # x_temp -= max_off

        q[:,0]=x_temp
        q[:,1]=y_temp

        q[:,2]=zd * (1 + (np.random.random(size=len(zd))-.5)*1e-3);

        print("Checking if any dust particles are at x>1 after: ", np.where(q[:,0] > 1.0*Lbox_xy))
        print(np.min(q[:,0]), np.max(q[:,0]))
        print(np.min(q[:,1]), np.max(q[:,1]))
        print(np.min(q[:,2]), np.max(q[:,2]))
        print(xd); print(q[:,0]);
        print(yd); print(q[:,1]);
        print(zd); print(q[:,2]);

        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=m_dust_array)

    print(len(np.arange(1,Ngas+1))); print(counter); print(len(q));
    file.close()

    #plot dust:
    if(include_dust):
        print("xd: ", q[:,0][0:500])
        plt.figure()
        plt.plot(q[:,0],q[:,1], marker='.', linestyle='None')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()



def makeIC_disk_stratified_v_3(DIMS=3, N_1D_0=256, N_1D_z=128, m_gas_0=4e-5, dustgas_massratio=0.01,
        fname='stratbox_2d_N100.hdf5', Lbox_x=12., Lbox_y=12., Lbox_z=8., include_dust=False):

    #For testing purposes:
    N_1D_all = []
    m_gas_all = []
    z0_all = []
    dz_all = []
    counter = 0.0;
    z_up_all = []
    z_low_all = []

    xv_g=np.zeros(0);yv_g=np.zeros(0);zv_g=np.zeros(0);
    xd=np.zeros(0);yd=np.zeros(0);zd=np.zeros(0);

    m_gas_array=np.zeros(0); m_dust_array=np.zeros(0);

    z0=1.e-10; iter=0; dz=0.; shift=0.;
    # Nbase = N_1D**3 / (np.exp(-z0**2) * Lbox_xy)

    dz=Lbox_z / N_1D_z
    # rho_0 = m_gas_0*N_1D**2 / (Lbox_xy**2 * dz)

    while(z0 < Lbox_z/2.): #here dividing box z length by two to get density porfile for just one side of midplane, then just copy for other half (above/below)
        print(iter,z0,dz)
        
        #This was the original approach where the dz spacing changed to account for density gradient
        # rho=np.exp(-z0**2) #check exponent for disk  
        # dz=((Lbox_xy**(DIMS-1.)) / (rho*Nbase))**(1./(1.*DIMS)); #Original
        # m_gas_cur=m_gas_0 * (Lbox_xy/dz)**2 / N_1D**2; print('m_gas: ' + str(m_gas_cur)); counter+=N_1D**2; #Original
        # x0=np.arange(0.,1.,1./N_1D) + shift;

        #This is the new approach where everything is held constant except for particle mass, which will account for the density gradient alone
        # m_gas_cur=m_gas_0 * np.exp(-z0**2); print('m_gas: ' + str(m_gas_cur)); counter+=N_1D**2; #Original
        # x0=np.arange(0.,1.,1./N_1D) + shift;

        #And this is the third version approach where dz and m_gas are kept constant and only N_1D varies to reflect the statification
        #TODO: check this!!
        N_1D_cur=int(N_1D_0*np.sqrt(np.exp(-z0**2))); print('N_1D: ' + str(N_1D_cur)); counter+=N_1D_cur**2; #Original
        x0=np.arange(0.,1.,1./N_1D_cur) + shift;

        #print("Initial x0: ", x0)
        while(x0.max() > 1.): x0[(x0 > 1.)]-=1.;
        while(x0.min() < 0.): x0[(x0 < 0.)]+=1.;
        #print("Final x0: ", x0)

        #Save N_1D at each z0 value to plot later:
        N_1D_all.append(N_1D_cur)
        m_gas_all.append(m_gas_0)
        z0_all.append(z0)
        dz_all.append(dz)

        #x0+=0.5*(0.5-x0[-1]);
        if(DIMS==3):
            x, y = np.meshgrid(x0,x0, sparse=False, indexing='xy'); x=x.flatten(); y=y.flatten();
            print(np.max(x), np.max(y))
        else:
            x=x0; y=0.*x;

        z=z0+dz/2.+np.zeros(x.size); #Original
        z_upper = z + (Lbox_z/2.); #for upper disk
        z_lower = (Lbox_z/2.) - z; #for lower disk
        z_lower[(z_lower < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
        z_up_all.append(z_upper[0]); z_low_all.append(z_lower[0]);

        m_gas = m_gas_0+np.zeros(x.size);
        #print(x.shape,y.shape,z.shape)
        print(x.shape,y.shape,z_upper.shape,z_lower.shape,m_gas.shape)

        #For upper disk (above midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_upper) #fix z for upper
        m_gas_array = np.append(m_gas_array,m_gas)

        #For lower disk (below midplane)
        xv_g = np.append(xv_g,x)
        yv_g = np.append(yv_g,y)
        zv_g = np.append(zv_g,z_lower) #fix z for lower
        m_gas_array = np.append(m_gas_array,m_gas)

        #dust
        if(include_dust):
            m_dust = dustgas_massratio*m_gas_0 + np.zeros(x0.size);

            z_full_d=z0+dz/2.+np.zeros(x0.size);
            z_upper_d = z_full_d + (Lbox_z/2.); #for upper disk
            z_lower_d = (Lbox_z/2.) - z_full_d; #for lower disk
            z_lower_d[(z_lower_d < 0.)] = 1.e-10 #maybe do this a faster way because we know this is only a problem for the last iteration
        
            yd = np.append(yd, x0) #upper disk
            zd = np.append(zd, z_upper_d)
            m_dust_array = np.append(m_dust_array,m_dust)

            yd = np.append(yd, x0) #lower disk
            zd = np.append(zd, z_lower_d)
            m_dust_array = np.append(m_dust_array,m_dust)

        z0 += dz; iter += 1;
        shift += dz/2.;

    Ngrains = 0;
    if(include_dust):
        #dust needs to be xv_d = 1.0 (or use above value)
        #yv_d = 0-1, zv_d = 0-Lbox_z 
        #-> uniform distribution, start by trying separation of 0.1 in z direction and 0.0035 in y direction
        #need to determine this from proper dust mass ratio of a disk

        #this is being removed because it is being done in the while loop instead to have uniform gas-dust ratio
        # y0 = np.arange(0., 1., 1./N_1D_all[0])
        # z0 = np.arange(0., Lbox_z, 0.1)
        # yd, zd = np.meshgrid(y0,z0, sparse=False, indexing='xy'); yd=yd.flatten(); zd=zd.flatten();

        xd_max_coord = 1.0
        xd = xd_max_coord + 0.0*yd
        Ngrains = xd.size

    volume=(Lbox_x*Lbox_y)*dz
    density=m_gas_0*np.array(N_1D_all)**2 / volume
    plt.figure()
    plt.plot(z0_all, density)
    plt.show()
    # exit()
    print("##################Checking lower plane coords:################## ");
    #print(z_lower[0], z[0], dz);
    #print(dz_all);
    print(z_up_all); print(z_low_all);

    if(DIMS<3):
        yv_g=1.*zv_g; zv_g=0.*zv_g;
        
    # make a regular 1D grid for particle locations (with N_1D elements and unit length)
    Ngas=xv_g.size;
    # m_target_gas = (1.-np.exp(-Lbox_z))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)
    # m_target_gas_test = (1.-np.exp(-Lbox_z**2))*(Lbox_xy**(DIMS-1))*rho_target / (1.*Ngas)

    print("###########HERE#########")
    print(Ngas)
    print("###########DONE#########")

    #Calculate gas density at midplane
    # density_at_all_z = np.zeros(len(N_1D_all))
    # for i in range(len(N_1D_all)):
    #     density_at_all_z[i] = (N_1D_all[i]**2)*m_gas_all[i] / (Lbox_xy*Lbox_xy*dz_all[i])

    # density_overall = integrate.trapezoid(density_at_all_z, dz_all)
    # print("Density overall: ", density_overall)
    
    density_at_miplane = (N_1D_all[0]**2)*m_gas_all[0] / (Lbox_x*Lbox_y*dz_all[0])
    print("Density at the midplane: ", density_at_miplane)        

    # #For testing dust-to-gas mass ratio:
    # time = 1200
    # num_Pgas_outer_edge = len(np.where(xv_g*12.0>=(12 - time*12/1200))[0])
    # print("Num gas particles in outer edge: ", num_Pgas_outer_edge)
    # total_gas_mass_outer_edge = num_Pgas_outer_edge*m_target_gas

    # total_dust_mass_outer_edge_thr = int(time/4.208)*15606*m_target_gas*dustgas_massratio

    # print("dust-to-gas ratio: ", total_dust_mass_outer_edge_thr/total_gas_mass_outer_edge)

    pylab.close('all')
    #pylab.axis([0.,1.,0.,1.])
    # pylab.plot(xv_g,yv_g,marker='.',color='black',linestyle='',rasterized=True);

    # exit()
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

    #q=np.zeros((Ngas,3)); q[:,0]=xv_g; q[:,1]=yv_g; q[:,2]=zv_g; print(xv_g); print(yv_g); print(zv_g);
    #n_part_gas = 128**3
    #print(1 + (np.random.random(size=n_part_gas)-.5)*1e-3); exit();
    q=np.zeros((Ngas,3)); 
    q[:,0]=(xv_g * (1 + (np.random.random(size=len(xv_g))-.5)*1e-3))*Lbox_x; 
    q[:,1]=(yv_g * (1 + (np.random.random(size=len(yv_g))-.5)*1e-3))*Lbox_y; 
    q[:,2]=zv_g * (1 + (np.random.random(size=len(zv_g))-.5)*1e-3);

    plt.figure()
    plt.plot(q[:,1],q[:,2], marker='.', linestyle='None')
    plt.show()

    print(xv_g); print(q[:,0]);
    print(yv_g); print(q[:,1]);
    print(zv_g); print(q[:,2]);

    print(np.min(q[:,0]), np.max(q[:,0])); print(np.min(q[:,1]),  np.max(q[:,1])); print(np.min(q[:,2]),  np.max(q[:,2]));

    p.create_dataset("Coordinates",data=q)
    p.create_dataset("Velocities",data=np.zeros((Ngas,3)))
    p.create_dataset("ParticleIDs",data=np.arange(1,Ngas+1))
    p.create_dataset("Masses",data=m_gas_array)
    p.create_dataset("InternalEnergy",data=(0.*xv_g+1.))

    if(include_dust):

        p = file.create_group("PartType3")
        q=np.zeros((Ngrains,3));

        x_temp = xd * (1 + (np.random.random(size=len(xd))-.5)*1e-3) * Lbox_x;
        x_temp[np.where(x_temp>12.0)] = 11.999;

        y_temp = yd * (1 + (np.random.random(size=len(yd))-.5)*1e-3) * Lbox_y;
        y_temp[np.where(y_temp>12.0)] = 11.999;

        print(len(np.where(x_temp>12.0)[0]))

        #to ensure no particles are at x > 1.0 -- doesn't work well when I run it with this IC
        # ok = np.where(x_temp > 1.0)
        # print("Checking if any dust particles are at x>1 before: ", ok)
        # max_off = np.max(x_temp[ok]-1.0)
        # x_temp -= max_off

        q[:,0]=x_temp
        q[:,1]=y_temp

        q[:,2]=zd * (1 + (np.random.random(size=len(zd))-.5)*1e-3);

        print("Checking if any dust particles are at x>1 after: ", np.where(q[:,0] > 1.0*Lbox_x))
        print(np.min(q[:,0]), np.max(q[:,0]))
        print(np.min(q[:,1]), np.max(q[:,1]))
        print(np.min(q[:,2]), np.max(q[:,2]))
        print(xd); print(q[:,0]);
        print(yd); print(q[:,1]);
        print(zd); print(q[:,2]);

        p.create_dataset("Coordinates",data=q)
        p.create_dataset("Velocities",data=np.zeros((Ngrains,3)))
        p.create_dataset("ParticleIDs",data=np.arange(Ngas+1,Ngrains+Ngas+1))
        p.create_dataset("Masses",data=m_dust_array)

    print(len(np.arange(1,Ngas+1))); print(counter); print(len(q));
    file.close()

    #plot dust:
    if(include_dust):
        print("xd: ", q[:,0][0:500])
        plt.figure()
        plt.plot(q[:,0],q[:,1], marker='.', linestyle='None')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

