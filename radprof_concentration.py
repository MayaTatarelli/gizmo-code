import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import h5py
import matplotlib.pyplot as pl
import pdb

def get_rad_dens_prof(P_File, ptype='PartType0', val2take='Density', xz=0, ngrid=1024, getmass=False):
    P = P_File[ptype]
    Pc = np.array(P['Coordinates'])

    xset = 0;
    yset = 1;
    zset = 2;

    if xz:
        yset = 2;
        zset = 1;

    xx = Pc[:, xset]
    yy = Pc[:, yset]
    zz = Pc[:, zset]

    x = 1.0 * (xx / xx.max() + 0.)
    y = 1.0 * (yy / yy.max() + 0.)
    z = 1.0 * (zz / zz.max() + 0.)

    if not(getmass):
        if xz == 0:
            yg, xg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))
            val = P[val2take][:]
            dg = interpolate.griddata((x, y), val, (xg, yg), method='linear', fill_value=np.median(val))
        elif xz == 1:
            zg, xg = np.meshgrid(np.linspace(0, 1, int(ngrid/3.)), np.linspace(0, 1, ngrid))
            val = P[val2take][:]
            dg = interpolate.griddata((x, z), val, (xg, zg), method='linear', fill_value=np.median(val))
        return np.sum(dg, 1)
    else:
        return np.histogram(x, bins=np.linspace(0, 1, ngrid))[0]

def plt_dens_prof(i, outputdir, profdir, ngrid=200):
    P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')
    gas_prof = get_rad_dens_prof(P_File, ngrid=ngrid)
    dust_prof = get_rad_dens_prof(P_File, ptype='PartType3', getmass=True, ngrid=ngrid)

    plotpos = np.linspace(-3, 3, ngrid)

    fig, ax = pl.subplots(1,1)
    ax.plot(plotpos[20:-20], gas_prof[20:-20]/np.max(gas_prof), color='royalblue')
    ax.plot(0.5*(plotpos[1:]+plotpos[:-1])[20:-20], dust_prof[20:-20]/np.max(dust_prof), color='darkorange')
    ax.set_yscale('log')
    ax.set_ylim((1e-3, 3))
    ax.set_title(r'%i$\Omega^{-1}$'%i)
    pl.savefig(profdir+'profile_%04i.png'%i)
    P_File.close()

def Gauss(x, A, loc, wd):
    y = A * np.exp(-(x-loc)**2/2/wd**2)
    return y

def fit_gaussian(fname, starti, endi, step, outputdir, profdir, ngrid=200):

    nsteps = int((endi-starti)/step) + 1

    g_amp = np.zeros(nsteps)
    g_loc = np.zeros(nsteps)
    g_wd = np.zeros(nsteps)

    for n, i in enumerate(np.arange(starti, endi+step, step)):
        print(i)
        P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')
        dust_prof = get_rad_dens_prof(P_File, ptype='PartType3', getmass=True, ngrid=ngrid)

        plotpos = np.linspace(-3, 3, ngrid)
        x_plot = 0.5*(plotpos[1:]+plotpos[:-1])

        param, covar = curve_fit(Gauss, x_plot[20:-20], dust_prof[20:-20])

        g_amp[n] = param[0]
        g_loc[n] = param[1]
        g_wd[n] = param[2]

        expected = Gauss(x_plot[20:-20],g_amp[n],g_loc[n],g_wd[n])

        x_plot = 0.5*(plotpos[1:]+plotpos[:-1])
        fig, ax = pl.subplots(1,1)
        ax.plot(x_plot[20:-20], dust_prof[20:-20]/np.max(dust_prof), color='darkorange')
        ax.plot(x_plot[20:-20], expected/expected.max(), linestyle='--')
        ax.set_yscale('log')
        ax.set_ylim((1e-3, 3))
        ax.set_title(r'%i$\Omega^{-1}$'%i)
        pl.savefig(profdir+'/dustprofile_gaussfit_%04i.png'%i)
        pl.close()
        P_File.close()

    np.savetxt(profdir+'/'+fname, np.c_[np.arange(starti, endi+step, step), g_amp, g_loc, g_wd])
        
def load_P(P_File, xz=0, ngrid=1024):
    P = P_File['PartType0']

    press = P['Density'][:] * P['InternalEnergy'][:]*(1.001-1.)

    if xz == 0:
        xa, ya, za = 0, 1, 2
    elif xz == 1:
        xa, ya, za = 0, 2, 1

    xx = P['Coordinates'][:,xa]
    yy = P['Coordinates'][:,ya]
    zz = P['Coordinates'][:,za]

    x = 1.0 * (xx / xx.max())
    y = 1.0 * (yy / yy.max())
    z = 1.0 * (zz / zz.max())

    if xz == 0:
        yg, xg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))
        Pgrid = interpolate.griddata((x, y), press, (xg, yg), method='linear')
        return Pgrid 
    elif xz == 1:
        zg, xg = np.meshgrid(np.linspace(0, 1, int(ngrid/3.)), np.linspace(0, 1, ngrid))
        Pgrid = interpolate.griddata((x, z), press, (xg, zg), method='linear')
        return Pgrid 

def plt_Pprof(i, outputdir, profdir, ngrid=200, Bump_wd=1, Bump_amp=1., Boxsize=6):
    P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')

    P_gas = load_P(P_File, xz=0, ngrid=ngrid)
    # print(P_gas)

    plotpos = np.linspace(-Boxsize/2, Boxsize/2, ngrid)

    Pexp = 1 + Bump_amp * np.exp(-plotpos**2/(2*Bump_wd**2)) 
    fig, ax = pl.subplots(1,1)
    ax.plot(plotpos, np.nanmedian(P_gas, axis=1), label='Simulation') #try summing (remove nan)
    ax.plot(plotpos, Pexp, label='Expected')
    ax.set_title(r'%i$\Omega^{-1}$'%i)
    ax.legend(loc='best')
    pl.savefig(profdir+'Pgas_profile_%04i.png'%i)
    pl.savefig(profdir+'Pgas_profile_%04i.pdf'%i)
    pl.close()
    P_File.close()

def plt_Pprof_maya(i, outputdir, profdir, ngrid=200, Bump_wd=1, Bump_amp=1., Boxsize=6):
    P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')

    P_gas = load_P(P_File, xz=0, ngrid=ngrid) #/1.0E1
    # print(P_gas)

    plotpos = np.linspace(-Boxsize/2, Boxsize/2, ngrid)

    Pexp = 1 + Bump_amp * np.exp(-plotpos**2/(2*Bump_wd**2)) 
    fig, ax = pl.subplots(1,1)
    #pdb.set_trace()
    ax.semilogy(plotpos, P_gas[:,100], label='Simulation') #np.nansum(P_gas, axis=1)
    ax.semilogy(plotpos, Pexp, label='Expected')
    # ax.set_ylim((1.0e-2,1.0e2))
    ax.set_ylabel("$P$") # confirm this, is it P/P_0 or just P
    ax.set_xlabel("$x/H$") # confirm this
    ax.set_title(r'%i$\Omega^{-1}$'%i)
    ax.legend(loc='best')
    #pl.savefig(profdir+'Pgas_profile_%04i.png'%i)
    pl.savefig(profdir+'Pgas_profile_mass_0_5_%04i.pdf'%i)
    pl.close()
    P_File.close()

def plt_Pprof_dust(i, outputdir, profdir, ngrid=200):
    P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')

    P_gas = load_P(P_File, xz=0, ngrid=ngrid)
    dust_prof = get_rad_dens_prof(P_File, ptype='PartType3', getmass=True, ngrid=ngrid)

    plotpos = np.linspace(-3, 3, ngrid)

    fig, ax = pl.subplots(1,1)
    ax.plot(plotpos, np.nanmedian(P_gas, axis=1), label='Gas pressure')
    ax.plot(0.5*(plotpos[1:]+plotpos[:-1]), dust_prof/np.max(dust_prof), label='Dust')
    ax.set_yscale('log')
    ax.set_ylim((1e-3, 3))
    ax.set_title(r'%i$\Omega^{-1}$'%i)
    ax.legend(loc='best', frameon=False)
    pl.savefig(profdir+'GasP_Dustpos_%04i.png'%i)
    pl.close()
    P_File.close()
 
def load_v(P_File, part='PartType3', xz=0, ngrid=1024, return_coords=False):
    P = P_File[part]

    if xz == 0:
        xa, ya, za = 0, 1, 2
    elif xz == 1:
        xa, ya, za = 0, 2, 1

    xx = P['Coordinates'][:,xa]
    yy = P['Coordinates'][:,ya]
    zz = P['Coordinates'][:,za]

    vx = P['Velocities'][:, xa]
    vy = P['Velocities'][:, ya]
    vz = P['Velocities'][:, za]

    x = 1.0 * (xx / xx.max())
    y = 1.0 * (yy / yy.max())
    z = 1.0 * (zz / zz.max())

    if xz == 0:
        xg, yg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
        #yg, xg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))
        vxgrid = interpolate.griddata((x, y), vx, (xg, yg), method='linear')
        vygrid = interpolate.griddata((x, y), vy, (xg, yg), method='linear')
        if(return_coords):
            return [xg, yg, vxgrid, vygrid]
        else:
            return vxgrid, vygrid
    elif xz == 1:
        zg, xg = np.meshgrid(np.linspace(0, 1, int(ngrid/3.)), np.linspace(0, 1, ngrid))
        vxgrid = interpolate.griddata((x, z), vx, (xg, zg), method='linear')
        vygrid = interpolate.griddata((x, z), vy, (xg, zg), method='linear')
        if(return_coords):
            return [xg, zg, vxgrid, vygrid]
        else:
            return vxgrid, vygrid

def plt_vprof(i, outputdir, profdir, ngrid=200, Pi=0.05, Bump_wd=1, Bump_amp=1., Boxsize=6):
    P_File = h5py.File(outputdir+'snapshot_%03i.hdf5'%i,'r')

    vx_gas, vy_gas = load_v(P_File, part='PartType0', xz=0, ngrid=ngrid)
    #vx_dust, vy_dust = load_v(P_File, part='ParType3', xz=0, ngrid=ngrid)

    plotpos = np.linspace(-Boxsize/2, Boxsize/2, ngrid)

    vphi_pert = -Pi-Bump_amp*(plotpos / Bump_wd**2)*np.exp(-plotpos**2/2/Bump_wd**2)/2/(1+Bump_amp*np.exp(-plotpos**2/2/Bump_wd**2))
    fig, ax = pl.subplots(1,1)
    ax.plot(plotpos, np.nanmedian(vy_gas, axis=1)+1.5*plotpos, label='Simulation')
    ax.plot(plotpos, vphi_pert, label='Perturbed')
    ax.set_title(r'%i$\Omega^{-1}$'%i)
    ax.legend(loc='best')
    pl.savefig(profdir+'vgasphi_profile_%04i.png'%i)
    P_File.close()
