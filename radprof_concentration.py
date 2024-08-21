import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate as interpolate
import h5py
import matplotlib.pyplot as pl
import pdb
import csv
import os
import math

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

def load_P_at_coord(P_File, part='PartType0', zmed_set=-1.e10, ngrid=1024, return_coords=False, plot_zx=False, plot_zy=False):
    P = P_File[part]

    press = P['Density'][:] * P['InternalEnergy'][:]*(1.001-1.)
    density = P['Density'][:]

    xa, ya, za = 0, 1, 2

    xx = P['Coordinates'][:,xa]
    yy = P['Coordinates'][:,ya]
    zz = P['Coordinates'][:,za]

    if plot_zx:
        frozen_coord = np.copy(yy)
    elif plot_zy:
        frozen_coord = np.copy(xx) 
    else:
        frozen_coord = np.copy(zz)

    zmx = np.max(frozen_coord) - np.min(frozen_coord);
    zzmed = np.median(frozen_coord);
    if (zmed_set > -1.e9): zzmed = zmed_set;
    dzz = np.abs(frozen_coord - zzmed);
    #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #Temporarily removed for testing
    if ('SmoothingLength' in P.keys()):
        ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
    else:
        ok = np.where(dzz < 0.05)

    x = 1.0 * (xx / xx.max())
    y = 1.0 * (yy / yy.max())
    z = 1.0 * (zz / zz.max())

    #Needs some cleaning up for plotting other than xy:
    if plot_zx:
        xg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
        pressgrid = interpolate.griddata((x[ok], z[ok]), press[ok], (xg, zg), method='linear')
        if(return_coords):
            return [xg, zg, pressgrid]
        else:
            return pressgrid
    elif plot_zy:
        yg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
        pressgrid = interpolate.griddata((y[ok], z[ok]), press[ok], (yg, zg), method='linear')
        if(return_coords):
            return [yg, zg, pressgrid]
        else:
            return pressgrid
    else:
        xg, yg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
        pressgrid = interpolate.griddata((x[ok], y[ok]), press[ok], (xg, yg), method='linear')
        densitygrid = interpolate.griddata((x[ok], y[ok]), density[ok], (xg, yg), method='linear')
        #maybe make it a weighted mean later according to density
        masked_pressgrid = np.ma.masked_array(pressgrid, np.isnan(pressgrid))
        masked_densitygrid = np.ma.masked_array(densitygrid, np.isnan(densitygrid))
        avg_press = np.ma.average(masked_pressgrid, axis=1, weights=masked_densitygrid)
        press = avg_press.filled(np.nan)

        press = np.nanmean(pressgrid, axis=1)
        if(return_coords):
            x = np.linspace(0, 1, ngrid)
            return [x, press]
        else:
            return press
    
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

def load_v_at_coord(P_File, part='PartType3', xz=0, zmed_set=-1.e10, ngrid=1024, return_coords=False, plot_zx=False, plot_zy=False):
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

    if plot_zx:
        frozen_coord = np.copy(yy)
    elif plot_zy:
        frozen_coord = np.copy(xx) 
    else:
        frozen_coord = np.copy(zz)

    zmx = np.max(frozen_coord) - np.min(frozen_coord);
    zzmed = np.median(frozen_coord);
    if (zmed_set > -1.e9): zzmed = zmed_set;
    dzz = np.abs(frozen_coord - zzmed);
    #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #Temporarily removed for testing
    if ('SmoothingLength' in P.keys()):
        ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
    else:
        ok = np.where(dzz < 0.05)


    # if xz == 0:
    #     zmx = np.max(zz) - np.min(zz);
    #     zzmed = np.median(zz);
    #     if (zmed_set > -1.e9): zzmed = zmed_set;
    #     dzz = np.abs(zz - zzmed);
    #     #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)]
    #     if ('SmoothingLength' in P.keys()):
    #         ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
    #     else:
    #         ok = np.where(dzz < 0.05)

    x = 1.0 * (xx / xx.max())
    y = 1.0 * (yy / yy.max())
    z = 1.0 * (zz / zz.max())


    if xz == 0:
        if plot_zx:
            xg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            vxgrid = interpolate.griddata((x[ok], z[ok]), vx[ok], (xg, zg), method='linear')
            vzgrid = interpolate.griddata((x[ok], z[ok]), vz[ok], (xg, zg), method='linear')
            vygrid = interpolate.griddata((x[ok], z[ok]), vy[ok], (xg, zg), method='linear')
            if(return_coords):
                return [xg, zg, vxgrid, vzgrid, vygrid]
            else:
                return vxgrid, vzgrid, vygrid
        elif plot_zy:
            yg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            vygrid = interpolate.griddata((y[ok], z[ok]), vy[ok], (yg, zg), method='linear')
            vzgrid = interpolate.griddata((y[ok], z[ok]), vz[ok], (yg, zg), method='linear')
            vxgrid = interpolate.griddata((y[ok], z[ok]), vx[ok], (yg, zg), method='linear')
            if(return_coords):
                return [yg, zg, vygrid, vzgrid, vxgrid]
            else:
                return vygrid, vzgrid, vxgrid
        else:
            xg, yg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            vxgrid = interpolate.griddata((x[ok], y[ok]), vx[ok], (xg, yg), method='linear')
            vygrid = interpolate.griddata((x[ok], y[ok]), vy[ok], (xg, yg), method='linear')
            vzgrid = interpolate.griddata((x[ok], y[ok]), vz[ok], (xg, yg), method='linear')
            if(return_coords):
                return [xg, yg, vxgrid, vygrid, vzgrid]
            else:
                return vxgrid, vygrid, vzgrid
       
    # if xz == 0:
    #     xg, yg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
    #     #yg, xg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))
    #     vxgrid = interpolate.griddata((x[ok], y[ok]), vx[ok], (xg, yg), method='linear')#, fill_value=np.median(vx[ok]))
    #     vygrid = interpolate.griddata((x[ok], y[ok]), vy[ok], (xg, yg), method='linear')#, fill_value=np.median(vy[ok]))
    #     if(return_coords):
    #         return [xg, yg, vxgrid, vygrid]
    #     else:
    #         return vxgrid, vygrid

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
#------------------------------------------------------------------
def calc_vorticity_gas_from_vel_and_coords(xx,yy,zz,vx,vy,vz,ngrid=100.):
    # dudx = np.gradient(vx,xx)
    # dudy = np.gradient(vx,yy)
    # dudz = np.gradient(vx,zz)

    # dvdx = np.gradient(vy,xx)
    # dvdy = np.gradient(vy,yy)
    # dvdz = np.gradient(vy,zz)

    # dwdx = np.gradient(vz,xx)
    # dwdy = np.gradient(vz,yy)
    # dwdz = np.gradient(vz,zz)
    # print("dvdx: ", dvdx)
    # print("dudy: ", dudy)

    delta = 1./ngrid
    ngrid_int = int(ngrid)
    vx = vx.reshape((ngrid_int,ngrid_int,ngrid_int))
    vy = vy.reshape((ngrid_int,ngrid_int,ngrid_int))
    vz = vz.reshape((ngrid_int,ngrid_int,ngrid_int))

    du = np.gradient(vx,delta)
    dv = np.gradient(vy,delta)
    dw = np.gradient(vz,delta)

    #vorticity_x = dwdy - dvdz
    vorticity_x = dw[1] - dv[2]
    print("vorticity_x: ", vorticity_x)
    #vorticity_y = dudz - dwdx
    vorticity_y = du[2] - dw[0]
    print("vorticity_y: ", vorticity_y)
    #vorticity_z = dvdx - dudy
    vorticity_z = dv[0] - du[1]
    print("vorticity_z: ", vorticity_z)

    vorticity_mag = np.sqrt(vorticity_x*vorticity_x + vorticity_y*vorticity_y + vorticity_z*vorticity_z)
    print("vorticity_mag: ", vorticity_mag, np.shape(vorticity_mag))
    return vorticity_mag

#------------------------------------------------------------------
def load_vorticity_at_plane(P_File, interp_files_dir="./interp_data_files/", create_interp_files=False, interpolate_density=False,
                            ngrid=100., ngridj=100j, part='PartType0',xz=0, zmed_set=-1.e10, return_coords=False, plot_zx=False, plot_zy=False):
    if(create_interp_files):
        print("\nCreating files with interpolated velocity data...\n")
        interpolate_3D_separate_files(P_File,part,val_to_interp='Velocities',ngridj=ngridj)
        print("Done!\n")
    else:
        print("\nNot creating files with interpolated velocity data, should already be created.\n")

    #Load in interpolated velocity data from files
    xgrid = np.loadtxt('./interp_data_files/xg', np.float64, unpack=False)
    ygrid = np.loadtxt('./interp_data_files/yg', np.float64, unpack=False)
    zgrid = np.loadtxt('./interp_data_files/zg', np.float64, unpack=False)

    vxgrid = np.loadtxt('./interp_data_files/vxgrid', np.float64, unpack=False)
    vygrid = np.loadtxt('./interp_data_files/vygrid', np.float64, unpack=False)
    vzgrid = np.loadtxt('./interp_data_files/vzgrid', np.float64, unpack=False)

    vorticity_mag = calc_vorticity_gas_from_vel_and_coords(xgrid, ygrid, zgrid, vxgrid, vygrid, vzgrid, ngrid=ngrid)

    if(interpolate_density):
        print("\nCreating files with interpolated density data...\n")
        interpolate_3D_separate_files(P_File,part,val_to_interp='Density',ngridj=ngridj)
        print("Done!\n")

    rhogrid = np.loadtxt('./interp_data_files/rhogrid', np.float64, unpack=False)
    ngrid_int = int(ngrid)
    rhogrid = rhogrid.reshape((ngrid_int,ngrid_int,ngrid_int))

    vorticity_plane = np.zeros((ngrid_int,ngrid_int))
    for i in range(ngrid_int):
        for j in range(ngrid_int):
            upper=0.
            lower=0.
            for k in range(ngrid_int):
                if(not math.isnan(vorticity_mag[i][j][k]*rhogrid[i][j][k])):
                    upper += vorticity_mag[i][j][k]*rhogrid[i][j][k]
                    lower += rhogrid[i][j][k]
            if(lower!=0.):
                vorticity_plane[i][j] = upper/lower

    # for i in range(ngrid_int):
    #     for j in range(ngrid_int):
    #         rho_max = np.nanmax(rhogrid[i][j])
    #         print(rho_max)
    #         for k in range(ngrid_int):
    #             if(not math.isnan(vorticity_mag[i][j][k]) and not math.isnan(rhogrid[i][j][k])):
    #                 if(rho_max!=0. and not math.isnan(rho_max)):
    #                     vorticity_plane[i][j] += vorticity_mag[i][j][k]*rhogrid[i][j][k]/rho_max
    #                 else: 
    #                     vorticity_plane[i][j] = 0.

    f = open('./interp_data_files/vorticity_plane', "x")
    f = open('./interp_data_files/vorticity_plane', "r+")
    np.savetxt(f,vorticity_plane.flatten())
    f.close()

    exit()

    #return vorticity_plane
    pl.figure()
    #pl.contourf([xgrid,ygrid], vorticity_plane)
    pl.imshow(vorticity_plane)
    pl.show()




    #TEMPORARY HERE
    pl.figure()
    pl.contourf([xgrid,ygrid], vorticity_mag)
    pl.show()
    return xgrid, ygrid, zgrid, vorticity_mag
    #############still need to fix below##############
    if plot_zx:
        frozen_coord = np.copy(yy)
    elif plot_zy:
        frozen_coord = np.copy(xx) 
    else:
        frozen_coord = np.copy(zz)

    zmx = np.max(frozen_coord) - np.min(frozen_coord);
    zzmed = np.median(frozen_coord);
    if (zmed_set > -1.e9): zzmed = zmed_set;
    dzz = np.abs(frozen_coord - zzmed);
    #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #Temporarily removed for testing
    if ('SmoothingLength' in P.keys()):
        ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
    else:
        ok = np.where(dzz < 0.05)


    # if xz == 0:
    #     zmx = np.max(zz) - np.min(zz);
    #     zzmed = np.median(zz);
    #     if (zmed_set > -1.e9): zzmed = zmed_set;
    #     dzz = np.abs(zz - zzmed);
    #     #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)]
    #     if ('SmoothingLength' in P.keys()):
    #         ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
    #     else:
    #         ok = np.where(dzz < 0.05)

    x = 1.0 * (xx / xx.max())
    y = 1.0 * (yy / yy.max())
    z = 1.0 * (zz / zz.max())


    if xz == 0:
        if plot_zx:
            xg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            vxgrid = interpolate.griddata((x[ok], z[ok]), vx[ok], (xg, zg), method='linear')
            vzgrid = interpolate.griddata((x[ok], z[ok]), vz[ok], (xg, zg), method='linear')
            vygrid = interpolate.griddata((x[ok], z[ok]), vy[ok], (xg, zg), method='linear') # added this for vorticity
            np.gradient(vxgrid)

            vorticity_mag_grid = interpolate.griddata((x[ok], z[ok]), vorticity_mag[ok], (xg, zg), method='linear')
            if(return_coords):
                return [xg, zg, vorticity_mag_grid]
            else:
                return vorticity_mag_grid
        elif plot_zy:
            yg, zg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            #vygrid = interpolate.griddata((y[ok], z[ok]), vy[ok], (yg, zg), method='linear')
            #vzgrid = interpolate.griddata((y[ok], z[ok]), vz[ok], (yg, zg), method='linear')
            vorticity_mag_grid = interpolate.griddata((y[ok], z[ok]), vorticity_mag[ok], (yg, zg), method='linear')

            if(return_coords):
                return [yg, zg, vorticity_mag_grid]
            else:
                return vorticity_mag_grid
        else:
            xg, yg = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))      
            #vxgrid = interpolate.griddata((x[ok], y[ok]), vx[ok], (xg, yg), method='linear')
            #vygrid = interpolate.griddata((x[ok], y[ok]), vy[ok], (xg, yg), method='linear')
            vorticity_mag_grid = interpolate.griddata((x[ok], y[ok]), vorticity_mag[ok], (xg, yg), method='linear')

            if(return_coords):
                return [xg, yg, vorticity_mag_grid]
            else:
                return vorticity_mag_grid
#------------------------------------------------------------------

def interpolate_3D_separate_files(P_File, part='PartType0',val_to_interp='Velocities',ngridj=100j):

    P = P_File[part]
    xa, ya, za = 0, 1, 2

    xx = P['Coordinates'][:,xa]
    yy = P['Coordinates'][:,ya]
    zz = P['Coordinates'][:,za]

    vx = P['Velocities'][:, xa]
    vy = P['Velocities'][:, ya]
    vz = P['Velocities'][:, za]

    rho = P['Density'][:]

    if(val_to_interp=='Velocities'):
        f1 = open('./interp_data_files/xg', "x")
        f1 = open('./interp_data_files/xg', "r+")

        f2 = open('./interp_data_files/yg', "x")
        f2 = open('./interp_data_files/yg', "r+")

        f3 = open('./interp_data_files/zg', "x")
        f3 = open('./interp_data_files/zg', "r+") 

        f4 = open('./interp_data_files/vxgrid', "x")
        f4 = open('./interp_data_files/vxgrid', "r+")

        f5 = open('./interp_data_files/vygrid', "x")
        f5 = open('./interp_data_files/vygrid', "r+")

        f6 = open('./interp_data_files/vzgrid', "x")
        f6 = open('./interp_data_files/vzgrid', "r+")
    elif(val_to_interp=='Density'):
        f4 = open('./interp_data_files/rhogrid', "x")
        f4 = open('./interp_data_files/rhogrid', "r+")
    else:
        print("\nSorry, not yet implemented for this value. Exiting!\n")
        exit()

    xx = 1.0 * (xx / xx.max())
    yy = 1.0 * (yy / yy.max())
    zz = 1.0 * (zz / zz.max())

    xg, yg, zg = np.mgrid[0:1:ngridj, 0:1:ngridj, 0:1:ngridj]

    if (val_to_interp=='Velocities'):
        print("\nInterpolating velocities...\n")
        print("Doing vxgrid\n")
        vxgrid = interpolate.griddata((xx, yy, zz), vx, (xg, yg, zg), method='linear')
        print(np.shape(vxgrid))
        print("Doing vygrid\n")
        vygrid = interpolate.griddata((xx, yy, zz), vy, (xg, yg, zg), method='linear')
        print("Doing vzgrid\n")
        vzgrid = interpolate.griddata((xx, yy, zz), vz, (xg, yg, zg), method='linear')

        #Saving grid coords data
        np.savetxt(f1,xg.flatten())
        np.savetxt(f2,yg.flatten())
        np.savetxt(f3,zg.flatten())
        f1.close()
        f2.close()
        f3.close()  

        np.savetxt(f4,vxgrid.flatten())
        np.savetxt(f5,vygrid.flatten())
        np.savetxt(f6,vzgrid.flatten())
        f4.close()
        f5.close()
        f6.close()

    else:
        print("\nInterpolating density...\n")
        rhogrid = interpolate.griddata((xx, yy, zz), rho, (xg, yg, zg), method='linear')
        print("Saving files\n")
        np.savetxt(f4,rhogrid.flatten())
        f4.close()

#------------------------------------------------------------------------------------------------------------------------------------

def interpolate_3D(xx,yy,zz,vx,vy,vz, interp_filename="interpolated_data.txt"):

    f = open(interp_filename, "x")

    f = open(interp_filename, "r+")

    xg, yg, zg = np.mgrid[0:1:3j, 0:1:3j, 0:1:3j]
    #xg, yg, zg = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j]
    #print(xg.flatten())

    vxgrid = interpolate.griddata((xx, yy, zz), vx, (xg, yg, zg), method='linear')
    vygrid = interpolate.griddata((xx, yy, zz), vy, (xg, yg, zg), method='linear')
    vzgrid = interpolate.griddata((xx, yy, zz), vz, (xg, yg, zg), method='linear')
    #vxgrid = interpolate.griddata((xx, yy), vx, (xg, yg), method='linear')
    #vygrid = interpolate.griddata((xx, yy), vy, (xg, yg), method='linear')

    #print(vxgrid)

    np.savetxt(f,xg.flatten())
    f.write("\n")
    np.savetxt(f,yg.flatten())
    f.write("\n")
    np.savetxt(f,zg.flatten())
    f.write("\n")
    np.savetxt(f,vxgrid.flatten())
    f.write("\n")
    np.savetxt(f,vygrid.flatten())
    f.write("\n")
    np.savetxt(f,vzgrid.flatten())
    f.write("\n")

    f.close()
    #points = np.append(np.append(xg,yg),xg)
    #velocities = np.append(np.append(vxgrid,vygrid),vxgrid)

    # with open('points.txt', 'w') as f_1:
    #     csv.writer(f_1, delimiter=' ').writerows(points.toList())

    # with open('velocities.txt', 'w') as f_2:
    #     csv.writer(f_2, delimiter=' ').writerows(velocities.toList())

    # with open("points.txt", "w") as txt_file:
    #     for line in points:
    #         txt_file.write(" ".join(line) + "\n")

    # with open("velocities.txt", "w") as txt_file_2:
    #     for line in velocities:
    #         txt_file_2.write(" ".join(line) + "\n")

    # f.write(xg)
    # f.write(yg)
    # f.write(zg)
    # f.write(vxgrid)
    # f.write(vygrid)
    # f.write(vzgrid)

    #return xg, yg, zg, vxgrid, vygrid, vzgrid
#------------------------------------------------------------------

def calc_vorticity_gas(P_File, part='PartType0'):

    P = P_File[part]

    xa, ya, za = 0, 1, 2

    #Load the coordinates and velocities of the gas particles
    xx = np.array(P['Coordinates'][:,xa])
    yy = np.array(P['Coordinates'][:,ya])
    zz = np.array(P['Coordinates'][:,za])

    vx = np.array(P['Velocities'][:, xa])
    vy = np.array(P['Velocities'][:, ya])
    vz = np.array(P['Velocities'][:, za])

    return calc_vorticity_gas_from_vel_and_coords(xx,yy,zz,vx,vy,vz)
#------------------------------------------------------------------

