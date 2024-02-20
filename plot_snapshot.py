"""
Plot a snapshot of shearing box

@author: Eve J. Lee
May 12th 2019
"""
import numpy as np
import h5py as h5py
import os.path
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pylab as pylab
import matplotlib.pyplot as plot
import scipy.interpolate as interpolate
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb
from radprof_concentration import *

def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    for extension_touse in [extension,'.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname

        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];

        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext;
            fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext;
            fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/snapdir_'+ext+'/'+snapshot_name+'_'+ext;
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snapdir_'+ext+'/'+'snap_'+ext;
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist!
    return fname_found, fname_base_found, fname_ext;

def load_snap(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    file = h5py.File(fname,'r')
    return file

def gas_rho_image(ax, snum=3, sdir='./output', vmin=0., vmax=0., boxL_z=2, ptype='PartType0',
                  cmap='terrain', xmax=1., xz=0, yz=0, gas_val_toplot='rho', rhocut=-2.5, save='dummy',
                  zmed_set=-1.e10, quiet=False, zdir='z', zs=0., plot_zx=False, plot_zy=False):
    P_File = load_snap(sdir, snum);
    P = P_File[ptype]
    Pc = np.array(P['Coordinates'])
    #dmax = np.max(Pc, axis=0)
    xset = 0;
    yset = 1;
    zset = 2;
    if (xz == 1):
        xset = 0;
        yset = 2;
        zset = 1;
    if (yz == 1):
        xset = 1;
        yset = 2;
        zset = 0;
    xx = Pc[:, xset];
    yy = Pc[:, yset];
    zz = Pc[:, zset]
    dmax = np.max(xx), np.max(yy), np.max(zz)
    print('gas_xminmax=', np.min(xx), np.max(xx))
    vx = P['Velocities'][:, 0];
    vy = P['Velocities'][:, 1];
    vz = P['Velocities'][:, 2]
    if (quiet == False):
        print('Var v_x_gas == ', np.min(vx), np.median(vx), np.max(vx), np.std(vx), ' (min/median/max/std)')
        print('Var v_y_gas == ', np.min(vy), np.median(vy), np.max(vy), np.std(vy), ' (min/median/max/std)')
        print('Var v_z_gas == ', np.min(vz), np.median(vz), np.max(vz), np.std(vz), ' (min/median/max/std)')
    if ('MagneticField' in P.keys()):
        bx = P['MagneticField'][:, 0];
        by = P['MagneticField'][:, 1];
        bz = P['MagneticField'][:, 2]
        if (quiet == False):
            print('Var B_x_gas == ', np.min(bx), np.median(bx), np.max(bx), np.std(bx), ' (min/median/max/std)')
            print('Var B_y_gas == ', np.min(by), np.median(by), np.max(by), np.std(by), ' (min/median/max/std)')
            print('Var B_z_gas == ', np.min(bz), np.median(bz), np.max(bz), np.std(bz), ' (min/median/max/std)')
            print('B_z_vol == ',
                  np.sum(bz * P['Masses'][:] / P['Density'][:]) / np.sum(P['Masses'][:] / P['Density'][:]))
    else:
        bx = vx;
        by = vy;
        bz = vz;

    if xz == 0 and yz == 0:
        # zmx = np.max(zz) - np.min(zz);
        # zzmed = np.median(zz);
        # print("***********************"); print("Max/Min z: ", np.max(zz), np.min(zz));
        # print("zmx/z-median: ", zmx, zzmed);
        # if (zmed_set > -1.e9): zzmed = zmed_set;
        # dzz = np.abs(zz - zzmed);
        # #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #Temporarily removed for testing
        # if ('SmoothingLength' in P.keys()):
        #     ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
        # else:
        #     ok = np.where(dzz < 0.05)
        if plot_zx:
            frozen_coord = np.copy(yy)
        elif plot_zy:
            frozen_coord = np.copy(xx) 
        else:
            frozen_coord = np.copy(zz)

        zmx = np.max(frozen_coord) - np.min(frozen_coord);
        zzmed = np.median(frozen_coord);
        print("***********************"); print("Max/Min z: ", np.max(frozen_coord), np.min(frozen_coord));
        print("zmx/z-median: ", zmx, zzmed);
        if (zmed_set > -1.e9): zzmed = zmed_set;
        dzz = np.abs(frozen_coord - zzmed);
        #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #Temporarily removed for testing
        if ('SmoothingLength' in P.keys()):
            ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
        else:
            ok = np.where(dzz < 0.05)

    print("Accepted values for frozen axis:", frozen_coord[ok])
    x = 1.0 * (xx / dmax[0] + 0.);
    y = 1.0 * (yy / dmax[1] + 0.);
    z = 1.0 * (zz / dmax[2] + 0.);

    if ((gas_val_toplot == 'bvec') | (gas_val_toplot == 'vvec')):
        yg, xg = np.mgrid[0:1:128j, 0:1:128j]
    elif xz == 0:
        if plot_zx:
            zg, xg = np.mgrid[0:1:2048j, 0:1:2048j] #double check these values for mgrid
        elif plot_zy:
            zg, yg = np.mgrid[0:1:2048j, 0:1:2048j] #double check these values for mgrid
        else:
            yg, xg = np.mgrid[0:1:2048j, 0:1:2048j]
    else:
        yg, xg = np.mgrid[0:1:680j, 0:1:2048j]

    v0 = P['Velocities']
    if (ptype == 'PartType0'):
        u = np.log10(P['Density'][:])
        u0 = P['InternalEnergy'][:]
        if (quiet == False):
            print('particle number = ', u.size)
            print('min/max/std internal energy = ', np.min(u0), np.max(u0), np.std(u0))
            print('min/max/std velocity = ', np.min(v0), np.max(v0), np.sqrt(np.std(v0[:, 0]) ** 2 + np.std(v0[:, 1]) ** 2 + np.std(v0[:, 2]) ** 2))
            print('min/max/std density = ', np.min(u), np.max(u), np.std(u))
            print("HERE")

    if (gas_val_toplot == 'p'):
        u = np.log(P['Density'][:] * P['InternalEnergy'][:])
    if ((gas_val_toplot == 'b') | (gas_val_toplot == 'bpt') | (gas_val_toplot == 'bx') | (gas_val_toplot == 'by') | (
            gas_val_toplot == 'bz')):
        wt = P['Masses'][:] / P['Density'][:];
        wt /= np.sum(wt)
        vx = P['MagneticField'][:, 0];
        vy = P['MagneticField'][:, 1];
        vz = P['MagneticField'][:, 2];
        vx -= np.sum(vx * wt);
        vy -= np.sum(vy * wt);
        vz -= np.sum(vz * wt);
        u = np.sqrt(vx * vx + vy * vy + vz * vz)
        if (gas_val_toplot == 'bx'): u = vx
        if (gas_val_toplot == 'by'): u = vy
        if (gas_val_toplot == 'bz'): u = vz
    if ((gas_val_toplot == 'v') | (gas_val_toplot == 'vpt') | (gas_val_toplot == 'vx') | (gas_val_toplot == 'vy') | (
            gas_val_toplot == 'vz')):
        wt = P['Masses'][:] / np.sum(P['Masses'][:])
        vx = P['Velocities'][:, 0];
        vy = P['Velocities'][:, 1];
        vz = P['Velocities'][:, 2];
        vx -= np.sum(vx * wt);
        vy -= np.sum(vy * wt);
        vz -= np.sum(vz * wt);
        u = np.sqrt(vx * vx + vy * vy + vz * vz)
        if (gas_val_toplot == 'vx'): u = vx
        if (gas_val_toplot == 'vy'): u = vy
        if (gas_val_toplot == 'vz'): u = vz
    if (gas_val_toplot == 'bvec'):
        wt = P['Masses'][:] / np.sum(P['Masses'][:])
        bx = P['MagneticField'][:, xset];
        by = P['MagneticField'][:, yset];
        bz = P['MagneticField'][:, zset]
        bx -= np.sum(bx * wt);
        by -= np.sum(by * wt);
        bz -= np.sum(bz * wt);
        u = bz;
    if (gas_val_toplot == 'vvec'):
        wt = P['Masses'][:] / np.sum(P['Masses'][:])
        bx = P['Velocities'][:, xset];
        by = P['Velocities'][:, yset];
        bz = P['Velocities'][:, zset]
        bx -= np.sum(bx * wt);
        by -= np.sum(by * wt);
        bz -= np.sum(bz * wt);
        u = bz;
    print('min/max_u_in_rhoplot = ', np.min(u), np.max(u))

    #pylab.axis([0., xmax, 0., 1.])
    if (vmax == 0): vmax = np.max(u)
    if (vmin == 0): vmin = np.min(u)
    if (gas_val_toplot == 'pt'):
        ax.plot(x[ok], y[ok], marker=',', linestyle='', color='black', zorder=3)
        P_File.close();
        return;

    if xz == 1:
        print("here 1#######################################################")
        dg = interpolate.griddata((x, z), u, (xg, yg), method='linear', fill_value=np.median(u));
        im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[0], 0, dmax[2]),
                       zorder=1);
    elif yz == 1:
        print("here 2##################################################################")
        dg = interpolate.griddata((y, z), u, (xg, yg), method='linear', fill_value=np.median(u));
        im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[1], 0, dmax[2]),
                       zorder=1);
    else:
        print("here 3#######################################################")
        #print("Min/Max/Med Density: ", np.min(u[ok]), np.max(u[ok]), np.median(u[ok]))
        if (plot_zx):
            dg = interpolate.griddata((x[ok], z[ok]), u[ok], (xg, zg), method='linear', fill_value=np.median(u[ok]));
            im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -boxL_z/2., boxL_z/2.),
                       zorder=1);
        elif (plot_zy):
            dg = interpolate.griddata((y[ok], z[ok]), u[ok], (yg, zg), method='linear', fill_value=np.median(u[ok]));
            im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -boxL_z/2., boxL_z/2.),
                       zorder=1);
        else:
            dg = interpolate.griddata((x[ok], y[ok]), u[ok], (xg, yg), method='linear', fill_value=np.median(u[ok]));

            #For boxL=6
            im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -3, 3),
                       zorder=1);
            #For boxL=10
            # im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-5, 5, -5, 5),
            #            zorder=1);

        # im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[0], 0, dmax[1]),
        #                zorder=1);
        
        # im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, #extent=(-3, 3, -3, 3),
        #                zorder=1);

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = pylab.colorbar(im, cax=cax)
    cbar.set_label(label="log $\\Sigma_g$", size=13, rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    #cbar.set_label(r'$\log\rho$', rotation=270)

    if ((gas_val_toplot == 'vvec') | (gas_val_toplot == 'bvec')):
        bxg = interpolate.griddata((x, y), bx, (xg, yg), method='linear');
        byg = interpolate.griddata((x, y), by, (xg, yg), method='linear');
        bg = np.sqrt(bxg * bxg + byg * byg);
        # pylab.streamplot(xg,yg,bxg,byg,color='black',linewidth=1.,arrowstyle='-',arrowsize=1,density=3.0,zorder=2)
        Q = ax.streamplot(xg, yg, bxg, byg, color=bg, cmap='cool', density=[2., 2.], linewidth=2, arrowstyle='->',
                             arrowsize=1.5, zorder=2)
        P_File.close();
        return;

def plotpts_w_gas(snum=0, sdir='./output', ptype='PartType3', boxL_z=2, width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=False, gas_val_toplot='rho', ptype_im='PartType0',
                  zmed_set=-1.e10, cmap='terrain', imdir='./images/', xz=0, yz=0):
    pylab.close('all');
    #plot.figure(1, figsize=(21., 7.))
    fig, ax = plot.subplots()
    P_File = load_snap(sdir, snum);
    print('Time == ', P_File['Header'].attrs['Time'])
    P = P_File[ptype]
    print('Dust-to-Gas Mass Ratio = ',
          np.sum(P_File['PartType3']['Masses'][:]) / np.sum(P_File['PartType0']['Masses'][:]))
    a = P['GrainSize'][:]
    print('Grain Size: min=', a.min(), ' max=', a.max())
    Pc = np.array(P['Coordinates']);

    vx = np.array(P['Velocities'][:, 0])
    vy = np.array(P['Velocities'][:, 1])
    vz = np.array(P['Velocities'][:, 2])
    print('Var v_x_dust == ', np.min(vx), np.median(vx), np.max(vx), np.std(vx), ' (min/median/max/std)')
    print('Var v_y_dust == ', np.min(vy), np.median(vy), np.max(vy), np.std(vy), ' (min/median/max/std)')
    print('Var v_z_dust == ', np.min(vz), np.median(vz), np.max(vz), np.std(vz), ' (min/median/max/std)')

    for subplot, xz, yz in zip([1, 2, 3], [xz], [yz]):  # ,1,0],[0,0,1]):
        xplotc = 0;
        yplotc = 1;
        depth_c = 2;  # plot xy, z=depth
        if (xz == 1):
            xplotc = 0;
            yplotc = 2;
            depth_c = 1;  # plot xz, y=depth
        if (yz == 1):
            xplotc = 1;
            yplotc = 2;
            depth_c = 0;  # plot yz, x=depth

        quiet = True
        if (subplot == 1): quiet = False  # give numbers once
        if xz == 0 and yz == 0:
            z = Pc[:, depth_c];
            zmx = np.max(z) - np.min(z);
            z0 = np.median(z);
            if (zmed_set > -1.e9): z0 = zmed_set
            zz = np.abs(z - z0);
            zz[(zz > 0.5 * zmx)] = zmx - zz[(zz > 0.5 * zmx)]
            ok = np.where(zz < width)
            #ok = np.where(np.abs(zz) > -1.)
            xx = Pc[:, xplotc][ok];
            yy = Pc[:, yplotc][ok];
            dmax = np.max(xx), np.max(yy)
        elif xz == 1:
            xx = Pc[:, 0]
            yy = Pc[:, 1]
            z0 = 0

        #x = 1.0 * (xx / dmax[0] + 0.);
        #y = 1.0 * (yy / dmax[1] + 0.);

        x = np.copy(xx)
        y = np.copy(yy)
        #pdb.set_trace()
        # np.savetxt("x.txt",x)
        # np.savetxt("y.txt",y)
        # XY = np.zeros((x.size()[0],4))
        # XY[:,0] = x[:]
        # XY[:,1] = y[:]
        # XY[:,2] = vx[:]
        # XY[:,3] = vx[:]
        #exit()

        #pylab.subplot(1, 3, subplot)
        ok_r = (np.random.rand(x.size) < cut_dust)
        gas_rho_image(ax, snum=snum, sdir=sdir, boxL_z=boxL_z, xmax=1., xz=xz, yz=yz, gas_val_toplot=gas_val_toplot, zmed_set=z0,
                      vmin=vmin, vmax=vmax, quiet=quiet, cmap=cmap, ptype=ptype_im)

        if (forsavedfigure == True):
            ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
                       linestyle='', color='black', markeredgecolor='None', zorder=3)
            #Plot velocity streamlines
            xg, yg, vxgrid, vygrid = load_v(P_File, part='PartType0', xz=0, ngrid=1024,return_coords=True)
            # ax.streamplot(xg*6, yg*6, vxgrid, vygrid,linewidth=1.0)

            #For boxL = 6
            # print("Doing box size 6")
            # ax.streamplot(xg*6-3.0, yg*6-3.0, vxgrid, vygrid,linewidth=1.0)
            # ax.set_xlim([-3,3])
            # ax.set_ylim([-3,3])

            #For boxL = 10
            print("Doing box size 10")
            ax.streamplot(xg*10-5.0, yg*10-5.0, vxgrid, vygrid,linewidth=1.0)
            ax.set_xlim([-5,5])
            ax.set_ylim([-5,5])

            #X,Y = np.meshgrid(x, y)
            #plot.streamplot(X,Y,vx,vy)
        else:
            ax.scatter(x[ok_r], y[ok_r], markersize, marker='.', alpha=alpha, color='black', zorder=3)
            #ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
            #           linestyle='', color='black', markeredgecolor='None', zorder=3)

        frame1 = plot.gca()  # get coordinate axes, to use for tick manipulation below
        #frame1.axes.xaxis.set_ticklabels([])  # no tick names
        #frame1.axes.yaxis.set_ticklabels([])  # no tick names
        #frame1.axes.get_xaxis().set_ticks([])  # no ticks
        #frame1.axes.get_yaxis().set_ticks([])  # no ticks
    
    ax.set_title('Time = %i'%P_File['Header'].attrs['Time'])

    if (forsavedfigure == True):
        ext = '000' + str(snum);
        if (snum >= 10): ext = '00' + str(snum)
        if (snum >= 100): ext = '0' + str(snum)
        if (snum >= 1000): ext = str(snum)
        #pylab.savefig(imdir + 'im_' + ext + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        pylab.savefig(imdir + 'im_mass_0_5_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
    P_File.close()
    pylab.close()
    #plot.show()


def plotpts_w_gas_no_dust(snum=0, sdir='./output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=False, gas_val_toplot='rho', ptype_im='PartType0',
                  zmed_set=-1.e10, boxL_z=2, cmap='terrain', imdir='./images/', xz=0, yz=0, 
                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None):
    pylab.close('all');
    #plot.figure(1, figsize=(21., 7.))
    fig, ax = plot.subplots()
    P_File = load_snap(sdir, snum);
    # print('Time == ', P_File['Header'].attrs['Time'])
    P = P_File[ptype]
    # print('Dust-to-Gas Mass Ratio = ',
    #       np.sum(P_File['PartType3']['Masses'][:]) / np.sum(P_File['PartType0']['Masses'][:]))
    # a = P['GrainSize'][:]
    # print('Grain Size: min=', a.min(), ' max=', a.max())
    Pc = np.array(P['Coordinates']);

    # vx = np.array(P['Velocities'][:, 0])
    # vy = np.array(P['Velocities'][:, 1])
    # vz = np.array(P['Velocities'][:, 2])
    # print('Var v_x_dust == ', np.min(vx), np.median(vx), np.max(vx), np.std(vx), ' (min/median/max/std)')
    # print('Var v_y_dust == ', np.min(vy), np.median(vy), np.max(vy), np.std(vy), ' (min/median/max/std)')
    # print('Var v_z_dust == ', np.min(vz), np.median(vz), np.max(vz), np.std(vz), ' (min/median/max/std)')

    for subplot, xz, yz in zip([1, 2, 3], [xz], [yz]):  # ,1,0],[0,0,1]):
        xplotc = 0;
        yplotc = 1;
        depth_c = 2;  # plot xy, z=depth
        if (xz == 1):
            xplotc = 0;
            yplotc = 2;
            depth_c = 1;  # plot xz, y=depth
        if (yz == 1):
            xplotc = 1;
            yplotc = 2;
            depth_c = 0;  # plot yz, x=depth

        quiet = True
        if (subplot == 1): quiet = False  # give numbers once
        if xz == 0 and yz == 0:

            xx = Pc[:, xplotc];
            yy = Pc[:, yplotc];
            zz = Pc[:, depth_c];

            #First if to freeze correct coordinate axis
            if plot_zx:
                frozen_coord = np.copy(yy)
            elif plot_zy:
                frozen_coord = np.copy(xx)
            else:
                frozen_coord = np.copy(zz)

            zmx = np.max(frozen_coord) - np.min(frozen_coord);
            coord0 = np.median(frozen_coord);
            if (zmed_set > -1.e9): coord0 = zmed_set
            dzz = np.abs(frozen_coord - coord0);
            #dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)] #DO I REMOVE?
            ok = np.where(dzz < width)

            xx = xx[ok];
            yy = yy[ok];
            zz = zz[ok];

           

            #######GOOD################################
            # z = Pc[:, depth_c];
            # zmx = np.max(z) - np.min(z);
            # z0 = np.median(z);
            # if (zmed_set > -1.e9): z0 = zmed_set
            # zz = np.abs(z - z0);
            # zz[(zz > 0.5 * zmx)] = zmx - zz[(zz > 0.5 * zmx)]
            # ok = np.where(zz < width)
            # #ok = np.where(np.abs(zz) > -1.)

            # xx = Pc[:, xplotc][ok];
            # yy = Pc[:, yplotc][ok];
            # dmax = np.max(xx), np.max(yy)
            ###########################################
        elif xz == 1:
            xx = Pc[:, 0]
            yy = Pc[:, 1]
            z0 = 0

        #x = 1.0 * (xx / dmax[0] + 0.);
        #y = 1.0 * (yy / dmax[1] + 0.);

        x = np.copy(xx)
        y = np.copy(yy)
        z = np.copy(zz)

        #pdb.set_trace()
        # np.savetxt("x.txt",x)
        # np.savetxt("y.txt",y)
        # XY = np.zeros((x.size()[0],4))
        # XY[:,0] = x[:]
        # XY[:,1] = y[:]
        # XY[:,2] = vx[:]
        # XY[:,3] = vx[:]
        #exit()

        #pylab.subplot(1, 3, subplot)
        ok_r = (np.random.rand(x.size) < cut_dust)
        print('oK_R = ', ok_r)
        print(x[ok_r])
        print('HERE coord0 = ', coord0)
        gas_rho_image(ax, snum=snum, sdir=sdir, boxL_z=boxL_z, xmax=1., xz=xz, yz=yz, gas_val_toplot=gas_val_toplot, zmed_set=coord0,
                      vmin=vmin, vmax=vmax, quiet=quiet, cmap=cmap, ptype=ptype_im, plot_zx=plot_zx, plot_zy=plot_zy)

        if (forsavedfigure == True):
            # this causes "black square"
            # ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
            #            linestyle='', color='black', markeredgecolor='None', zorder=3)
            # proof:
            # ax.plot(0.0, 0.0, marker='^', markersize=markersize, alpha=alpha,
            #            linestyle='', color='black', markeredgecolor='None', zorder=3)
            #Plot velocity streamlines
            #xg, yg, vxgrid, vygrid = load_v(P_File, part='PartType0', xz=0, ngrid=1024,return_coords=True)

            if plot_zx:
                xg, zg, vxgrid, vzgrid = load_v_at_coord(P_File, part='PartType0', xz=0, zmed_set=coord0, ngrid=1024,return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)
                ax.streamplot(xg*6-3.0, zg*boxL_z-(boxL_z/2), vxgrid, vzgrid,linewidth=1.0, color=str_color)

            elif plot_zy:
                yg, zg, vygrid, vzgrid = load_v_at_coord(P_File, part='PartType0', xz=0, zmed_set=coord0, ngrid=1024,return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)
                ax.streamplot(yg*6-3.0, zg*boxL_z-(boxL_z/2), vygrid, vzgrid,linewidth=1.0, color=str_color)
            else:
                xg, yg, vxgrid, vygrid = load_v_at_coord(P_File, part='PartType0', xz=0, zmed_set=coord0, ngrid=1024,return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)
                ax.streamplot(xg*6-3.0, yg*6-3.0, vxgrid, vygrid,linewidth=1.0, color=str_color)

            # ax.streamplot(xg*6, yg*6, vxgrid, vygrid,linewidth=1.0)

            ax.set_xlim([-3,3])
            if (plot_zx | plot_zy):
                ax.set_ylim([-boxL_z/2,boxL_z/2])
            else:
                ax.set_ylim([-3,3])

            #X,Y = np.meshgrid(x, y)
            #plot.streamplot(X,Y,vx,vy)
        else:
            ax.scatter(x[ok_r], y[ok_r], markersize, marker='.', alpha=alpha, color='black', zorder=3)
            #ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
            #           linestyle='', color='black', markeredgecolor='None', zorder=3)

        frame1 = plot.gca()  # get coordinate axes, to use for tick manipulation below
        #frame1.axes.xaxis.set_ticklabels([])  # no tick names
        #frame1.axes.yaxis.set_ticklabels([])  # no tick names
        #frame1.axes.get_xaxis().set_ticks([])  # no ticks
        #frame1.axes.get_yaxis().set_ticks([])  # no ticks

    ax.set_xlabel(xlabel, fontsize=13)
    #ax.set_xticklabels(fontsize=10)
    ax.set_ylabel(ylabel, fontsize=13)
    #ax.set_yticklabels(fontsize=10)

    if plot_zx:
        ax.set_title('Density Profile of Plane $y/H$ = %1.2f at Time = %i'%(coord0-3.0, snum), fontsize=12)
    elif plot_zy:
        ax.set_title('Density Profile of Plane $x/H$ = %1.2f at Time = %i'%(coord0-3.0, snum), fontsize=12)
    else:
        ax.set_title('Density Profile of Plane $z/H$ = %1.2f at Time = %i'%(coord0-(boxL_z/2.0), snum), fontsize=12)

    #ax.set_title('Time = %i'%P_File['Header'].attrs['Time'])

    if (forsavedfigure == True):
        ext = '000' + str(snum);
        print(snum)
        if (snum >= 10): ext = '00' + str(snum)
        if (snum >= 100): ext = '0' + str(snum)
        if (snum >= 1000): ext = str(snum)
        print(ext)
        #pylab.savefig(imdir + 'im_' + ext + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        #pylab.savefig(imdir + 'im_mass_0_5_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
        pylab.savefig(imdir + 'im_density_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0.075)

    P_File.close()
    pylab.close()
    # plot.show()
#------------------------------------------------------------------
def dummy_func(x,y):
    return x+y
def plot_vorticity_mag(snum=0, sdir='./output', partType='PartType0', zmed_set=-1.e10, cmap='terrain', imdir='./images/',
                        boxL_x=6, boxL_y=6, boxL_z=2, plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$'):

    pylab.close('all');

    P_File = load_snap(snum=snum, sdir=sdir)

    xgrid, ygrid, zgrid, vorticity_mag = load_vorticity_at_plane(P_File=P_File, part=partType, zmed_set=zmed_set, return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)
    plot.figure()
    plot.contourf([xgrid,ygrid], vorticity_mag)
    plot.show()

    exit()

    #xg, yg, vorticity_mag_grid = load_vorticity_at_plane(P_File=P_File, part=partType, zmed_set=zmed_set, return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)
    # print(np.shape(xg))
    # print(np.shape(yg))
    #xg, yg, vxgrid, vygrid = load_v_at_coord(P_File, part=partType, xz=0, zmed_set=zmed_set, ngrid=1024, return_coords=True, plot_zx=plot_zx, plot_zy=plot_zy)

    #plot.figure(1, figsize=(21., 7.))
    fig, ax = plot.subplots()
    cmap='RdYlBu'
    # vmin=0
    # vmax=75
    # im = ax.imshow(vorticity_mag_grid, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -3, 3),zorder=1);
    vmin=-10
    vmax=10
    # im = ax.imshow(vygrid, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -3, 3),zorder=1);
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = pylab.colorbar(im, cax=cax)
    # cbar.set_label(label="Vorticity Magnitude, $|\\omega|$", size=13, rotation=90, labelpad=10)
    # cbar.ax.tick_params(labelsize=10)
    # ext = '99' # CHANGE THIS
    # filename_=imdir + 'im_vorticity_mag' + ext + '.pdf'
    #print("Saving plot as %s" % filename_)
    #pylab.savefig(filename_, dpi=150, bbox_inches='tight', pad_inches=0.075)
    pylab.close()
    #plot.contour([xg,yg], velocity_x_grid)
    #plot.contourf([xg,yg], vorticity_mag_grid)
    # plot.contourf([xg,yg,], dummy_func(xg,yg))
    plot.figure()
    plot.imshow(vorticity_mag_grid)
    plot.show()
#------------------------------------------------------------------

#To test changing z value being plotted
def plotpts_w_gas_no_dust_diff_z(snum=0, sdir='./output', boxL_z=2, ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=False, gas_val_toplot='rho', ptype_im='PartType0',
                  zmed_set=-1.e10, cmap='terrain', imdir='./images/', xz=0, yz=0):
    pylab.close('all');
    #plot.figure(1, figsize=(21., 7.))
    fig, ax = plot.subplots()
    P_File = load_snap(sdir, snum);
    # print('Time == ', P_File['Header'].attrs['Time'])
    P = P_File[ptype]
    # print('Dust-to-Gas Mass Ratio = ',
    #       np.sum(P_File['PartType3']['Masses'][:]) / np.sum(P_File['PartType0']['Masses'][:]))
    # a = P['GrainSize'][:]
    # print('Grain Size: min=', a.min(), ' max=', a.max())
    Pc = np.array(P['Coordinates']);

    # vx = np.array(P['Velocities'][:, 0])
    # vy = np.array(P['Velocities'][:, 1])
    # vz = np.array(P['Velocities'][:, 2])
    # print('Var v_x_dust == ', np.min(vx), np.median(vx), np.max(vx), np.std(vx), ' (min/median/max/std)')
    # print('Var v_y_dust == ', np.min(vy), np.median(vy), np.max(vy), np.std(vy), ' (min/median/max/std)')
    # print('Var v_z_dust == ', np.min(vz), np.median(vz), np.max(vz), np.std(vz), ' (min/median/max/std)')

    for subplot, xz, yz in zip([1, 2, 3], [xz], [yz]):  # ,1,0],[0,0,1]):
        xplotc = 0;
        yplotc = 1;
        depth_c = 2;  # plot xy, z=depth
        if (xz == 1):
            xplotc = 0;
            yplotc = 2;
            depth_c = 1;  # plot xz, y=depth
        if (yz == 1):
            xplotc = 1;
            yplotc = 2;
            depth_c = 0;  # plot yz, x=depth

        quiet = True
        if (subplot == 1): quiet = False  # give numbers once
        if xz == 0 and yz == 0:
            z = Pc[:, depth_c];
            zmx = np.max(z) - np.min(z);
            z0 = np.median(z);
            if (zmed_set > -1.e9): z0 = zmed_set
            zz = np.abs(z - z0);
            zz[(zz > 0.5 * zmx)] = zmx - zz[(zz > 0.5 * zmx)]
            ok = np.where(zz < width)
            #ok = np.where(np.abs(zz) > -1.)
            xx = Pc[:, xplotc][ok];
            yy = Pc[:, yplotc][ok];
            dmax = np.max(xx), np.max(yy)
        elif xz == 1:
            xx = Pc[:, 0]
            yy = Pc[:, 1]
            z0 = 0

        #x = 1.0 * (xx / dmax[0] + 0.);
        #y = 1.0 * (yy / dmax[1] + 0.);

        x = np.copy(xx)
        y = np.copy(yy)
        #pdb.set_trace()
        # np.savetxt("x.txt",x)
        # np.savetxt("y.txt",y)
        # XY = np.zeros((x.size()[0],4))
        # XY[:,0] = x[:]
        # XY[:,1] = y[:]
        # XY[:,2] = vx[:]
        # XY[:,3] = vx[:]
        #exit()

        #pylab.subplot(1, 3, subplot)
        ok_r = (np.random.rand(x.size) < cut_dust)
        print('oK_R = ', ok_r)
        print(x[ok_r])

        #Changing z0 value
        z0 = 1.99
        print('HERE z0 = ', z0)

        gas_rho_image(ax, snum=snum, sdir=sdir, boxL_z=boxL_z, xmax=1., xz=xz, yz=yz, gas_val_toplot=gas_val_toplot, zmed_set=z0,
                      vmin=vmin, vmax=vmax, quiet=quiet, cmap=cmap, ptype=ptype_im)

        if (forsavedfigure == True):
            ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
                       linestyle='', color='black', markeredgecolor='None', zorder=3)
            #Plot velocity streamlines
            xg, yg, vxgrid, vygrid = load_v(P_File, part='PartType0', xz=0, ngrid=1024,return_coords=True)
            # ax.streamplot(xg*6, yg*6, vxgrid, vygrid,linewidth=1.0)
            ax.streamplot(xg*6-3.0, yg*6-3.0, vxgrid, vygrid,linewidth=1.0)
            ax.set_xlim([-3,3])
            ax.set_ylim([-3,3])

            #X,Y = np.meshgrid(x, y)
            #plot.streamplot(X,Y,vx,vy)
        else:
            ax.scatter(x[ok_r], y[ok_r], markersize, marker='.', alpha=alpha, color='black', zorder=3)
            #ax.plot(x[ok_r], y[ok_r], marker='.', markersize=markersize, alpha=alpha,
            #           linestyle='', color='black', markeredgecolor='None', zorder=3)

        frame1 = plot.gca()  # get coordinate axes, to use for tick manipulation below
        #frame1.axes.xaxis.set_ticklabels([])  # no tick names
        #frame1.axes.yaxis.set_ticklabels([])  # no tick names
        #frame1.axes.get_xaxis().set_ticks([])  # no ticks
        #frame1.axes.get_yaxis().set_ticks([])  # no ticks
    
    ax.set_title('Time = %i'%P_File['Header'].attrs['Time'])

    if (forsavedfigure == True):
        ext = '000' + str(snum);
        if (snum >= 10): ext = '00' + str(snum)
        if (snum >= 100): ext = '0' + str(snum)
        if (snum >= 1000): ext = str(snum)
        #pylab.savefig(imdir + 'im_' + ext + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        #pylab.savefig(imdir + 'im_mass_0_5_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
        pylab.savefig(imdir + 'im_density_z0_2_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0)

    P_File.close()
    pylab.close()
    #plot.show()

