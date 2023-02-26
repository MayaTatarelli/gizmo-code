"""
Plot a snapshot of shearing box

@author: Eve J. Lee
May 12th 2019
"""
import numpy as np
import h5py as h5py
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
import matplotlib.pyplot as plot
import scipy.interpolate as interpolate
from mpl_toolkits.mplot3d import Axes3D
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

def gas_rho_image(ax, snum=3, sdir='./output', vmin=0., vmax=0., ptype='PartType0',
                  cmap='terrain', xmax=1., xz=0, yz=0, gas_val_toplot='rho', rhocut=-2.5, save='dummy',
                  zmed_set=-1.e10, quiet=False, zdir='z', zs=0.):
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
        zmx = np.max(zz) - np.min(zz);
        zzmed = np.median(zz);
        if (zmed_set > -1.e9): zzmed = zmed_set;
        dzz = np.abs(zz - zzmed);
        dzz[(dzz > 0.5 * zmx)] = zmx - dzz[(dzz > 0.5 * zmx)]
        if ('SmoothingLength' in P.keys()):
            ok = np.where(dzz < 0.5 * P['SmoothingLength'][:])
        else:
            ok = np.where(dzz < 0.05)

    x = 1.0 * (xx / dmax[0] + 0.);
    y = 1.0 * (yy / dmax[1] + 0.);
    z = 1.0 * (zz / dmax[2] + 0.);

    if ((gas_val_toplot == 'bvec') | (gas_val_toplot == 'vvec')):
        yg, xg = np.mgrid[0:1:128j, 0:1:128j]
    elif xz == 0:
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
            print('min/max/std velocity = ', np.min(v0), np.max(v0),
                  np.sqrt(np.std(v0[:, 0]) ** 2 + np.std(v0[:, 1]) ** 2 + np.std(v0[:, 2]) ** 2))
            print('min/max/std density = ', np.min(u), np.max(u), np.std(u))

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
        # print("here 1#######################################################")
        dg = interpolate.griddata((x, z), u, (xg, yg), method='linear', fill_value=np.median(u));
        im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[0], 0, dmax[2]),
                       zorder=1);
    elif yz == 1:
        # print("here 2##################################################################")
        dg = interpolate.griddata((y, z), u, (xg, yg), method='linear', fill_value=np.median(u));
        im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[1], 0, dmax[2]),
                       zorder=1);
    else:
        # print("here 3#######################################################")
        dg = interpolate.griddata((x[ok], y[ok]), u[ok], (xg, yg), method='linear', fill_value=np.median(u[ok]));
        # im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(0, dmax[0], 0, dmax[1]),
        #                zorder=1);
        im = ax.imshow(dg, interpolation='bicubic', vmin=vmin, vmax=vmax, cmap=cmap, extent=(-3, 3, -3, 3),
                       zorder=1);

    cbar = pylab.colorbar(im)
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

def plotpts_w_gas(snum=0, sdir='./output', ptype='PartType3', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
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
        gas_rho_image(ax, snum=snum, sdir=sdir, xmax=1., xz=xz, yz=yz, gas_val_toplot=gas_val_toplot, zmed_set=z0,
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
        pylab.savefig(imdir + 'im_' + ext + '.png', dpi=150, bbox_inches='tight', pad_inches=0)
        pylab.savefig(imdir + 'im_' + ext + '.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
    P_File.close()
    pylab.close()
    #plot.show()


