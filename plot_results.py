# maya tatarelli
# import h5py
import matplotlib.pyplot as pl
from radprof_concentration import *
from plot_snapshot import *
#========================================================
# - Streamline plot
#--------------------------------------------------------
def plot_streamlines(
    snum,
    sdir,
    imdir,
    ptype='PartType0'):

    P_File = load_snap(sdir, snum);
    xg, yg, vxgrid, vygrid = load_v(P_File, part='PartType0', xz=0, ngrid=1024,return_coords=True)
        
    # plot commands
    #fig, ax = pl.subplots(figsize=(6,6))
    pl.streamplot(xg, yg, vxgrid, vygrid)
    pl.xlabel('x')
    pl.ylabel('y')
    #pl.streamplot(yg, xg, vygrid, vxgrid)
    pl.savefig(imdir+'streamplot_'+str(snum)+'.pdf',dpi=500)
    pl.close()
    return
#========================================================

# gas_rho_image(ax, snum=3, sdir='./output', vmin=0., vmax=0., ptype='PartType0',
#                   cmap='terrain', xmax=1., xz=0, yz=0, gas_val_toplot='rho', rhocut=-2.5, save='dummy',
#                   zmed_set=-1.e10, quiet=False, zdir='z', zs=0.)

# for i in range(0,70):
#     plot_streamlines(
#         snum=i,
#         sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-23/output/',
#         imdir='./runs/2023-02-23/streamplots/',
#         ptype='PartType0')

#gas_rho_image(snum=54,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output',imdir='./streamplot_images/',ptype='PartType3',alpha=0.08, forsavedfigure=True)

#exit()

# plot_streamlines(
#     snum=20,
#     sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-23/output/',
#     imdir='./runs/2023-02-23/streamplots/',
#     ptype='PartType0')

plotpts_w_gas(snum=100,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-24/output/',imdir='./runs/2023-02-24/run_no_dust/',alpha=0.08, forsavedfigure=True,cmap='hot',vmin=-1.2,vmax=3.2)
plt_Pprof_maya(i=100, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-24/output/', profdir='./runs/2023-02-24/run_no_dust/')
exit()

plotpts_w_gas(snum=54,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output',ptype='PartType3',alpha=0.08, forsavedfigure=True)
plt_Pprof_maya(i=54, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output/', profdir='./plots/')


plotpts_w_gas(snum=200,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/output', alpha=0.08, forsavedfigure=True, imdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/images/')
plt_Pprof_maya(i=200, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/output/', profdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/plots/')



plotpts_w_gas(snum=53,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/output', alpha=0.08, forsavedfigure=True)
plt_Pprof_maya(i=53, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-02-01-00/output/', profdir='./plots/')

plt_Pprof_maya(i=54, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output/', profdir='./plots/')

# plt_Pprof_maya(i=54, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output/', profdir='./plots/')


plotpts_w_gas(snum=54,sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-30-01/output',ptype='PartType3',alpha=0.08, forsavedfigure=True)
plotpts_w_gas(
	snum=52, 
	sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-31-00/output',
	ptype='PartType3', 
	width=0.05, 
	cut_dust=1., 
	alpha=0.08, 
	markersize=5.,
    vmin=0, 
    vmax=0, 
    forsavedfigure=True, 
    gas_val_toplot='p', 
    ptype_im='PartType0',
    zmed_set=-1.e10, 
    cmap='terrain', 
    imdir='./images/', 
    xz=0, 
    yz=0)

exit()

plt_Pprof_maya(i=52, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-31-00/output/', profdir='./plots/')
plt_Pprof_maya(i=58, outputdir='/Users/mayascomputer/Codes/gizmo_code/runs/2023-01-31-01/output/', profdir='./plots/')
