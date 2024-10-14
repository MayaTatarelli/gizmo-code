import matplotlib.pyplot as pl
from radprof_concentration import *
from plot_snapshot import *

# plotpts_w_gas(snum=1,sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_test_1/output/',imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_test_1/',alpha=0.08, forsavedfigure=True,cmap='hot',vmin=-1.2,vmax=3.2)
# plt_Pprof_maya(i=1, outputdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_test_1/output/', profdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_test_1/')


# plotpts_w_gas_no_dust(snum=68, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_test_1/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=-1e10, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_test_1/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=68, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_test_1/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=3.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_test_1/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=68, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=3.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/no_damping/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=68, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/no_damping/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=1.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/no_damping/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

#damping_in_run_loop

# plotpts_w_gas_no_dust(snum=84, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_in_run_loop/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_in_run_loop/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=84, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_in_run_loop/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=3.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_in_run_loop/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=84, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/damping_in_run_loop/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=1.0, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/damping_in_run_loop/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)



# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType3', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/z_1_dust_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType3', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/y_1_dust_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType3', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0',
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/x_1_dust_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)


#Adding dust particles
# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=61, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=9.65, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)


#Adding dust particles with outflow
# plotpts_w_gas_no_dust(snum=87, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=87, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=87, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
#                   zmed_set=9.5, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_outflow_fix/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)


#box_xy_12_z_8_dust_steady_state
# plotpts_w_gas_no_dust(snum=119, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=5.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=119, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=119, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

#box_xy_12_z_8_dust_steady_state_off_center
# plotpts_w_gas_no_dust(snum=27, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=27, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=27, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_off_center_2/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

#PLOTTING GAS FLUX

# plotpts_w_gas_no_dust(snum=900, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/gas_flux_testing/test_xy_H4_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None, return_gas_flux=True, vel_comp='y')

# plotpts_w_gas_no_dust(snum=900, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=-0.02, vmax=0.02, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/gas_flux_testing/test_xz_H6_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='z')


# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=-0.01, vmax=0.01, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=7.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/gas_flux_testing/xz_H7_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='x')

# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=-0.01, vmax=0.01, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=11.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/gas_flux_testing/xz_H11_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='x')

# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=-0.01, vmax=0.01, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=5.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/gas_flux_testing/xz_H5_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='x')

# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=-0.01, vmax=0.01, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=1.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/gas_flux_testing/xz_H1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='x')


# plotpts_w_gas_no_dust(snum=900, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=8.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC/gas_flux_testing/xz_H8_vz_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None, return_gas_flux=True, vel_comp='z')

# exit()

#STEADY STATE NO PLANET WITH LBOXXY FIXED
# plotpts_w_gas_no_dust(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/z_H4_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/y_H6_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/x_H6_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

# exit()


plotpts_w_gas_no_dust(snum=559, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=4.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/z_H4_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=559, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/y_H6_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

plotpts_w_gas_no_dust(snum=559, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_2_2res_c/x_H6_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

exit()

plotpts_w_gas_no_dust(snum=270, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=4.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/z_H4_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=270, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/y_H6_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)


plotpts_w_gas_no_dust(snum=300, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=4.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/z_H4_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=300, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/y_H6_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

plotpts_w_gas_no_dust(snum=401, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=4.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/z_H4_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=401, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet_fixLboxxy_no_BC_add_dust/y_H6_dust_streamlines_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

exit()

#STEADY STATE NO PLANET
# plotpts_w_gas_no_dust(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/z_H4_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=6.01, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/y_H6_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=1.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_planet/x_H1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

exit()

#STEADY STATE
# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
                  zmed_set=11, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/y_H11_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=369, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

exit()

#ADDING DUST
# plotpts_w_gas_no_dust(snum=270, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
# 	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
# 	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2_add_dust/z_1_', xz=0, yz=0, 
# 	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=350, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
                  zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/y_1_', xz=0, yz=0, 
                  plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

plotpts_w_gas_no_dust(snum=270, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
                  zmed_set=9.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_test_2_add_dust/x_1_', xz=0, yz=0, 
                  plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

exit()
plotpts_w_gas_no_dust(snum=250, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)
exit()
plotpts_w_gas_no_dust(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)
plotpts_w_gas_no_dust(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

plotpts_w_gas_no_dust(snum=300, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)
# plotpts_w_gas_no_dust(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust_test/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
# 	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
# 	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust_test/z_1_', xz=0, yz=0, 
# 	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)
exit()
for i in range(11): 
	plotpts_w_gas_no_dust(snum=i+90, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=57, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=57, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=57, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_2Mp/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)


# plotpts_w_gas_no_dust(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_no_damping/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)
