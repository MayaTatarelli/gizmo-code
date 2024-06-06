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

#STEADY STATE
# plotpts_w_gas_no_dust(snum=351, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/z_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=355, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12,  boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/y_1_', xz=0, yz=0, 
#                   plot_zx=True, plot_zy=False, xlabel='$x/H$', ylabel='$z/H$', str_color=None)

# plotpts_w_gas_no_dust(snum=355, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
#                   vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=False,
#                   zmed_set=6.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state/x_1_', xz=0, yz=0, 
#                   plot_zx=False, plot_zy=True, xlabel='$y/H$', ylabel='$z/H$', str_color=None)

#ADDING DUST
plotpts_w_gas_no_dust(snum=183, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/output', ptype='PartType0', width=0.05, cut_dust=1., alpha=0.1, markersize=5.,
	                  vmin=0, vmax=0, forsavedfigure=True, gas_val_toplot='rho', ptype_im='PartType0', include_dust=True,
	                  zmed_set=4.0, boxL_xy=12, boxL_z=8, cmap='hot', imdir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Winter2024/3d_shearing_box/box_xy_12_z_8_dust_steady_state_add_dust/z_1_', xz=0, yz=0, 
	                  plot_zx=False, plot_zy=False, xlabel='$x/H$', ylabel='$y/H$', str_color=None)
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
