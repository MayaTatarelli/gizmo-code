import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interpolate
from plot_snapshot import load_snap, check_if_filename_exists
from radprof_concentration import load_v
from global_disk import *

#For testing new Keplerian IC:
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_vary_prtcl_mass_large_inner_spacing.hdf5', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=6, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_ic_2d_new_dr/output/', phil=False)

# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_new_dr_002.hdf5', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_ic_2d_new_dr_002/output/', phil=False)

# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand.hdf5', phil=False)
# plot_velocity_streamlines(snum=2, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_ic_2d_with_cs/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/', phil=False)
# plot_velocity_streamlines(snum=7, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=2, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', phil=False)


# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=38, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', phil=False)


# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho_1_5/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_high_rho_1_5/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_high_rho_1_5/output/', phil=False)


# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', phil=False)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)

# plot_velocity_streamlines(snum=3, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand/output/', phil=False)


# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_with_dust.hdf5', ptype='PartType3', phil=False)

# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho/output/', dr_factor=0.9, p=-1.0, rho_target=3.0,temp_p=0.0)
# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho_1_5/output/', dr_factor=0.6, p=-1.5, rho_target=5.0,temp_p=0.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_nonuniform_high_rho_1_5/output/', dr_factor=0.6, p=-1.5, rho_target=10.0,temp_p=0.0)


# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', phil=False)


# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', phil=False)

# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_low_tol/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)

# plot_velocity_streamlines(snum=20, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_with_bndry/output/', phil=False)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_with_bndry/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=20, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_with_bndry/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_with_bndry/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=20, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_with_bndry/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)

# calculate_radial_vel(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', vel_to_plot='radial')
# calculate_radial_vel(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', vel_to_plot='radial')
# calculate_radial_vel(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', vel_to_plot='radial')

# plot_velocity_streamlines(snum=13, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', phil=False)
# plot_velocity_streamlines(snum=80, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', phil=False)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0)
# calculate_radial_vel(snum=13, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', vel_to_plot='radial')

# plot_value_profile(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.0)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_03H/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_02H/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.2)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0)

# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=2, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0)

# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0)
# plot_value_profile(snum=53, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=36, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0)

# plot_value_profile(snum=36, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_03H/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_02H/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.2)
# plot_value_profile(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.0)

# plot_velocity_streamlines(snum=36, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', phil=False)
# plot_velocity_streamlines(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_03H/', phil=False)
# plot_velocity_streamlines(snum=32, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_02H/', phil=False)
# plot_velocity_streamlines(snum=53, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output/', phil=False)

#Only pressure gradient force with no other inner boundary condition, 0.3H
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', phil=False)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', phil=False)
# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3, plot_all=True)
# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3, plot_all=True)
# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='temp', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3, plot_all=True)
# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_only_inner_press_grad/output/', val_to_plot='vel_radial', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=0.3, plot_all=True)

# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0, plot_all=False)
# exit()
# plot_value_profile(snum=36, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_inner_press_grad/output_H/', val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0, plot_all=True)
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_temp/')
# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_temp/')
# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_temp/')
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', phil=False)
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', phil=False)
# exit()
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=3.0, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/')
# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=3.0, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/')
# exit()

# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=3.0, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/')

# plot_value_profile(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=3.0, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/')


####Trying to test uniform#######################################

# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=1.5, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/other_')

# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=1.5, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/other_')

# plot_value_profile(snum=100, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs/output/',
# 					val_to_plot='vel_radial', dr_factor=0.9, p=0.0, temp_p=0.0, rho_target=1.5, H_factor=0.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/uniform/other_')
#################################################################

# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=False, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_IE_IEPred/testrho/')

#Template for fixed_IE_IEPred run
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_IE_IEPred/')

#Template for fixed_IE_IEPred_10H run
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=10.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_IE_IEPred_10H/')
# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=10.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_IE_IEPred_10H/')
# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=10.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_IE_IEPred_10H/')
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', phil=False)
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred_10H/output/', phil=False)


#adding_outer_radii_press_force
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', phil=False)

#adding_outer_radii_press_force_rho_10
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', phil=False)

#adding_outer_radii_press_force_rho_20
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_20/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_20/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_20/output/', phil=False)
# plot_velocity_streamlines(snum=73, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_20/output/', phil=False)
# exit()

#adding_outer_radii_press_force_rho_30
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_30/output/', phil=False)
# plot_velocity_streamlines(snum=21, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_30/output/', phil=False)
# exit()

#adding_outer_radii_press_force_rho_40
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', phil=False)
# plot_velocity_streamlines(snum=299, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', phil=False)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/resolution_test/res_test_5/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/resolution_test/res_test_5/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/resolution_test/res_test_4/', phil=False)

# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# plot_velocity_streamlines(snum=751, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# exit()


#adding_viscosity - test_3
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_3/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_3/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_3/', phil=False)
# plot_velocity_streamlines(snum=189, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_3/', phil=False)
# exit()

#keplerian_ic_2d_rho_temp_gradient_mass_0_01
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)
# plot_velocity_streamlines(snum=100, sdir='/Users/mayatatare19i/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_mass_0_01/output/', phil=False)

#adding_viscosity - test_1
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', phil=False)

#damping_boundaries - test0
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', phil=False)

#damping_boundaries - damping code in run loop in run.c
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/damping_in_run_loop/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/damping_in_run_loop/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/damping_in_run_loop/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/damping_in_run_loop/output/', phil=False)

#test_boundary_cond
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_inner_bndry_cond/output/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_inner_bndry_cond/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_inner_bndry_cond/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_inner_bndry_cond/output/', phil=False)

# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/inner_press_grad_only/output/', phil=False)

# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/remove_energy_switch/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_confining_press_grad/output/', phil=False)

# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_bndry_cond/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_confining/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_press_grad/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_confining_press_grad_large_inner/output/', phil=False)

#smaller inner radius
# plot_velocity_streamlines(snum=48, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=49, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=51, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=52, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)

# plot_velocity_streamlines(snum=195, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=196, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=197, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=198, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=199, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)

#Smaller inner radius with differing bndry conditions
# plot_velocity_streamlines(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/smaller_in_bndry/output/', phil=False)
# plot_velocity_streamlines(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/inner_press_grad_only_01_radius/output/', phil=False)
# plot_velocity_streamlines(snum=99, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/no_inner_bndry_cond_01_radius/output/', phil=False)

#Inner bndry inflow (setting mass to 0)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/inner_bndry_inflow/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/inner_bndry_inflow_01_radius/output/', phil=False, vmax=9e-3)

#adding_viscosity - no_in_bndry_cond_low_visc
# plot_velocity_streamlines(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/no_in_bndry_cond_low_visc/output/', phil=False, vmax=5e-3)

#IC
# plot_velocity_streamlines(snum=0, sdir='/', use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01.hdf5', phil=False, vmax=5e-3)

#adding_viscosity - inflow_low_visc
# plot_velocity_streamlines(snum=370, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inflow_low_visc/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inflow_low_visc/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_both_outer_half_pressgrad_low_visc/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_outflow_half_pressgrad_low_visc/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_confining_low_visc/output/', phil=False, vmax=5e-3)

#inner_inflow_outer_both_low_visc_vary_prtcl_mass_2
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_both_low_visc_vary_prtcl_mass_2/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_both_low_visc_vary_prtcl_mass_2/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_both_low_visc_vary_prtcl_mass_2/output/', phil=False, vmax=5e-3)

#outer_both_low_visc_vary_prtcl_mass_large_r_out
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_vary_prtcl_mass_large_r_out.hdf5',vmax=5e-3)
# plot_velocity_streamlines(snum=1000, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_out/output/', phil=False, vmax=5e-3)

#inner_inflow_outer_press_grad_high_visc_min
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/inner_inflow_outer_press_grad_high_visc_min/output/', phil=False, vmax=5e-3)

#outer_both_low_visc_vary_prtcl_mass_large_r_out_r_in
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_out_r_in/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=1000, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_out_r_in/output/', phil=False, vmax=5e-3)

#outer_both_low_visc_vary_prtcl_mass_large_r_outflow
# plot_velocity_streamlines(snum=71, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_outflow/output/', phil=False, vmax=5e-3)
# plot_velocity_streamlines(snum=1000, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_outflow/output/', phil=False, vmax=5e-3)

#outer_both_high_visc_vary_prtcl_mass_large_r_outflow
# plot_velocity_streamlines(snum=71, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_high_visc_vary_prtcl_mass_large_r_outflow/output/', phil=False, vmax=5e-3)

#outer_both_med_visc_vary_prtcl_mass_large_r_outflow
# plot_velocity_streamlines(snum=494, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_med_visc_vary_prtcl_mass_large_r_outflow/output/', phil=False, vmax=5e-3)

#outer_both_med_visc_vary_prtcl_mass_large_r_outflow_2
# plot_velocity_streamlines(snum=500, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_med_visc_vary_prtcl_mass_large_r_outflow_2/output/', phil=False, vmax=5e-3)

#outer_both_low_visc_vary_prtcl_mass_large_r_out_confine
# plot_velocity_streamlines(snum=183, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_out_confine/output/', phil=False, vmax=5e-3)

#adding_ghost_particles
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ic_2d_vary_prtcl_mass_ghost_particles.hdf5', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_4/output/', plot_ghost=True, phil=False, vmax=7e-3)
# plot_velocity_streamlines(snum=97, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_4/output/', plot_ghost=True, phil=False, vmax=7e-3)
# plot_velocity_streamlines(snum=122, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_4/output/', plot_ghost=True, phil=False, vmax=7e-3)
# plot_velocity_streamlines(snum=881, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_4/output/', plot_ghost=True, phil=False, vmax=7e-3)

#test_6_circOrbitPos
# plot_velocity_streamlines(snum=583, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_6_circOrbitPos/output/', plot_ghost=False, phil=False, vmax=4e-3)
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_6_circOrbitPos/output/', plot_ghost=True, phil=False, vmax=7e-3)
# plot_velocity_streamlines(snum=583, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_6_circOrbitPos/output/', plot_ghost=True, phil=False, vmax=7e-3)

#test_7
# plot_velocity_streamlines(snum=356, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_7/output/', plot_ghost=True, phil=False, vmax=7e-3)
plot_velocity_streamlines(snum=356, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_ghost_particles/test_7/output/', plot_ghost=False, phil=False, vmax=7e-3)

#adding_viscosity - test_2
# plot_velocity_streamlines(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', phil=False)
# plot_velocity_streamlines(snum=1, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', phil=False)
# plot_velocity_streamlines(snum=70, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', phil=False)

#Template for fixed_vphi run
# plot_value_profile(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_vphi/')

# plot_value_profile(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_vphi/')

# plot_value_profile(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/',
# 					val_to_plot='rho', dr_factor=0.1, p=-1.0, rho_target=5.0, H_factor=1.0,
# 					plot_all=True, output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/temp_profiles_for_diff_bound_cond/fixed_vphi/')

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/', phil=False)
# plot_velocity_streamlines(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/', phil=False)

####################################################################################
#For research progress meeting with Eve -- July 19th 2023
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_phil/output/')
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_phil/output/')

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_phil_public_00/output/')
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_phil_public_00/output/')

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_maya_public_with_vel_new_test/output/', phil=False)
# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_maya_public_with_vel_new_test/output/', phil=False)
####################################################################################

# plot_velocity_streamlines(snum=5, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_phil/output/')
# plot_velocity_streamlines(snum=1, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_maya_public_with_vel_phil_coords/output/', phil=True)
# plot_velocity_streamlines(use_fname=True, fname='./keplerian_disk_2d_with_vel_r_0.hdf5', phil=False)
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ics.hdf5', phil=True)
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_disk_2d_with_vel_phil_coords_r_0.hdf5', phil=True)

# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_maya_public_updated_IC_no_tol/output/', phil=True)

# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_disk_2d_updated_w_shift_phil_coords.hdf5', phil=True)

# plot_velocity_streamlines(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_disk_2d_updated_phil_coords_omega00/output/', phil=False)
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_disk_2d_updated_phil_coords_minusphi/output/', phil=False)
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_disk_2d_updated_phil_coords_minusphi.hdf5', phil=True)

#For comparing initial velocities:
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_phil_public_00/output/')
# plot_velocity_streamlines(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_phil_ic_maya_code/output/', phil=False)
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_disk_2d_updated_w_shift_phil_coords.hdf5', phil=True)
# plot_velocity_streamlines(use_fname=True, fname='./ICs/keplerian_ics.hdf5', phil=True)

# plot_velocity_streamlines(snum=8, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_disk_2d_updated_maya_coords/output/', phil=False)
# plot_velocity_streamlines(snum=8, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/keplerian_disk_2d_updated_phil_coords_intEnergyFix/output/', phil=False)
# plot_velocity_streamlines(snum=8, sdir='/Users/mayascomputer/Codes/gizmo-public/runs/2d_keplerian/phil_test/output/', phil=False)

