import numpy as np 
import matplotlib.pyplot as plt
from global_disk import *

runs = np.array(['test_boundary_cond/bndry_confining_press_grad_large_inner/output/', 'test_boundary_cond/bndry_confining_press_grad/output/', 'test_boundary_cond/no_bndry_cond/output/', 'test_boundary_cond/bndry_confining/output/', 'test_boundary_cond/bndry_press_grad/output/'])
fig_dir = np.array(['bndry_confining_press_grad_large_inner/', 'bndry_confining_press_grad/', 'no_bndry_cond/', 'bndry_confining/', 'bndry_press_grad/'])
titles = np.array(['Pressure gradient and confining at boundaries (large inner bndry radius)', 'Pressure gradient and confining at boundaries', 'No boundary conditions', 'Only confining boundaries', 'Only pressure gradient at boundaries'])
bndry_in = np.array([0.5, 0.227, 0.2, 0.2, 0.227])

#Density profile
def plot_density(snum=0, r_in=0.2, rho_target=4.42e-3, runs=np.array(['./run/']), fig_dir=np.array(['./run/']), titles=np.array(['run']), bndry_in=np.array([0.0])):
	for i in range(len(runs)):

		all_r, num_particles_at_r, density, rho_volume, density_theoretical = get_value_profile(snum=snum,
										sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'+runs[i],
										val_to_plot='rho', ptype='PartType0', r_in=r_in,
										dr_factor=0.1, p=-1.0, rho_target=rho_target, temp_p=-0.5, plot_all=False)

		# all_r = np.array([0,1,2,3])
		# density = np.array([1,1,2,3])
		# density_theoretical = np.array([1,1,2,3])	

		plt.figure()
		plt.plot(all_r, density/density_theoretical, marker='.', linestyle='None', label='Actual/Theoretical')
		plt.axvline(x=bndry_in[i], linestyle='--',color='orange', label='Inner bndry forcing limit')
		plt.xlabel('Radius', size=11)
		plt.ylabel("$\\Sigma$", size=13)
		plt.xlim([r_in, 2.0])
		plt.title(titles[i])
		plt.legend()
		plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'surf_density_'+str(snum)+'.pdf')
		# plt.show()
		# exit()

#Velocity Profile
def plot_velocity(snum=0, r_in=0.2, rho_target=4.42e-3, runs=np.array(['./run/']), fig_dir=np.array(['./run/']), titles=np.array(['run']), bndry_in=np.array([0.0])):
	for i in range(len(runs)):
		all_r, num_particles_at_r, all_vel_radial, all_vel_phi, v_phi_theoretical = get_value_profile(snum=snum,
										sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'+runs[i],
										val_to_plot='vel_both', ptype='PartType0', r_in=r_in,
										dr_factor=0.1, p=-1.0, rho_target=rho_target, temp_p=-0.5, plot_all=False)

		plt.figure()
		plt.plot(all_r, all_vel_radial, marker='.', linestyle='None', label='Actual')
		plt.axhline(y=0, linestyle='--',color='orange', label='Theor')
		plt.xlabel('Radius', size=11)
		plt.ylabel("Radial velocity", size=11)
		# plt.xlim([0.2, 0.35])
		plt.title(titles[i])
		plt.legend()
		plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'vel_radial_'+str(snum)+'.pdf')
		# plt.show()

		plt.figure()
		# plt.plot(all_r, all_vel_phi, marker='.', linestyle='None', label='Actual')
		# plt.plot(all_r, v_phi_theoretical, marker='.', linestyle='None', label='Theor')
		plt.plot(all_r, all_vel_phi/v_phi_theoretical, marker='.', linestyle='None', label='Actual/Theoretical')
		plt.xlabel('Radius', size=11)
		plt.ylabel("Azimuthal velocity", size=11)
		# plt.xlim([0.2, 0.35])
		plt.title(titles[i])
		plt.legend()
		plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'vel_phi_'+str(snum)+'.pdf')
		# plt.show()
		# exit()

#Acceleration profile
def plot_radial_accel(snum=0, r_in=0.2, rho_target=4.42e-3, runs=np.array(['./run/']), fig_dir=np.array(['./run/']), titles=np.array(['run']), bndry_in=np.array([0.0])):
	for i in range(len(runs)):
		calculate_radial_accel(snum=snum, 
							sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'+runs[i],
							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
							ptype='PartType0', r_in=r_in, p=-1.0, temp_p=-0.5, rho_target=rho_target, gamma = 7./5.,
							plot_title=titles[i],
							output_plot_dir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'radial_accel_'+str(snum)+'.pdf')

def plot_radial_profiles(snum=0, r_in=0.2, rho_target=4.42e-3, runs=np.array(['./run/']), fig_dir=np.array(['./run/']), titles=np.array(['run']), bndry_in=np.array([0.0])):
	plot_density(snum, r_in, rho_target, runs, fig_dir, titles, bndry_in)
	plot_radial_accel(snum, r_in, rho_target, runs, fig_dir, titles, bndry_in)

#Damping

# plot_velocity(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['damping_boundaries/damping_in_run_loop/output/']), fig_dir=np.array(['damping_in_run_loop/']), titles=np.array(['Damping at inner bndry']), bndry_in=np.array([0.227]))

# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['damping_boundaries/damping_in_run_loop/output/']), fig_dir=np.array(['damping_in_run_loop/']), titles=np.array(['Damping at inner bndry']), bndry_in=np.array([0.227]))

# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['damping_boundaries/test0/output/']), fig_dir=np.array(['damping_inner_bndry/']), titles=np.array(['Damping at inner bndry (code in analytic grav)']), bndry_in=np.array([0.227]))


# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['test_boundary_cond/no_inner_bndry_cond/output/']), fig_dir=np.array(['no_inner_bndry_cond/']), titles=np.array(['No inner bndry conditions']), bndry_in=np.array([0.227]))


# plot_velocity(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['test_boundary_cond/inner_press_grad_only/output/']), fig_dir=np.array(['inner_press_grad_only/']), titles=np.array(['Inner bndry with only press grad']), bndry_in=np.array([0.227]))

# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['test_boundary_cond/inner_press_grad_only/output/']), fig_dir=np.array(['inner_press_grad_only/']), titles=np.array(['Inner bndry with only press grad']), bndry_in=np.array([0.227]))

#Smaller inner radius
# plot_radial_profiles(snum=200, r_in=0.1, rho_target=8.38e-3, 
# 	runs=np.array(['test_boundary_cond/smaller_in_bndry/output/']), fig_dir=np.array(['smaller_in_bndry/']), titles=np.array(['Smaller inner radius (at 0.1)']), bndry_in=np.array([0.227]))

#Smaller inner radius with differing bndry conditions
# plot_radial_profiles(snum=0, r_in=0.1, rho_target=8.38e-3, 
# 	runs=np.array(['test_boundary_cond/inner_press_grad_only_01_radius/output/', 'test_boundary_cond/no_inner_bndry_cond_01_radius/output/']), fig_dir=np.array(['inner_press_grad_only_01_radius/', 'no_inner_bndry_cond_01_radius/']), titles=np.array(['Smaller inner radius (at 0.1) - inner press grad only', 'Smaller inner radius (at 0.1) - no inner bndry cond']), bndry_in=np.array([0.227, 0.0]))

#Inner bndry inflow (setting mass to 0)

# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['test_boundary_cond/inner_bndry_inflow/output/']), fig_dir=np.array(['inner_bndry_inflow/']), titles=np.array(['Inner bndry inflow (0.2 inner radius)']), bndry_in=np.array([0]))

# plot_radial_profiles(snum=100, r_in=0.1, rho_target=8.38e-3, 
# 	runs=np.array(['test_boundary_cond/inner_bndry_inflow_01_radius/output/']), fig_dir=np.array(['inner_bndry_inflow_01_radius/']), titles=np.array(['Inner bndry inflow (0.1 inner radius)']), bndry_in=np.array([0]))

#Adding viscosity

plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
	runs=np.array(['adding_viscosity/no_in_bndry_cond_low_visc/output/']), fig_dir=np.array(['no_in_bndry_cond_low_visc/']), titles=np.array(['No inner bndry cond - low viscosity (0.2 inner radius)']), bndry_in=np.array([0]))

#Removing energy entropy switch
# plot_radial_profiles(snum=200, r_in=0.2, rho_target=4.42e-3, 
# 	runs=np.array(['test_boundary_cond/remove_energy_switch/output/']), fig_dir=np.array(['remove_energy_switch/']), titles=np.array(['Remove energy entropy switch']), bndry_in=np.array([0.227]))

#Testing snap 0 for different runs (should all look the same)
# calculate_radial_accel(snum=0, 
# 							sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/',
# 							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 							plot_title='Damping inner boundary',
# 							output_plot_dir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/damping_inner_bndry/radial_accel_200.pdf')

# calculate_radial_accel(snum=0, 
# 							sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_confining_press_grad/output/',
# 							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 							plot_title='Pressure gradient and confining at boundaries',
# 							output_plot_dir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/damping_inner_bndry/radial_accel_200.pdf')

# calculate_radial_accel(snum=0, 
# 							sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/bndry_confining/output/',
# 							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 							plot_title='Only confining boundaries',
# 							output_plot_dir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/damping_inner_bndry/radial_accel_200.pdf')

# calculate_radial_accel(snum=0, 
# 							sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/',
# 							use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01.hdf5',
# 							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 							plot_title='Damping inner boundary',
# 							output_plot_dir='/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/damping_inner_bndry/radial_accel_200.pdf')
	

