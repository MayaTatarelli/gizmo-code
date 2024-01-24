from global_disk import *
import numpy as np
import matplotlib.pyplot as plt 

r = np.linspace(0.2,2,1000)
nu = 5e-8
Omega = r**(-3./2.)

viscous_timescale = (r**2)/nu
evolution_timescale = 1000/Omega

plt.figure()
plt.plot(r, viscous_timescale, label='viscous timescale')
plt.plot(r, evolution_timescale, linestyle='--', color='k', label='evolution timescale')
plt.legend()
plt.show()

exit()



all_r, num_particles_at_r, all_centrifugal_accel = get_value_profile(use_fname=True, 
													fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
													val_to_plot='centrifugal_accel', ptype='PartType0',
													p=-1.0, rho_target=5.0, temp_p=-0.5,
													plot_all=False)

v_phi_theoretical = np.sqrt((all_r**(-3/2))**2 * all_r**2 + (-1.0-7/4)*0.0025/all_r**0.5)
centrifugal_theoretical = v_phi_theoretical*v_phi_theoretical/all_r

diff = all_centrifugal_accel - centrifugal_theoretical
print(np.sum(num_particles_at_r))
plt.figure()
plt.plot(all_r,diff)
plt.show()
exit()

run_dir = '/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'
runs= 'adding_outer_radii_press_force_rho_40'

all_r, num_particles_at_r, density, density_theoretical, all_vel_phi, v_phi_theoretical = get_value_profile(snum=50,
														sdir=run_dir+runs+'/output/',
														val_to_plot='resolution_test', ptype='PartType0',
														p=-1.0, rho_target=40.0, temp_p=-0.5,
														plot_all=False)
diff_density = density - density_theoretical
diff_velocity = all_vel_phi - v_phi_theoretical

diff_density = diff_density[~np.isnan(diff_density)]
diff_velocity = diff_velocity[~np.isnan(diff_velocity)]

avg_diff_density = np.average(diff_density)
avg_diff_velocity = np.average(diff_velocity)

print(avg_diff_density)
print(avg_diff_velocity)

exit()

all_r, num_particles_at_r, all_centrifugal_accel = get_value_profile(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/',
													use_fname=False, 
													fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_10.hdf5',
													val_to_plot='centrifugal_accel', ptype='PartType0',
													p=-1.0, rho_target=5.0, temp_p=-0.5,
													plot_all=False)

v_phi_theoretical = np.sqrt((all_r**(-3/2))**2 * all_r**2 + (-1.0-7/4)*0.0025/all_r**0.5)
centrifugal_theoretical = v_phi_theoretical*v_phi_theoretical/all_r

diff = all_centrifugal_accel - centrifugal_theoretical

plt.figure()
plt.plot(all_r,diff)
plt.show()

# get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC')

# get_radial_force_balance(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', 
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 50',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

# get_radial_force_balance(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', 
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 100',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


# get_radial_force_balance(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', 
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 200',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')