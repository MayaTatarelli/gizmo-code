from global_disk import *
import numpy as np
import matplotlib.pyplot as plt

run_dir = '/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'
# runs = np.array(['adding_outer_radii_press_force', 'adding_outer_radii_press_force_rho_10', 'adding_outer_radii_press_force_rho_20', 'adding_outer_radii_press_force_rho_30'])

#RUNS FOR NARVAL RESOLUTION TEST 
# runs = np.array(['resolution_test/res_test_1/', 'resolution_test/res_test_2/', 'resolution_test/res_test_3/', 'resolution_test/res_test_4/'])#, 'resolution_test/res_test_5/'])
# rho_target = 20.0
# runs_density = np.array([5.0,10.0,20.0,30.0])

#RUNS FOR TESTING ERROR AT DIFF RES INCLUDING VISCOSITY AND KEEPING MASS AS 0.01M_solar
runs = np.array(['adding_viscosity/res_test_1/', 'adding_viscosity/res_test_2/', 'adding_viscosity/res_test_3/', 'adding_viscosity/test_1/'])#, 'resolution_test/res_test_5/'])
rho_target = 4.42e-3

all_diff_density = np.zeros(0)
all_diff_velocity = np.zeros(0)
all_numP = np.zeros(0)
for i in range(len(runs)):
	# numP, diff_density, diff_velocity = calculate_avg_error(snum=20, sdir=run_dir+runs[i]+'/output/',
	# 							ptype='PartType0', dr_factor=0.1, p=-1.0, rho_target=runs_density[i], temp_p=-0.5)

	numP, diff_density, diff_velocity = calculate_avg_error(snum=200, sdir=run_dir+runs[i],
								ptype='PartType0', dr_factor=0.1, p=-1.0, rho_target=rho_target, temp_p=-0.5)

	print(numP)
	print(diff_density)
	print(diff_velocity)
	all_diff_density = np.append(all_diff_density, diff_density)
	all_diff_velocity = np.append(all_diff_velocity, diff_velocity)
	all_numP = np.append(all_numP, numP)

# all_diff_density = np.zeros(0)
# all_diff_velocity = np.zeros(0)
# all_num_particles = np.zeros(0)

# for i in range(1):#len(runs_density)):
# 	all_r, num_particles_at_r, density, density_theoretical, all_vel_phi, v_phi_theoretical = get_value_profile(snum=50,
# 														sdir=run_dir+runs[i]+'/output/',
# 														val_to_plot='resolution_test', ptype='PartType0',
# 														p=-1.0, rho_target=runs_density[i], temp_p=-0.5,
# 														plot_all=False)

# 	diff_density = density - density_theoretical
# 	diff_velocity = all_vel_phi - v_phi_theoretical
# 	print(diff_density)
# 	print(diff_velocity)

# 	diff_density = diff_density[~np.isnan(diff_density)]
# 	diff_velocity = diff_velocity[~np.isnan(diff_velocity)]

# 	avg_diff_density = np.average(diff_density)
# 	avg_diff_velocity = np.average(diff_velocity)

# 	print(avg_diff_density)
# 	print(avg_diff_velocity)

# 	all_diff_density = np.append(all_diff_density, avg_diff_density)
# 	all_diff_velocity = np.append(all_diff_velocity, avg_diff_velocity)

# 	print(np.sum(num_particles_at_r))
# 	all_num_particles = np.append(all_num_particles, np.sum(num_particles_at_r))

# exit()
# print("density: ", all_diff_density)
# print("velocity: ", all_diff_velocity)
# print("number particles: ", all_num_particles)

# density =  np.abs(np.array([-0.20977717, 0.17211527, -1.35483677]))
# velocity = np.abs(np.array([-0.00362217, -0.0039867, -0.00089053]))
# number_particles = np.abs(np.array([109547., 219923., 435760.]))

plt.figure()
plt.plot(all_numP, all_diff_density, marker='.')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.title('Resolution Analysis for Density (snapshot 200)', size=13)
plt.xlabel('Number of particles', size=11)
plt.ylabel("<$\\sigma_{\\Sigma}$>", size=15)
plt.show()

plt.figure()
plt.plot(all_numP, all_diff_velocity, marker='.')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.title('Resolution Analysis for Velocity (snapshot 200)', size=13)
plt.xlabel('Number of particles', size=11)
plt.ylabel("<$\\sigma_{v_{\\phi}}$>", size=15)
plt.show()