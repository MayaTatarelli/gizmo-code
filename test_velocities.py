import numpy as np 
import matplotlib.pyplot as plt
from global_disk import get_value_profile, get_velocities

runs = np.array(['bndry_confining_press_grad_large_inner/output/', 'bndry_confining_press_grad/output/', 'no_bndry_cond/output/', 'bndry_confining/output/', 'bndry_press_grad/output/'])
fig_dir = np.array(['bndry_confining_press_grad_large_inner/', 'bndry_confining_press_grad/', 'no_bndry_cond/', 'bndry_confining/', 'bndry_press_grad/'])
titles = np.array(['Pressure gradient and confining at boundaries (large inner bndry radius)', 'Pressure gradient and confining at boundaries', 'No boundary conditions', 'Only confining boundaries', 'Only pressure gradient at boundaries'])

for i in range(len(runs)):
	all_r, num_particles_at_r, all_vel_radial, all_vel_phi, v_phi_theoretical = get_value_profile(
								snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/test_boundary_cond/'+runs[i],
								val_to_plot='vel_both', ptype='PartType0',
								dr_factor=0.1, p=-1.0, rho_target=4.42e-3, temp_p=-0.5, plot_all=False)

	plt.figure()
	plt.plot(all_r, all_vel_radial, marker='.', linestyle='None', label='Actual')
	plt.axhline(y=0, linestyle='--',color='orange', label='Theor')
	plt.xlabel('Radius', size=11)
	plt.ylabel("Radial velocity", size=11)
	# plt.xlim([0.2, 0.35])
	plt.title(titles[i])
	plt.legend()
	plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'vel_radial_200.pdf')
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
	plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'vel_phi_200.pdf')
	# plt.show()
	# exit()
exit()

all_r, num_particles_at_r, all_vel_radial, all_vel_phi, v_phi_theoretical = get_value_profile(
								snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/',
								val_to_plot='vel_both', ptype='PartType0',
								dr_factor=0.1, p=-1.0, rho_target=4.42e-3, temp_p=-0.5, plot_all=False)

# all_r, all_vel_radial, all_vel_phi, v_phi_theoretical = get_velocities(
# 					snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/',
# 					ptype='PartType0', dr_factor=0.1, p=-1.0, rho_target=4.42e-3, temp_p=-0.5)

# all_r = np.array([1,2,3,4])
# all_vel_radial = np.array([1,2,3,4])
# all_vel_phi = np.array([1,2,3,4])
# v_phi_theoretical = np.array([3,4,5,6])

plt.figure()
plt.plot(all_r, all_vel_radial, marker='.', linestyle='None', label='Actual')
plt.axhline(y=0, linestyle='--',color='orange', label='Theor')
plt.xlabel('Radius', size=11)
plt.ylabel("Radial velocity", size=11)
# plt.xlim([0.2, 0.35])
plt.title("Damping out at boundaries")
plt.legend()
plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/test_boundary_cond_results/damping_boundaries_results/vel_radial_200.pdf')
# plt.show()

plt.figure()
plt.plot(all_r, all_vel_phi, marker='.', linestyle='None', label='Actual')
plt.plot(all_r, v_phi_theoretical, marker='.', linestyle='None', label='Theor')
plt.xlabel('Radius', size=11)
plt.ylabel("Azimuthal velocity", size=11)
# plt.xlim([0.2, 0.35])
plt.title("Damping out at boundaries")
plt.legend()
plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/test_boundary_cond_results/damping_boundaries_results/vel_phi_200.pdf')
# plt.show()