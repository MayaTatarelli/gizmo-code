from global_disk import *
import numpy as np
import matplotlib.pyplot as plt 


def plot_viscosity(snum=0, r_in=0.2, r_out=2.0, rho_target=4.42e-3, nu = 3.305e-8, runs=np.array(['./run/']), fig_dir=np.array(['./run/']), titles=np.array(['run'])):
	for i in range(len(runs)):

		all_r, num_particles_at_r, surf_density, rho_volume, all_temp, all_vel_radial = get_value_profile(snum=snum,
												sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/'+runs[i],
												val_to_plot='rho', ptype='PartType0', r_in=r_in, r_out=r_out,
												dr_factor=0.1, p=-1.0, rho_target=rho_target, temp_p=-0.5, plot_all=True)#load file
		c_s = all_temp**0.5
		H = c_s / all_r**(-3/2)

		alpha_exp = nu / (c_s*H)
		
		for j in range (len(alpha_exp)):
			if(alpha_exp[j] >= 6e-5):
				alpha_0 = r_in*alpha_exp[j]
				print(alpha_0)
				print(j)
				break

		print(alpha_exp[0:50])

		alpha_theor = alpha_0*all_r**(-1)

		plt.figure()
		plt.plot(all_r, alpha_exp, marker='.', linestyle='None', label="Simulation $\\alpha$")
		plt.plot(all_r, alpha_theor, label="Theoretical $\\alpha$")
		plt.xlabel('Radius', size=11)
		plt.ylabel("Viscosity param $\\alpha$", size=13)
		plt.xlim([r_in, r_out])
		plt.title(titles[i])
		plt.legend()
		plt.show()
		plt.savefig('/Users/mayatatarelli/Desktop/Maya_Masters/Research/Fall2023/test_boundary_cond_results_2/'+fig_dir[i]+'viscosity_'+str(snum)+'.pdf')

plot_viscosity(snum=500, r_in=0.2, r_out=4.0, rho_target=4.42e-3, nu = 3.305e-8, runs=np.array(['adding_viscosity/outer_both_low_visc_vary_prtcl_mass_large_r_out_r_in/output/']), fig_dir=np.array(['../test_vary_particle_mass/outer_both_low_visc_vary_prtcl_mass_large_r_out_r_in/']), titles=np.array(['Low Viscosity ($\\nu$ = 3.305e-8)']))