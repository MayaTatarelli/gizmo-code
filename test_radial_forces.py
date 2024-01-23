from global_disk import get_radial_force_balance, test_radial_gradient, test_radial_gradient_direct, calculate_radial_accel
import matplotlib.pyplot as plt
import numpy as np

#adding viscosity - test_2
# calculate_radial_accel(snum=70, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_2/', 
# 						use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


#keplerian_ic_2d_rho_temp_gradient_mass_0_01
# calculate_radial_accel(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/resolution_test/res_test_4/', 
# 						use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=20.0, gamma = 7./5.,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


#adding viscosity - test_3
# calculate_radial_accel(snum=189, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_3/', 
# 						use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

#adding viscosity - test_1
# snums = np.array([0,1,200,402,620,751])
# for i in range(len(snums)):
# 	calculate_radial_accel(snum=snums[i], sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_viscosity/test_1/', 
# 							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
# 							output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


#damping_boundaries - test0
snums = np.array([0,1,10,100,200])
for i in range(len(snums)):
	calculate_radial_accel(snum=snums[i], sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/damping_boundaries/test0/output/', 
							use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
							ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=4.42e-3, gamma = 7./5.,
							output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

# calculate_radial_accel(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/resolution_test/res_test_4/', 
# 						use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_4.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, rho_target=20.0, gamma = 7./5.,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

exit()
# test_radial_gradient_direct()

# test_radial_gradient()
# exit()
# get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_fix_vphi.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

# get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand.hdf5',
# 						ptype='PartType0', p=0.0, temp_p=0.0, to_plot='v_phi',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

# get_radial_force_balance(snum=200, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand/output/', 
# 						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
# 						ptype='PartType0', p=0.0, temp_p=0.0, to_plot='v_phi',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
#######################
#rho_20
# get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=20, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_30/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', rho_target=30.0, subplot_title='Snapshot 0',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()
get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_10.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 0',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
get_radial_force_balance(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 10',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
get_radial_force_balance(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_10/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 100',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

# get_radial_force_balance(snum=300, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_20/output/', 
# 						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 300',
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


exit()
#rho_40
get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_40.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 0',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
get_radial_force_balance(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 10',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
get_radial_force_balance(snum=100, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 100',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=300, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force_rho_40/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 300',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')


exit()
get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/adding_outer_radii_press_force/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 0',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()


get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='IC',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=0, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all', subplot_title='Snapshot 0',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()
get_radial_force_balance(snum=10, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=50, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5, to_plot='all',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
plt.show()
# get_radial_force_balance(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/', 
# 						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
# get_radial_force_balance(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fix_vphi/output/', 
# 						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
# 						ptype='PartType0', p=-1.0, temp_p=-0.5,
# 						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

exit()

get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient_fix_vphi.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()
get_radial_force_balance(use_fname=True, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()
get_radial_force_balance(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=50, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/fixed_IE_IEPred/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=-1.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()
#Fixed cs factor in inner pressure boundary condition -- CAREFUL  WHY IS p=0.0 ???
get_radial_force_balance(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=0.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
exit()

get_radial_force_balance(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=0.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=200, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_cs/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=0.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')
#Fixed temperature to not change
get_radial_force_balance(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=0.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')

get_radial_force_balance(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_2d_rho_temp_gradient_fixed_temp/output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', p=0.0, temp_p=-0.5,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/')