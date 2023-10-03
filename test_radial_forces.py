from global_disk import get_radial_force_balance, test_radial_gradient, test_radial_gradient_direct
import matplotlib.pyplot as plt

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