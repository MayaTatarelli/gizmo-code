import numpy as np
from makeIC import *


# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=1.5, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=2.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_rho.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=1.5, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_rho_rand.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=1.5, m_target_gas=1e-4)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_low_dr.hdf5', dr_factor=0.6, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=3.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep_uniform_high_rho_rand.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=3.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_cs_subkep_nonuniform_rho_1_5.hdf5', dr_factor=0.6, gamma=7/5, internal_energy=9.e-5, p=-1.5, r_in=0.2, r_out=2., rho_target=10.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_with_dust.hdf5', dr_factor=0.9, gamma=7/5, internal_energy=9.e-5, p=0.0, r_in=0.2, r_out=2., rho_target=2.0, m_target_gas=1e-4, include_dust=True)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=5.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_10.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=10.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_40.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=40.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_30.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=30.0, m_target_gas=1e-4)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_5.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=5e-5)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_4.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=8e-5)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_3.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=1e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_2.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=2e-4)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_rho_20_1.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=20.0, m_target_gas=4e-4)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=1.768e-8)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_res_1.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=1.0e-08)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_res_2.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=2.0e-08)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_res_3.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=3.536e-08)

#Testing moving inner boundary to 0.1 instead of 0.2
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_inner0_1.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.1, r_out=2., rho_target=8.38e-3, m_target_gas=1.768e-8)

#Testing varying particle mass
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_vary_prtcl_mass.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=1.768e-8, vary_particle_mass=True, num_particle_r_in=2000)

#Testing varying particle mass with larger outer radius
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_vary_prtcl_mass_large_r_out.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=4., rho_target=4.42e-3, m_target_gas=1.768e-8, vary_particle_mass=True, num_particle_r_in=2000)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_test_1.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=1.4144e-07)
# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_mass_0_01_test_2.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=4.42e-3, m_target_gas=7.072e-08)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_no_therm_vel.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=6.0, m_target_gas=1e-4)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_high_rho_temp_gradient.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=6.0, m_target_gas=1e-4)

# makeIC_keplerian_disk_2d(fname='./ICs/keplerian_ic_2d_rho_temp_gradient_fix_vphi.hdf5', dr_factor=0.1, gamma=7/5, internal_energy=9.e-5, p=-1.0, r_in=0.2, r_out=2., rho_target=5.0, m_target_gas=1e-4)

#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='./ICs/keplerian_disk_2d_with_vel_phil_coords_r_0.hdf5', p=0., r_in=0.1, r_out=2., rho_target=1.)
#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='keplerian_disk_2d_with_vel_r_1.hdf5', p=-1., r_in=0.1, r_out=2., rho_target=1.)
#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='keplerian_disk_2d_with_vel_r_3_2.hdf5', p=-3./2., r_in=0.1, r_out=2., rho_target=1.)

#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='keplerian_disk_2d_r_0.hdf5', p=0., r_in=0.1, r_out=2., rho_target=1.)
#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='keplerian_disk_2d_r_1.hdf5', p=-1., r_in=0.1, r_out=2., rho_target=1.)
#makeIC_keplerian_disk_2d(Nbase=1.0e4, fname='keplerian_disk_2d_r_3_2.hdf5', p=-3./2., r_in=0.1, r_out=2., rho_target=1.)

#makeIC_stratified()

#makeIC_box_uniform_gas(DIMS=2, N_1D=128, fname='gasgrain_2d_128_boxSize_10_unifmu.hdf5', BoxSize=10.)

#makeIC_box_uniform_gas(DIMS=3, N_1D=128, fname='gasgrain_3d_128_unifmu.hdf5', BoxSize=6.)

#For testing
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=3.5e5, Ngrains_Ngas=1, fname='tmp.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)

#for 128 at base
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=3.5e5, Ngrains_Ngas=1, fname='stratbox_proper_disk_3d_N128bottom_perturb.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)

#for 128*3 at base with x=y=10H
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=1.7e6, Ngrains_Ngas=1, fname='stratbox_proper_disk_3d_N384bottom_boxL_10.hdf5', Lbox_xy=10., Lbox_z=4., rho_target=1.)

#double check randomization of particles
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=3.5e5, Ngrains_Ngas=1, fname='stratbox_disk_3d_N128bottom_perturb_1.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=3.5e5, Ngrains_Ngas=1, fname='stratbox_disk_3d_N128bottom_perturb_2.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)

#for 64 at top
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=2.125e6, Ngrains_Ngas=1, fname='stratbox_disk_3d_N64top_perturb.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)

#dont use too many particles
#makeIC_disk_stratified_no_dust(DIMS=3, Nbase=5.0e6, Ngrains_Ngas=1, fname='stratbox_disk_3d_N128bottom_perturb.hdf5', Lbox_xy=6., Lbox_z=2., rho_target=1.)

#For z = 4 on top and 4 on bottom (z=8 total)
# makeIC_disk_stratified_no_dust(DIMS=3, Nbase=8e5, Ngrains_Ngas=1, fname='./ICs/stratbox_disk_3d_z8.hdf5', Lbox_xy=6., Lbox_z=8., rho_target=1.)

makeIC_disk_stratified_no_dust(DIMS=3, Nbase=2e6, Ngrains_Ngas=1, fname='./ICs/stratbox_disk_3d_z8_xy12.hdf5', Lbox_xy=12., Lbox_z=8., rho_target=1.)
