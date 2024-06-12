import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.interpolate as interpolate
from plot_snapshot import load_snap, check_if_filename_exists
from radprof_concentration import load_v
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab
import math

def plot_velocity_streamlines(snum=0, sdir='./output/', 
								use_fname=False, fname='./ICs/keplerian_ics.hdf5',
								plot_ghost=False,
								ptype='PartType0', phil=True, vmax=5e-3):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		print(fname)
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	PIDs = np.array(P['ParticleIDs'])
	xx = Pc[:, 0]#-2.
	yy = Pc[:, 1]#-2.
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	vz = np.array(P['Velocities'][:, 2])
	massP = np.array(P['Masses'])
	print(len(Pc))

	# exit()
	#TEMPORARY
	# ok = np.where(xx>2.95)
	# ok = np.where(xx[ok]<5.05)
	# radius = np.sqrt((xx-4.)**2 + (yy-4.)**2)
	# ok = np.where(radius<2.)
	# xx = xx[ok]; yy = yy[ok];
	# vx = vx[ok]; vy = vy[ok];

	if(use_fname == False):
		density = np.array(P['Density'])
		# density = density[ok]

	if(plot_ghost == True):
		ok = np.where(PIDs==0)
		Pc = Pc[ok]
		PIDs = PIDs[ok]
		xx = xx[ok]
		yy = yy[ok]
		vx = vx[ok]
		vy = vy[ok]
		vz = vz[ok]
		massP = massP[ok]
		density = density[ok]
		
	x = 1.0 * (xx / xx.max())
	y = 1.0 * (yy / yy.max())

	print(np.min(x), np.max(x))
	print(np.min(y), np.max(y))
	# exit()

	# For Interpolation
	# ngrid=2048
	# xg, yg = np.meshgrid(np.linspace(np.min(x), 1, ngrid), np.linspace(np.min(y), 1, ngrid))
	# vxgrid = interpolate.griddata((x, y), vx, (xg, yg), method='linear')
	# vygrid = interpolate.griddata((x, y), vy, (xg, yg), method='linear')

	#Plot velocity streamlines
	# xg, yg, vxgrid, vygrid = load_v(P_File, part='PartType0', xz=0, ngrid=2048,return_coords=True)
	# print (xg[2000][1500:1620]); print(yg[2000][1500:1620]);
	# print (vxgrid[2000][1500:1620]); print(vygrid[2000][1500:1620]);
	# exit()

	fig, ax = plt.subplots()

	# val = 1
	# if (phil==True):
	# 	print("Phil")
	# 	val = 6
	# 	ax.set_xlim([2,6])
	# 	ax.set_ylim([2,6])
	# else:
	# 	print("Maya")
	# 	val = 4
	# 	ax.set_xlim([0,4])
	# 	ax.set_ylim([0,4])

	cmap='Spectral_r' #'PuBu'
	# dg = interpolate.griddata((x, y), density, (xg, yg), method='linear')#, fill_value=np.median(density));
	# im = ax.imshow(dg, interpolation='bicubic', cmap=cmap, extent=(np.min(x), 1, np.min(y), 1,), zorder=1);

	if(plot_ghost == False):
		ok = np.where(PIDs != 0)
	else:
		ok = np.where(PIDs == 0)

	im =ax.scatter(xx[ok], yy[ok], marker='.', vmin=0., vmax=vmax, c=density[ok], cmap=cmap, zorder=3)
	# im =ax.scatter(xx, yy, marker='.', vmin=0., c=massP, cmap=cmap, zorder=3)

	val = 10
	# ax.streamplot(xg, yg, vxgrid, vygrid,linewidth=1.0, density = 4., zorder=3)
	# ax.plot(x,y, marker = '.', markersize=1, linestyle='None')

	# ax.streamplot((xg*val), (yg*val), vxgrid, vygrid,linewidth=1.0, density = 4., zorder=3)
	# ax.plot(xx,yy, marker = '.', markersize=1, linestyle='None')
	# plt.show()
	# exit()

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = pylab.colorbar(im, cax=cax)
	cbar.set_label(label="$\\Sigma_g$", size=16, rotation=270, labelpad=20)
	cbar.ax.tick_params(labelsize=10)
	ax.set_yticks(np.arange(1, 10, 1))
	ax.set_xticks(np.arange(1, 10, 1))
	# ax.set_xlabel("x")
	# ax.set_ylabel("y")
	ax.set_title("Snapshot "+str(snum))
	ax.set_aspect('equal', adjustable='box')
	# plt.savefig('/Users/mayatatarelli/Desktop/disk_image_'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
	plt.show()

def plot_gas_density(snum=0, sdir='./output/', 
								use_fname=False, fname='./ICs/keplerian_ics.hdf5',
								ptype='PartType0', phil=True):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0]
	y = Pc[:, 1]
	rho = np.array(P['Density'])
	mass = np.array(P['Masses'])

	print(len(x))

def plot_value_profile(snum=0, sdir='./output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						val_to_plot='rho', ptype='PartType0',
						dr_factor=0.1, p=0.0, rho_target=1.0, temp_p=-0.5, H_factor=1.0,
						plot_all=False,
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/'):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - 2.0
	y = Pc[:, 1] - 2.0
	massP = P['Masses'][0]
	if (use_fname == True):
		densitiesP = x*0.0
	else:
		densitiesP = P['Density']
	internal_energyP = P['InternalEnergy']
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	v_radial = (x*vx + y*vy)/np.sqrt(x**2 + y**2)
	r = np.sqrt(x*x + y*y)

	dr_factor=dr_factor
	# internal_energy=9e-5
	gamma=7/5
	#c_s = np.sqrt((gamma-1)*internal_energy)

	#Some initial calculations --ONLY INCLUDE ONCE USING TEMP GRADIENT
	r_ref = 1.
	c_s_ref = 0.05
	internal_energy_ref = c_s_ref**2/(gamma-1.)
	T_ref = c_s_ref**2
	#Here, the reference point is r = 1
	T_0 = T_ref*r_ref**(-temp_p)

	num_particles_at_r = np.zeros(0)
	all_r = np.zeros(0)
	all_dr = np.zeros(0)
	all_densities = np.zeros(0)
	all_temp = np.zeros(0)
	all_vel_radial = np.zeros(0)

	# r_in = np.min(r); r_out = np.max(r);
	r_in = 0.2; r_out = 2.0;

	print("r_in: ", r_in); print("r_out: ", r_out);
	# exit()
	r_cur=r_in; iter=0; dr=0;
	while(r_cur <= r_out):
		T_cur = T_0*r_cur**temp_p
		c_s = T_cur**0.5
		if(temp_p==0.0):
			c_s = np.sqrt((gamma-1)*9e-5)
		dr = dr_factor*c_s/r_cur**(-3./2.)
		ok = np.where((r>=r_cur) & (r<r_cur+dr))
		numP_cur = len(ok[0])
		# print(numP_cur)
		num_particles_at_r = np.append(num_particles_at_r, numP_cur)

		#Temperature
		internal_energy_cur = np.sum(internal_energyP[ok])/len(internal_energyP[ok])
		temp_cur = (gamma-1.)*internal_energy_cur

		#Density
		density_cur = np.sum(densitiesP[ok])/len(densitiesP[ok])
		# density_cur = density_cur * temp_cur**0.5 * r_cur**(3/2) #Just a test for density -- not good

		#Radial velocity
		vel_radial_cur = np.sum(v_radial[ok])/len(v_radial[ok])

		all_r = np.append(all_r, r_cur)
		all_dr = np.append(all_dr, dr)
		all_densities = np.append(all_densities, density_cur)
		all_temp = np.append(all_temp, temp_cur)
		all_vel_radial = np.append(all_vel_radial, vel_radial_cur)
		print(r_cur)
		r_cur+=dr; iter+=1;

	if(val_to_plot=='rho' or plot_all==True):
		density = num_particles_at_r*massP/(2*np.pi*all_r*all_dr)
		# rho_0 = 0.6
		# rho_0 = density[0]/r_in**p
		rho_0 = rho_target/r_in**p
		print(rho_0)
		density_profile_theor = rho_0*all_r**p

		#real_density_ok = np.where(all_densities<15.0)

		plt.figure()
		plt.plot(all_r, density, marker='.', linestyle='None', label='Azimuthally averaged density')
		plt.plot(all_r, all_densities, marker='.', linestyle='None', label='Azimuthally averaged code density')
		plt.plot(all_r, density_profile_theor, linewidth=2, label='Theoretical density')
		plt.xlabel('r', size=13)
		plt.ylabel('$\\Sigma_g$', size=13)
		plt.title('Radial Surface Density Profile of Disk')
		plt.legend()

		typename='density'
		plt.savefig(output_plot_dir+typename+'_snap'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0)

	if(val_to_plot=='temp' or plot_all==True):
		temp_profile_theor = T_0*all_r**temp_p
		plt.figure()
		plt.plot(all_r, all_temp, marker='.', linestyle='None', label='Azimuthally averaged temp')
		plt.plot(all_r, temp_profile_theor, linewidth=2, label='Theoretical temp')
		plt.xlabel('r', size=13)
		plt.ylabel('Temp', size=13)
		plt.title('Radial Temperature Profile of Disk')
		plt.legend()
		typename='temp'
		plt.savefig(output_plot_dir+typename+'_snap'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
		

	if(val_to_plot=='vel_radial' or plot_all==True):
		boundary = r_in+H_factor*(0.05*r_in**(5./4.))
		plt.figure()
		plt.plot(all_r, all_vel_radial, marker='.', linestyle='None', label='Azimuthally averaged radial velocity')
		plt.axhline(y=0, color='k', linestyle='--')
		if (use_fname == False):
			boundary = r_in+H_factor*(0.05*r_in**(5./4.))
			plt.axvline(x=boundary, color='k', linestyle='--')
		plt.xlabel('r', size=13)
		plt.ylabel('$V_{radial}$', size=13)
		plt.title('Radial Velocity Component')
		plt.legend()
		typename='velradial'
		plt.savefig(output_plot_dir+typename+'_snap'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
		

	# plt.savefig('/Users/mayascomputer/Desktop/Maya_Masters/Research/Group_Presentation/2D_global_disk_plots/radial_'+typename+'_'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
	# plt.show()

#NOT fully fixed to take r_out input
def get_radial_force_balance(snum=0, sdir='./output/', 
						use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
						ptype='PartType0', r_in=0.2, r_out=2.0, p=0.0, temp_p=-0.5, to_plot='all', rho_target=5.0, subplot_title='test',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/'):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - r_out
	y = Pc[:, 1] - r_out
	if (use_fname == True): #not using rho_volume from this right now
		all_r, num_P_at_r, density, rho_volume, density_theoretical = get_value_profile(snum=0, sdir='./output/', 
						use_fname=True, fname=fname,
						val_to_plot='rho', ptype='PartType0', r_in=r_in, r_out=r_out,
						p=p, rho_target=rho_target, temp_p=temp_p,
						plot_all=False)
		# print(np.sum(num_P_at_r))
		# print(len(x))
		# print(np.sum(num_P_at_r))
		# print(len(Pc))
		# exit()
		densitiesP = np.zeros(0)
		for i in range(len(density)):
			density_at_r = np.zeros(int(num_P_at_r[i])) + density[i]
			# print(density_at_r)
			# print(density[i])
			# print(len(density_at_r))
			# print(num_P_at_r[i], "\n")
			densitiesP = np.append(densitiesP, density_at_r)
		# print(len(densitiesP))
	else:
		densitiesP = P['Density'][:]

	internal_energyP = P['InternalEnergy'][:]
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	v_radial = (x*vx + y*vy)/np.sqrt(x**2 + y**2) #This is actually r_dot, which it the same as radial velocity
	v_phi = (x*vy - y*vx) / (x**2 + y**2) #This is actually phi_dot
	r = np.sqrt(x*x + y*y)
	phi = get_phi(x,y)
	gamma=7./5.

	#Need pressure gradient term (density, cs, and omega)
	#pressure value at each particle:
	# pressure = (gamma-1) * densitiesP * internal_energyP * v_phi * r**0.5 #Old
	pressure = np.sqrt((gamma-1) * internal_energyP) * densitiesP  * v_phi #* r**(-1)
	#interpolate pressure to grid:
	ngrid=1024
	#Maybe change to min and max from actual particles
	rg, phig = np.meshgrid(np.linspace(r_in,2,ngrid), np.linspace(0,2*np.pi,ngrid))
	Pressure_grid = interpolate.griddata((r, phi), pressure, (rg, phig), method='cubic')
	# print(Pressure_grid)
	Density_grid = interpolate.griddata((r, phi), densitiesP, (rg, phig), method='cubic')
	internal_energyP_grid = interpolate.griddata((r, phi), internal_energyP, (rg, phig), method='cubic')
	rho_volume_grid =  Density_grid * rg**(-3/2) / np.sqrt((gamma-1) * internal_energyP_grid)

	delta_r = rg[0][1] - rg[0][0]
	delta_phi = phig[1][0] - phig[0][0]
	dPdphi, dPdr = np.gradient(Pressure_grid, delta_phi, delta_r)
	print(dPdr)
	press_grad_accel = - dPdr/rho_volume_grid
	print(press_grad_accel)
	#Old calculation of pressure gradient
	####################################################
	# xg, yg = np.meshgrid(np.linspace(np.min(x), np.max(x), ngrid), np.linspace(np.min(y), np.max(y), ngrid))
	# Pressure_grid = interpolate.griddata((x, y), pressure, (xg, yg), method='linear')
	# Density_grid = interpolate.griddata((x, y), densitiesP, (xg, yg), method='linear')
	# r_grid = np.sqrt(xg*xg + yg*yg)
	
	# dPdx, dPdy = np.gradient(Pressure_grid)
	# dPdr = dPdx * (r_grid/xg) + dPdy * (r_grid/yg)
	# press_grad_accel = - dPdr/Density_grid
	####################################################
	# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	# ax.scatter(phi_grid, r_grid, s=1.)
	# ax.set_rmax(2)
	# ax.set_rticks([0.1,0.5, 1, 1.5, 2])  # Less radial ticks
	# ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
	# ax.grid(True)
	# plt.show()

	#Need centrifugal (azimuthal and radial velocities)
	centrifugal_accel_i = v_phi*v_phi*r #New
	centrifugal_accel = interpolate.griddata((r, phi), centrifugal_accel_i, (rg, phig), method='cubic')
	#Temporarily removing
	# v_phi_grid = interpolate.griddata((r, phi), v_phi, (rg, phig), method='linear')
	# centrifugal_accel = v_phi_grid*v_phi_grid*rg #because here v_phi is really phi_dot

	v_phi_kep_grid = 1 / rg**0.5
	centrifugal_accel_kep = v_phi_kep_grid*v_phi_kep_grid*rg

	# #####TESTING#####
	# sum=0
	# for i in range(len(all_r)):
	# 	ok = np.where(r==all_r[i])
	# 	print(len(ok[0]))
	# 	sum+=len(ok[0])
	# 	# if(i==10):
	# 	# 	exit()
	# print(sum)
	# exit()
	# plt.figure()
	# plt.plot()
	# plt.show()
	#####END TESTING#####

	#Need gravity
	grav_accel = - 1/(rg*rg)
	# grav_accel = - 1/(r_grid*r_grid) #Old

	all_radius = rg[0]
	print("############################################")
	print("all radius: ", all_radius)
	all_centrifugal_accel = np.zeros(0)
	all_grav_accel = np.zeros(0)
	all_press_grad_accel = np.zeros(0)

	for i in range(len(rg[0])):
		# total_accel_cur = total_accel[:,i]
		# total_accel_cur = total_accel_cur[~np.isnan(total_accel_cur)]
		# all_total_accel = np.append(all_total_accel, np.sum(total_accel_cur)/len(total_accel_cur))
		print("Current radius: ", rg[:,i])
		#Centrifugal
		centrifugal_accel_cur = centrifugal_accel[:,i]
		centrifugal_accel_cur = centrifugal_accel_cur[~np.isnan(centrifugal_accel_cur)]
		all_centrifugal_accel = np.append(all_centrifugal_accel, np.sum(centrifugal_accel_cur)/len(centrifugal_accel_cur))

		#Gravity
		grav_accel_cur = grav_accel[:,i]
		grav_accel_cur = grav_accel_cur[~np.isnan(grav_accel_cur)]
		all_grav_accel = np.append(all_grav_accel, np.sum(grav_accel_cur)/len(grav_accel_cur))

		#Pressure gradient
		press_grad_accel_cur = press_grad_accel[:,i]
		press_grad_accel_cur = press_grad_accel_cur[~np.isnan(press_grad_accel_cur)]
		all_press_grad_accel = np.append(all_press_grad_accel, np.sum(press_grad_accel_cur)/len(press_grad_accel_cur))

	all_total_accel = all_centrifugal_accel + all_grav_accel + all_press_grad_accel

	ok = np.where(all_radius>0.0)

	if(to_plot=='all'):
		print("Doing net accel")
		accel_to_plot = all_total_accel
		plot_title = 'Radial Net Acceleration'
		y_label = '$a_{r,net}$'
	if(to_plot=='centrifugal'):
		print("Doing centrifugal accel")
		accel_to_plot = all_centrifugal_accel
		plot_title = 'Radial Centrifugal Acceleration'
		y_label = '$a_{r,cent}$'
	if(to_plot=='gravity'):
		print("Doing gravity accel")
		accel_to_plot = all_grav_accel
		plot_title = 'Radial Gravitational Acceleration'
		y_label = '$a_{r,grav}$'
	if(to_plot=='pressure'):
		print("Doing pressure accel")
		accel_to_plot = all_press_grad_accel
		plot_title = 'Radial Pressure Gradient Acceleration'
		y_label = '$a_{r,press}$'
	# if(to_plot=='v_phi'):
	# 	print("Doing azimuthal velocity")
	# 	total_accel = v_phi_grid
	# 	plot_title = 'Azimuthal Velocity'
	# 	y_label = '$V_{\\phi}$'

	# neg = np.where(all_total_accel<0)
	# pos = np.where(all_total_accel>0)
	# print(neg)
	# print(pos)
	print(all_total_accel[ok][0:2])
	print(all_radius[ok][0:2])

	print("radius", all_radius[ok][0:10])
	print("Pressure gradient accel", all_press_grad_accel[ok][0:10])
	print("Centrifugal accel", all_centrifugal_accel[ok][0:10])
	print(all_grav_accel[ok][0:10])

	theoretical_press_grad_accel = -(0.05**2) * (-11/4) * (all_radius[ok])**-1.5

	theoretical_centrifugal_accel = -all_grav_accel[ok] - theoretical_press_grad_accel
	# v_phi_theoretical = np.sqrt((all_radius[ok]**(-3/2))**2 * all_radius[ok]**2 + (-1.0-7/4)*0.0025/all_radius[ok]**0.5)
	# theoretical_centrifugal_accel = v_phi_theoretical*v_phi_theoretical/all_radius[ok]

	fig, axs = plt.subplots(3)
	fig.suptitle(subplot_title)
	axs[0].plot(all_radius[ok], all_press_grad_accel[ok]-theoretical_press_grad_accel, label='Pressure grad accel (Actual-Theoretical)')
	axs[1].plot(all_radius[ok], all_centrifugal_accel[ok]-theoretical_centrifugal_accel, label='Centrifugal accel (Actual-Theoretical)')
	axs[2].plot(all_radius[ok], accel_to_plot[ok], label="Azimuthally averaged net accel")
	
	axs[0].axhline(y=0, color='k', linestyle='--')
	axs[1].axhline(y=0, color='k', linestyle='--')
	axs[2].axhline(y=0, color='k', linestyle='--')

	plt.xlabel('r', size=13)
	axs[0].set_ylabel('$a_{r,press}$', size=12)
	axs[1].set_ylabel('$a_{r,cent}$', size=12)
	axs[2].set_ylabel('$a_{r,net}$', size=12)

	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	plt.show()
	
	plt.figure()
	plt.plot(all_radius[ok], accel_to_plot[ok], marker='.', label="Azimuthally averaged net accel")
	# plt.plot(all_radius[ok], theoretical_press_grad_accel, label="Theoretical pressure grad acceleration")
	plt.plot(all_radius[ok], all_press_grad_accel[ok]-theoretical_press_grad_accel, label='Pressure grad accel (Actual-Theoretical)')
	plt.plot(all_radius[ok], all_centrifugal_accel[ok]-theoretical_centrifugal_accel, label='Centrifugal accel (Actual-Theoretical)')
	plt.axhline(y=0, color='k', linestyle='--')
	plt.xlabel('r', size=13)
	plt.ylabel(y_label, size=13)
	plt.title(plot_title+' '+subplot_title)
	plt.legend()
	plt.show()

	# all_total_accel_by_radius = np.zeros(0)
	# all_radius = np.zeros(0)
	# r_in = 0.2; r_out=2.0; dr = 3e-3; r_cur=r_in; iter=0;
	# while(r_cur <= r_out):
	# 	ok = np.where((r_grid>=r_cur) & (r_grid<r_cur+dr))
	# 	total_accel_cur = np.sum(total_accel[ok]) / len(total_accel[ok])

	# 	all_total_accel_by_radius = np.append(all_total_accel_by_radius, total_accel_cur)
	# 	all_radius = np.append(all_radius, r_cur)
	# 	r_cur+=dr; iter+=1;
	# print(iter)

	# plt.figure()
	# plt.plot(all_radius, all_total_accel_by_radius, marker='.', linestyle='None', label="Azimuthally averaged")
	# plt.axhline(y=0, color='k', linestyle='--')
	# plt.xlabel('r', size=13)
	# plt.ylabel('$a_r$', size=13)
	# plt.title('Radial Acceleration')
	# plt.legend()
	# plt.show()

	# fig, ax = plt.subplots()
	# cmap='hot'
	# im = ax.imshow(total_accel, interpolation='bicubic', cmap=cmap, extent=(np.min(x), np.max(x), np.min(y), np.max(y),), zorder=1);

	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# cbar = pylab.colorbar(im, cax=cax)
	# cbar.set_label(label="Radial Accel", size=16, rotation=270, labelpad=14)
	# cbar.ax.tick_params(labelsize=10)
	# ax.set_xlabel("x")
	# ax.set_ylabel("y")
	# # plt.savefig('/Users/mayascomputer/Desktop/Maya_Masters/Research/Group_Presentation/2D_global_disk_plots/disk_uniform_density_'+str(snum)+'.pdf', dpi=150, bbox_inches='tight', pad_inches=0)
	# plt.show()

def calculate_radial_accel(snum=0, sdir='./output/', use_fname=False, fname='./ICs/keplerian_ic_2d_rho_temp_gradient.hdf5',
						ptype='PartType0', r_in=0.2, r_out=2.0, width_ghost_in=0.0, width_ghost_out=0.0, p=-1.0, temp_p=-0.5, rho_target=20.0, gamma = 7./5.,
						plot_title='test',
						output_plot_dir='/Users/mayascomputer/Codes/gizmo_code/images_plots/keplerian_disk_tests/'):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - r_out - width_ghost_out
	y = Pc[:, 1] - r_out - width_ghost_out
	internal_energyP = P['InternalEnergy'][:]
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	r_dot = (x*vx + y*vy)/np.sqrt(x**2 + y**2) #This is actually r_dot, which it the same as radial velocity
	phi_dot = (x*vy - y*vx) / (x**2 + y**2) #This is phi_dot
	r = np.sqrt(x*x + y*y)
	phi = get_phi(x,y)
	gamma=gamma

	if (use_fname == True):
		all_r, num_P_at_r, density, rho_volume, density_theoretical = get_value_profile(use_fname=True, fname=fname,
																		val_to_plot='rho', ptype='PartType0', r_in=r_in, r_out=r_out,
																		width_ghost_in=width_ghost_in, width_ghost_out=width_ghost_out,
																		p=p, rho_target=rho_target, temp_p=temp_p,
																		plot_all=False)
		densitiesP = np.zeros(0)
		for i in range(len(density)):
			density_at_r = np.zeros(int(num_P_at_r[i])) + density[i]
			# print(density_at_r)
			# print(density[i])
			# print(len(density_at_r))
			# print(num_P_at_r[i], "\n")
			densitiesP = np.append(densitiesP, density_at_r)
		# print(len(densitiesP))
	else:
		densitiesP = P['Density'][:]

	#Calculate the pressure for all particles -- later calculate gradient to get accel due to pressure gradient
	pressure = np.sqrt((gamma-1) * internal_energyP) * densitiesP  * phi_dot

	#Calculate gravitational acceleration at all particles
	grav_accel = - 1/(r*r)

	#Calculate centrifugal acceleration at all particles
	centrifugal_accel = phi_dot*phi_dot*r

	#Need to azimuthally average over all particles at a given radius -- determine which radius the particles are at
	#only using all_r and num_particles_at_r for now:
	# all_r, num_particles_at_r, density, rho_volume, all_temp, all_vel_radial = get_value_profile(snum=snum, sdir=sdir, 
	# 																			use_fname=use_fname, fname=fname,
	# 																			val_to_plot='rho', ptype=ptype,
	# 																			dr_factor=0.1, p=p, rho_target=rho_target, 
	# 																			temp_p=temp_p, plot_all=True)

	#Some initial calculations --ONLY INCLUDE ONCE USING TEMP GRADIENT
	r_ref = 1.
	c_s_ref = 0.05
	internal_energy_ref = c_s_ref**2/(gamma-1.)
	T_ref = c_s_ref**2
	#Here, the reference point is r = 1
	T_0 = T_ref*r_ref**(-temp_p)
	r_in = r_in; r_out = r_out + 0.001;
	mult = 1e8
	dr_factor = 0.1

	pressure_by_r = np.zeros(0)
	grav_accel_by_r = np.zeros(0)
	centrifugal_accel_by_r = np.zeros(0)
	all_r = np.zeros(0)

	r_cur=1.0*r_in-width_ghost_in; iter=0; dr=0;
	while(r_cur <= r_out+width_ghost_out):
		T_cur = T_0*r_cur**temp_p
		c_s = T_cur**0.5
		if(temp_p==0.0):
			c_s = np.sqrt((gamma-1)*9e-5)
		dr = dr_factor*c_s/(r_cur**(-3./2.))
		r_next = math.floor((r_cur+dr)*mult)/mult
		ok = np.where((r>=r_cur) & (r<(r_cur+dr)))

		#Pressure
		pressure_cur = np.sum(pressure[ok])/len(pressure[ok])

		#Gravitational
		grav_accel_cur = np.sum(grav_accel[ok])/len(grav_accel[ok])

		#Centrifugal
		centrifugal_accel_cur = np.sum(centrifugal_accel[ok])/len(centrifugal_accel[ok])

		pressure_by_r = np.append(pressure_by_r, pressure_cur)
		grav_accel_by_r = np.append(grav_accel_by_r, grav_accel_cur)
		centrifugal_accel_by_r = np.append(centrifugal_accel_by_r, centrifugal_accel_cur)
		all_r = np.append(all_r, r_cur)
		r_cur+=dr; iter+=1;

	#Calculate pressure gradient using diffP/diffr
	pressure_grad_accel_by_r = np.gradient(pressure_by_r, all_r)

	#Sum three accels
	net_radial_accel = pressure_grad_accel_by_r + grav_accel_by_r + centrifugal_accel_by_r

	#Theoretical accelerations
	theoretical_press_grad_accel = -(0.05**2) * (-11/4) * (all_r)**-1.5
	theoretical_centrifugal_accel = -grav_accel_by_r - theoretical_press_grad_accel

	#Plot individual accelerations vs r
	if (use_fname == True):
		subplot_title = plot_title + " (IC)"
	else:
		subplot_title = plot_title + " (Snapshot " + str(snum) + ")"

	fig, axs = plt.subplots(3)
	fig.suptitle(subplot_title)
	axs[0].plot(all_r, pressure_grad_accel_by_r-theoretical_press_grad_accel, label='Pressure grad accel (Actual-Theoretical)')
	axs[1].plot(all_r, centrifugal_accel_by_r-theoretical_centrifugal_accel, label='Centrifugal accel (Actual-Theoretical)')
	axs[2].plot(all_r, net_radial_accel, label="Azimuthally averaged net accel")
	
	axs[0].axhline(y=0, color='k', linestyle='--')
	axs[1].axhline(y=0, color='k', linestyle='--')
	axs[2].axhline(y=0, color='k', linestyle='--')

	plt.xlabel('r', size=13)
	axs[0].set_ylabel('$a_{r,press}$', size=12)
	axs[1].set_ylabel('$a_{r,cent}$', size=12)
	axs[2].set_ylabel('$a_{r,net}$', size=12)

	# axs[0].set_xlim([0.2,0.25])
	# axs[1].set_xlim([0.2,0.25])
	# axs[2].set_xlim([0.2,0.25])

	# axs[0].set_ylim([-10,1])
	# axs[1].set_ylim([-10,1])
	# axs[2].set_ylim([-15,1])

	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	plt.savefig(output_plot_dir)
	# plt.show()

	# #Plot net accel vs r
	# plt.figure()
	# plt.plot(all_r, net_radial_accel)
	# plt.axhline(y=0, linestyle='--')
	# plt.xlabel('Radius', size=11)
	# plt.ylabel('$a_r$', size=11)
	# plt.show()

def calculate_radial_vel(snum=0, sdir='./output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						ptype='PartType0', vel_to_plot='radial'):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - 2.0
	y = Pc[:, 1] - 2.0
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])

	if(vel_to_plot=='radial'):
		v_to_plot = (x*vx + y*vy)/np.sqrt(x**2 + y**2)
		v_label = "$v_r$"
	elif(vel_to_plot=='phi'):
		v_to_plot = (x*vy - y*vx) / (x**2 + y**2)
		v_label = "$v_{\\Phi}$"
	else:
		print("Not a valid velocity type input!")
		exit()

	print(v_to_plot[0:500])
	print(v_to_plot[90000:90500])

	ngrid=4096
	xg, yg = np.meshgrid(np.linspace(np.min(x), np.max(x), ngrid), np.linspace(np.min(y), np.max(y), ngrid))
	
	fig, ax = plt.subplots()
	cmap='plasma'

	dg = interpolate.griddata((x, y), v_to_plot, (xg, yg), method='linear')#, fill_value=np.median(density));
	im = ax.imshow(dg, interpolation='bicubic', cmap=cmap, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), zorder=1);
	# im =ax.scatter(x, y, marker='.', c=v_to_plot, cmap=cmap, zorder=3)

	print('v_r max: ', np.max(v_to_plot))
	print('v_r min: ', np.min(v_to_plot))

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cbar = pylab.colorbar(im, cax=cax)
	cbar.set_label(label=v_label, size=13, rotation=90, labelpad=10)
	cbar.ax.tick_params(labelsize=10)
	plt.show()

def test_radial_gradient():
	ngrid=1024
	x = np.linspace(-2,2,ngrid)
	y = np.linspace(-2,2,ngrid)
	r = np.sqrt(x*x + y*y)
	# phi = get_phi(x,y)
	# dfdr_exact = -1. / (r*r)

	xg, yg = np.meshgrid(x, y)
	r_grid = np.sqrt(xg*xg + yg*yg)
	phi_grid = get_phi(xg,yg)

	f = 1. / np.sqrt(xg*xg + yg*yg) #1
	# f = xg*xg + yg*yg #2
	# f = 1. / (xg*xg + yg*yg) #3


	delta_x = xg[0][1] - xg[0][0]
	delta_y = yg[1][0] - yg[0][0]
	dfdx, dfdy = np.gradient(f,delta_x,delta_y)
	dfdr_calc_grid = dfdx * (r_grid/xg) + dfdy * (r_grid/yg) #Original
	# dfdr_calc_grid = np.sqrt(dfdx[0]**2 + dfdx[1]**2) * (r_grid/xg) + np.sqrt(dfdy[0]**2 + dfdy[1]**2) * (r_grid/yg) #Test
	dfdr_exact_grid = -1. / (xg*xg + yg*yg) #1
	# dfdr_exact_grid = 2.*np.sqrt(xg*xg + yg*yg) #2
	# dfdr_exact_grid = -2. / np.sqrt(xg*xg + yg*yg)**3 #3

	# ALL FOR TESTING #######################
	dfdx_exact_grid = - xg * (xg**2 + yg**2)**(-3/2)
	dfdy_exact_grid = - yg * (xg**2 + yg**2)**(-3/2)
	
	avg_err_dx = np.sum((dfdx_exact_grid.flatten() - dfdx.flatten())**2)/len(dfdx.flatten())
	avg_err_dy = np.sum((dfdy_exact_grid.flatten() - dfdy.flatten())**2)/len(dfdy.flatten())
	avg_err_dr = np.sum((dfdr_exact_grid.flatten() - dfdr_calc_grid.flatten())**2)/len(dfdr_calc_grid.flatten())
	print(avg_err_dx)
	print(avg_err_dy)
	print(avg_err_dr)
	# exit()
	print("dfdx: \n")
	print(dfdx, "\n")
	print(dfdx_exact_grid, "\n")
	print("dfdy: \n")
	print(dfdy, "\n")
	print(dfdy_exact_grid, "\n")
	# print(f)
	# exit()
	# all_dfdr_calc = dfdr_calc_grid.flatten()
	# all_radius = r_grid.flatten()
	# print(np.unique(r))
	# print(len(np.unique(r)))
	#########################################

	all_dfdr_calc = np.zeros(0)
	all_dfdr_exact = np.zeros(0)
	all_radius = np.zeros(0)
	r_in = np.min(r); r_out=np.max(r); dr = 8e-3; r_cur=r_in; iter=0;
	print(r_in, r_out)
	while(r_cur <= r_out):
		ok = np.where((r_grid.flatten()>=r_cur) & (r_grid.flatten()<r_cur+dr))
		dfdr_calc_cur = np.sum(dfdr_calc_grid.flatten()[ok]) / len(dfdr_calc_grid.flatten()[ok])
		dfdr_exact_cur = np.sum(dfdr_exact_grid.flatten()[ok]) / len(dfdr_exact_grid.flatten()[ok])

		all_dfdr_calc = np.append(all_dfdr_calc, dfdr_calc_cur)
		all_dfdr_exact = np.append(all_dfdr_exact, dfdr_exact_cur)
		all_radius = np.append(all_radius, r_cur)
		r_cur+=dr; iter+=1;
	print(iter)
	# exit()

	dfdr_exact = -1. / (all_radius*all_radius) #1
	# dfdr_exact = 2.*all_radius #2
	# dfdr_exact = -2. / np.sqrt(all_radius*all_radius)**3 #3
	plt.figure()
	plt.plot(all_radius, all_dfdr_exact, marker='.', linestyle='None', label="Theoretical radial gradient from grid")
	plt.plot(all_radius, dfdr_exact, label="Theoretical radial gradient")
	plt.plot(all_radius, all_dfdr_calc, marker='.', linestyle='None', label="Calculated radial gradient")
	plt.xlabel('r', size=13)
	plt.ylabel('$df/dr$', size=13)
	plt.title('Testing Radial Gradient')
	plt.legend()
	plt.show()

def test_radial_gradient_direct():
	ngrid=1024
	rg, phig = np.meshgrid(np.linspace(0.2,2,ngrid), np.linspace(0,2*np.pi,ngrid))

	fpolar = 1. / rg
	dfpolardr_exact_grid = -1. / rg**2
	dfpolardr_exact = -1. / np.linspace(0.2,2,ngrid)**2

	delta_r = rg[0][1] - rg[0][0]
	delta_phi = phig[1][0] - phig[0][0]

	dfdphi, dfdr = np.gradient(fpolar, delta_phi, delta_r)
	print(dfdphi)
	diff = dfpolardr_exact_grid - dfdr
	print(diff)

	all_dfdr_calc = np.zeros(0)
	all_dfdr_exact = np.zeros(0)
	all_radius = rg[0]

	for i in range(len(rg[0])):
		all_dfdr_calc = np.append(all_dfdr_calc, np.sum(dfdr[:,i])/len(dfdr[:,i]))
		all_dfdr_exact = np.append(all_dfdr_exact, np.sum(dfpolardr_exact_grid[:,i])/len(dfpolardr_exact_grid[:,i]))

	plt.figure()
	# plt.plot(all_radius, all_dfdr_exact, marker='.', linestyle='None', label="Theoretical radial gradient from grid")
	plt.plot(all_radius, all_dfdr_calc, marker='.', linestyle='None', label="Calculated radial gradient from grid")
	plt.plot(all_radius, dfpolardr_exact, label="Theoretical radial gradient")
	plt.xlabel('r', size=13)
	plt.ylabel('$df/dr$', size=13)
	plt.title('Testing Radial Gradient')
	plt.legend()
	plt.show()

	# x = np.linspace(0.5,2,ngrid)
	# y = np.linspace(0.5,2,ngrid)
	# r = np.sqrt(x*x + y*y)
	
	# xg, yg = np.meshgrid(x, y)
	# r_grid = np.sqrt(xg*xg + yg*yg)

	# phi_grid = np.arctan2(yg,xg)
	# ok = np.where(phi_grid<0)
	# phi_grid[ok] = phi_grid[ok]+2*np.pi

	# print(r_grid)
	# print(phi_grid)

	# Test Functions
	# f = 1. / np.sqrt(xg*xg + yg*yg) #1
	# grad_r = np.gradient(r_grid)
	# print(grad_r)
	# delta_r = r_grid[0][1] - r_grid[0][0]
	# delta_phi = yg[1][0] - yg[0][0]
	# dfdr, dfdphi = np.gradient(f,r_grid[0], phi_grid[0])

########################################################################################################################
# HELPER FUNCTIONS:
def get_phi(x,y):
	phi = np.arctan2(y,x)
	ok = np.where(phi<0)
	phi[ok] = phi[ok]+2*np.pi
	return phi

def get_value_profile(snum=0, sdir='./output/', 
						use_fname=False, fname='./ICs/keplerian_ics.hdf5',
						val_to_plot='rho', ptype='PartType0', r_in=0.2, r_out=2.0, width_ghost_in=0.0, width_ghost_out=0.0,
						dr_factor=0.1, p=0.0, rho_target=1.0, temp_p=-0.5, plot_all=False):
	if (use_fname == True):
		print("Using filename for initial conditions file\n")
		P_File = h5py.File(fname,'r')
	else:
		print("Using snapshot file\n")
		P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - r_out - width_ghost_out
	y = Pc[:, 1] - r_out - width_ghost_out
	massP = P['Masses'][0]
	massP_array = P['Masses']

	if (use_fname == True):
		densitiesP = x*0.0
	else:
		densitiesP = P['Density']
	internal_energyP = P['InternalEnergy']
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	v_radial = (x*vx + y*vy)/np.sqrt(x**2 + y**2)
	phi_dot = ((x*vy) - (y*vx)) / (x**2 + y**2)
	r = np.sqrt(x*x + y*y)

	dr_factor=dr_factor
	# internal_energy=9e-5
	gamma=7/5
	#c_s = np.sqrt((gamma-1)*internal_energy)

	#Some initial calculations --ONLY INCLUDE ONCE USING TEMP GRADIENT
	r_ref = 1.
	c_s_ref = 0.05
	internal_energy_ref = c_s_ref**2/(gamma-1.)
	T_ref = c_s_ref**2
	#Here, the reference point is r = 1
	T_0 = T_ref*r_ref**(-temp_p)

	num_particles_at_r = np.zeros(0)
	mass_particles_at_r = np.zeros(0)
	all_r = np.zeros(0)
	all_dr = np.zeros(0)
	all_densities = np.zeros(0)
	all_temp = np.zeros(0)
	all_vel_radial = np.zeros(0)
	all_phi_dot = np.zeros(0)

	# all_rho_volume = np.zeros(0)

	# r_in = np.min(r); r_out = np.max(r);
	r_in = r_in; r_out = r_out+0.001;
	mult = 1e8

	#ADD BACK
	# for i in range(len(r)):
	# 	r[i] = round(r[i], 8)
		# r[i] = math.floor(r[i]*mult)/mult
	print("r_in: ", r_in); print("r_out: ", r_out);
	# exit()
	r_cur=1.0*r_in - width_ghost_in; iter=0; dr=0;
	while(r_cur <= r_out+width_ghost_out):
		T_cur = T_0*r_cur**temp_p
		c_s = T_cur**0.5
		if(temp_p==0.0):
			c_s = np.sqrt((gamma-1)*9e-5)
		dr = dr_factor*c_s/(r_cur**(-3./2.))
		#NOTE: here I need to round so that it does not include particles at the next r (since dr is so small initially)
		#TODO: check if this 1e8 is really okay
		# print('##################')
		# print(r_cur)
		# print(r_cur+dr)
		r_next = math.floor((r_cur+dr)*mult)/mult
		print(r_next)
		# ok = np.where((r>=math.floor(r_cur*mult)/mult) & (r<r_next)) #math.floor(r_cur*1e8)/1e8 #ADD BACK
		ok = np.where((r>=r_cur) & (r<(r_cur+dr)))

		if(val_to_plot=='histogram'):
			if (iter==0 or iter==1 or iter==100 or iter==101 or iter==200 or iter==500):
				massP_at_r = massP_array[ok]
				print(massP_at_r)
				plt.figure()
				plt.hist(massP_at_r)#, range=[0.0,1e-7])
				plt.show()

		# print('Current dr: ', dr)
		# print('Current radius: ', r[ok])
		numP_cur = len(ok[0])
		print(numP_cur)
		massP_cur = np.sum(massP_array[ok])/len(massP_array[ok])

		num_particles_at_r = np.append(num_particles_at_r, numP_cur)
		mass_particles_at_r = np.append(mass_particles_at_r, massP_cur)

		#Temperature
		internal_energy_cur = np.sum(internal_energyP[ok])/len(internal_energyP[ok])
		temp_cur = (gamma-1.)*internal_energy_cur

		#Density
		density_cur = np.sum(densitiesP[ok])/len(densitiesP[ok])
		# rho_volume_cur = density_cur * r_cur**(-3/2) / temp_cur**0.5
		# density_cur = density_cur * temp_cur**0.5 * r_cur**(3/2) #Just a test for density -- not good

		#Radial velocity
		vel_radial_cur = np.sum(v_radial[ok])/len(v_radial[ok])

		#Phi dot
		phi_dot_cur = np.average(phi_dot[ok])# np.sum(phi_dot[ok])/len(phi_dot[ok])

		all_r = np.append(all_r, r_cur)
		all_dr = np.append(all_dr, dr)
		all_densities = np.append(all_densities, density_cur)
		all_temp = np.append(all_temp, temp_cur)
		all_vel_radial = np.append(all_vel_radial, vel_radial_cur)
		all_phi_dot = np.append(all_phi_dot, phi_dot_cur)
		# all_rho_volume = np.append(all_rho_volume, rho_volume_cur)
		r_cur+=dr; iter+=1;
	
	print("ITER: ", iter)

	density_by_numP = num_particles_at_r*mass_particles_at_r/(2*np.pi*all_r*all_dr)

	if(use_fname == True): #case for IC file
		density = density_by_numP
	else:
		density = all_densities

	rho_volume = density * all_r**(-3/2) / all_temp**0.5

	all_centrifugal_accel = all_phi_dot*all_phi_dot*all_r
	all_vel_phi = all_phi_dot*all_r

	#Theoretical density and v_phi calculation
	rho0 = rho_target/r_in**(-1.0)
	density_theoretical = rho0*all_r**(-1.0)
	v_phi_theoretical = np.sqrt((all_r**(-3/2))**2 * all_r**2 + (-1.0-7/4)*0.0025/all_r**0.5)


	if(plot_all == True):
		return all_r, num_particles_at_r, density, rho_volume, all_temp, all_vel_radial
	else:
		if(val_to_plot=='rho'):
			return all_r, num_particles_at_r, density, rho_volume, density_theoretical
		if(val_to_plot=='temp'):
			return all_r, num_particles_at_r, all_temp
		if(val_to_plot=='vel_radial'):
			return all_r, num_particles_at_r, all_vel_radial
		if(val_to_plot=='vel_phi'):
			return all_r, num_particles_at_r, all_vel_phi
		if(val_to_plot=='vel_both'):
			return all_r, num_particles_at_r, all_vel_radial, all_vel_phi, v_phi_theoretical
		if(val_to_plot=='centrifugal_accel'):
			return all_r, num_particles_at_r, all_centrifugal_accel
		if(val_to_plot=='resolution_test'):
			return all_r, num_particles_at_r, density, density_theoretical, all_vel_phi, v_phi_theoretical

def calculate_avg_error(snum=0, sdir='./output/',
						ptype='PartType0', dr_factor=0.1, p=0.0, rho_target=1.0, temp_p=-0.5):
	
	print("Using snapshot file\n")
	P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - 2.0
	y = Pc[:, 1] - 2.0
	density = P['Density']
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	phi_dot = ((x*vy) - (y*vx)) / (x**2 + y**2)
	all_r = np.sqrt(x*x + y*y)
	v_phi = phi_dot*all_r
	numP = len(all_r)
	r_in=0.2; r_out=2.0;

	#Theoretical density and v_phi calculation
	rho0 = rho_target/r_in**(-1.0)
	density_theoretical = rho0*all_r**(-1.0)
	v_phi_theoretical = np.sqrt((all_r**(-3/2))**2 * all_r**2 + (-1.0-7/4)*0.0025/all_r**0.5)

	# diff_density = np.average(density - density_theoretical)
	# diff_velocity = np.average(v_phi - v_phi_theoretical)

	diff_density = (np.sum((density - density_theoretical)**2)/numP)**0.5
	diff_velocity = (np.sum((v_phi - v_phi_theoretical)**2)/numP)**0.5

	return numP, diff_density, diff_velocity

def get_velocities(snum=0, sdir='./output/',
					ptype='PartType0', dr_factor=0.1, p=0.0, rho_target=1.0, temp_p=-0.5):
	
	print("Using snapshot file\n")
	P_File = load_snap(sdir, snum)

	P = P_File[ptype]
	Pc = np.array(P['Coordinates'])
	x = Pc[:, 0] - 2.0
	y = Pc[:, 1] - 2.0
	vx = np.array(P['Velocities'][:, 0])
	vy = np.array(P['Velocities'][:, 1])
	phi_dot = ((x*vy) - (y*vx)) / (x**2 + y**2)
	all_r = np.sqrt(x*x + y*y)
	v_phi = phi_dot*all_r
	v_radial = (x*vx + y*vy)/np.sqrt(x**2 + y**2)
	numP = len(all_r)
	r_in=0.2; r_out=2.0;

	#Theoretical v_phi calculation
	v_phi_theoretical = np.sqrt((all_r**(-3/2))**2 * all_r**2 + (-1.0-7/4)*0.0025/all_r**0.5)

	return all_r, v_radial, v_phi, v_phi_theoretical

# plot_gas_density(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_ic_phil/output/')
# plot_gas_density(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/keplerian_disk_2d_updated_w_shift_phil_coords/output/', phil=False)
# plot_gas_density(snum=10, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_phil_public_00/output/')

# plot_gas_density(snum=0, sdir='/Users/mayascomputer/Codes/gizmo_code/runs/2d_keplerian_test_runs/kep_ic_phil_public_00/output/')
# plot_gas_density(use_fname=True, fname='./ICs/keplerian_ics.hdf5', phil=True)



