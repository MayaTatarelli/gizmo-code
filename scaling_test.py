import numpy as np
import matplotlib.pyplot as plt

#rho_10
num_cores_10 = np.array([32, 64, 128, 256, 512, 1024])
run_time_hrs_10 = np.array([2.28, 1.9, 2.28, 3.47])


#rho_20
num_cores_20 = np.array([32, 48, 64, 80, 88, 96, 104, 128, 160, 192, 256])
run_time_hrs_20 = np.array([3.0, 2.25, 2.0, 1.9, 1.87, 1.77, 1.87, 1.88, 2.08, 2.63, 2.23])

num_cores_20_beluga = np.array([32, 48, 80, 88, 96, 128, 160, 256])
run_time_hrs_20_beluga = np.array([3.7, 2.92, 2.33, 2.45, 2.2, 2.05, 2.55, 2.6])

#3D runs:
num_cores_3D = np.array([256, 272, 288, 304, 320, 336])
run_time_hrs_3D = np.array([6.87, 7.07, 6.98, 6.89, 7.05, 7.27])
run_time_hrs_3D_beluga = np.array([0, 0, 0, 0, 0, 0])

plt.figure()

plt.plot(num_cores_20_beluga, run_time_hrs_20_beluga, marker='o', label='Beluga runs')

plt.plot(num_cores_20, run_time_hrs_20, marker='o', label='Narval runs')

plt.axvline(x=40, linestyle='--', color='#1f77b4', label='Full node on Beluga (40 cores)')
plt.axvline(x=64, linestyle='--', color='#ff7f0e', label='Full node on Narval (64 cores)')

plt.xlabel('# of cores')
plt.ylabel('runtime (hrs) for 200 orbits')
plt.title('Scaling Analysis on Narval and Beluga (2D global)')
plt.legend()
plt.show()
exit()

plt.figure()
plt.plot(num_cores_3D, run_time_hrs_3D_beluga, marker='o')
plt.axvline(x=280, linestyle='--', color='k', label='7 nodes (280 cores)')
plt.xlabel('# of cores')
plt.ylabel('runtime (hrs) for 100 orbits')
plt.title('Scaling Analysis on Beluga (3D shearing box)')
plt.legend()
plt.show()
exit()

plt.figure()
plt.plot(num_cores_3D, run_time_hrs_3D, marker='o')
plt.axvline(x=256, linestyle='--', color='k', label='4 nodes (256 cores)')
plt.xlabel('# of cores')
plt.ylabel('runtime (hrs) for 100 orbits')
plt.title('Scaling Analysis (3D shearing box)')
plt.legend()
plt.show()
exit()

plt.figure()
plt.plot(num_cores_20, run_time_hrs_20, marker='o')
plt.axvline(x=64, linestyle='--', color='k', label='1 node (64 cores)')
plt.xlabel('# of cores')
plt.ylabel('runtime (hrs) for 200 orbits')
plt.title('Scaling Analysis')
plt.legend()
plt.show()

exit()
plt.figure()
plt.plot(num_cores_20[0:5], run_time_hrs_20[0:5], marker='o')
plt.axvline(x=64, linestyle='--', color='k', label='1 node (64 cores)')
plt.xlabel('# of cores')
plt.ylabel('runtime (hrs) for 200 orbits')
plt.title('Scaling Analysis')
plt.legend()
plt.show()