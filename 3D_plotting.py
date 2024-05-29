import plotly.graph_objects as go 
import plotly.express as px 
import scipy.interpolate as interpolate
import numpy as np
from plot_snapshot import load_snap, check_if_filename_exists

def plot_3D_shearing_box(snum=0, sdir='./output', plot_dust=False, Lbox_xy=12, Lbox_z=8):

    df = px.data.tips() 

    P_File = load_snap(sdir, snum);

    P_gas = P_File['PartType0']
    Pc_gas = np.array(P_gas['Coordinates'])
    gas_density = np.log10(P_gas['Density'][:])
    gas_velocity = np.array(P_gas['Velocities'])

    x_gas = Pc_gas[:, 0]
    y_gas = Pc_gas[:, 1]
    z_gas = Pc_gas[:, 2]
    vx_gas = gas_velocity[:, 0]
    vy_gas = gas_velocity[:, 1]
    vz_gas = gas_velocity[:, 2]

    print('Successfully input data for gas particles \n')

    if(plot_dust):
        P_dust = P_File['PartType3']
        Pc_dust = np.array(P_dust['Coordinates'])
        dust_density = np.log10(P_dust['Density'][:])
        dust_velocity = np.array(P_dust['Velocities'])

        x_dust = Pc_dust[:, 0]
        y_dust = Pc_dust[:, 1]
        z_dust = Pc_dust[:, 2]
        vx_dust = dust_velocity[:, 0]
        vy_dust = dust_velocity[:, 1]
        vz_dust = dust_velocity[:, 2]
        print('Successfully input data for dust particles \n')

    ngrid=10
    #ngrid_z = round(ngrid*Lbox_z/Lbox_xy)

    ngridj = 10j

    x_gas = 1.0 * (x_gas / x_gas.max())
    y_gas = 1.0 * (y_gas / y_gas.max())
    z_gas = 1.0 * (z_gas / z_gas.max())

    X_gas, Y_gas, Z_gas = np.mgrid[0:1:ngridj, 0:1:ngridj, 0:1:ngridj]

    # X_gas, Y_gas, Z_gas = np.meshgrid(np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid), np.linspace(0, 1, ngrid))
    print('Successfully created 3D meshgrid \n')

    Density_gas = interpolate.griddata((x_gas, y_gas, z_gas), gas_density, (X_gas, Y_gas, Z_gas), method='linear')
    print('Successfully interpolated density values \n')

    # fig = go.Figure(data=go.Volume( 
    #     x=x_gas, 
    #     y=y_gas, 
    #     z=z_gas,  
    #     # isomin=-1.0, 
    #     # isomax=1.0, 
    #     value=gas_density,
    #     opacity=0.1,  
    #     # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]], 
    #     colorscale='RdBu'
    #     )) 
      
    # fig.show()

    fig = go.Figure(data=go.Volume( 
        x=X_gas.flatten()*boxL_xy-(boxL_xy/2), 
        y=Y_gas.flatten()*boxL_xy-(boxL_xy/2), 
        z=Z_gas.flatten()*boxL_z-(boxL_z/2),  
        # isomin=-1.0, 
        # isomax=1.0, 
        value=Density_gas.flatten(), 
        opacity=0.1,  
        # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]], 
        colorscale='RdBu'
        )) 
      
    fig.show()

plot_3D_shearing_box(snum=51, sdir='/Users/mayatatarelli/Codes/gizmo-code/runs/3d_shearing_box/box_xy_12_z_8_dust/output', plot_dust=False, Lbox_xy=12, Lbox_z=8)

