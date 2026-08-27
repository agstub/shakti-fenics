# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')
import numpy as np
from params import rho_i, rho_w, g
from pathlib import Path
from dolfinx.io import gmsh
from netCDF4 import Dataset
from load_lakes import lake_inventory
from model import model
from shapely import Point

def initialize(comm):
    # select lake from inventory and set geographic bounds
    lake_name = 'Thw_124' 
    
    # Define mesh (see create_mesh.ipynb notebook for example)
    domain, *_ = gmsh.read_from_msh("../meshes/"+lake_name+"_mesh_fine.msh", comm, gdim=2)
    
    # initialize model object
    md = model(comm,domain)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    md.lake_name = lake_name
    
    # set results name for saving
    md.N_bdry = 4.5e5 # boundary condition function for N at outflow

    # lake outline from geodataframe, for defining lake boundary
    md.outline = lake_inventory.loc[lake_inventory['name']==md.lake_name]
    md.outline = md.outline.scale(xfact=1e3,yfact=1e3,origin=(0,0,0)) # convert to meters
    md.set_lake_bdry(md.outline)   # set lake boundary function
    
    # set neighbor lake boundaries (future work)
    neighbors = []
    for name in lake_inventory['name'].to_list():
        outline_test = lake_inventory.loc[lake_inventory['name']==name]
        x_test = float(outline_test.centroid.x.iloc[0])*1e3
        y_test = float(outline_test.centroid.y.iloc[0])*1e3
        if (x_test>md.x.min())&(x_test<md.x.max())&(y_test>md.y.min())&(y_test<md.y.max()):
            neighbors.append(name)
    # if neighbors:
    #     for name in neighbors:
    #         outline_nbr = lake_inventory.loc[lake_inventory['name']==name].scale(xfact=1e3,yfact=1e3,origin=(0,0,0))
    #         for j in range(md.lake_bdry.x.array.size):
    #             point = Point(md.domain.geometry.x[j,0],md.domain.geometry.x[j,1])
    #             md.lake_bdry.x.array[j] += outline_nbr.geometry.contains(point).iloc[0]
    # md.lake_bdry.x.array[:] = (md.lake_bdry.x.array[:]>0)
    # md.lake_bdry.x.scatter_forward()
    
    # put neighbors in model config    
    neighbors = md.comm.gather(neighbors,root=0)
    if md.rank==0:
        neighbors_list = [item for sublist in neighbors for item in sublist]
        neighbors_list = list(set(neighbors_list))
        md.model_config['neighbors'] = neighbors_list
    
    # define bed geometry (BedMachine Antarctica)
    bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineAntarctica-v3.nc')
    bed = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
    x = bedmachine['x'][:].data.astype(np.float64)
    y = np.flipud(bedmachine['y'][:].data.astype(np.float64))
    bed_interp = md.interp_data("z_b", x, y, bed)
    del bedmachine, x, y, bed
    comm.barrier()  

    # define surface elevation (ICESat-2 ATL14)
    atl14 = Dataset('/Users/agstubbl/Desktop/ICESat-2/ATL14_A3_0325_100m_004_05.nc')
    h = atl14['h'][:].filled()               # elevation (m)
    x = atl14['x'][:].filled()               # x coordinate array (m)
    y = atl14['y'][:].filled()               # y coordinate array (m)
    h_interp = md.interp_data("z_s", x, y, h)
    del atl14, h, x, y
    comm.barrier()  

    # Geoethermal heat flux: AQ1 GHF (Stal) 
    aq1 = Dataset('/Users/agstubbl/Desktop/GHF/aq1_01_20.nc')
    ghf = aq1['Q'][:].data
    x = aq1['X'][:].data
    y = aq1['Y'][:].data
    _ = md.interp_data("G", x, y, ghf)
    del aq1, x, y, ghf
    comm.barrier()  

    # define initial conditions
    md.b_init.x.array[:] = 0.001  
    md.N_init.interpolate(lambda x:md.N_bdry+0*x[0])      
    md.qx_init.interpolate(lambda x: 0*x[0]) 
    md.qy_init.interpolate(lambda x: 0*x[0])  
    
    # TEST!!! --> this works well
    md.N_init.x.array[:] = (1-md.lake_bdry.x.array[:])*md.N_bdry + (3.58e5)*md.lake_bdry.x.array[:]
    md.N_init.x.scatter_forward()
    
    # try initializing from reference results
    t_d = 8*365
    md.init_from_results('../results/Thw_124_reffine',t_d)

    # # define outflow boundary based on minimum potenetial condition (best checked by plotting in notebook)
    potential_interp = lambda x,y: rho_i*g*h_interp((x,y)) + (rho_w-rho_i)*g*bed_interp((x,y))
    P_min, P_max, P_std = 0,0,0
    potential__ = comm.gather(potential_interp(md.x,md.y),root=0)
    if md.rank == 0:
        potential__ = np.concatenate(potential__)
        P_min, P_max, P_std = np.min(potential__),np.max(potential__),np.std(potential__)
    comm.barrier()    
    P_min, P_max, P_std = comm.bcast(P_min, root=0),comm.bcast(P_max, root=0),comm.bcast(P_std, root=0)
    md.OutflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_min),2*P_std)
    md.InflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_max),0.7*P_std)
    
    # decide if outflow is allowed or not (default True)
    md.outflow_on = True

    # decide if lake is represented with a storage-type term (default True)
    md.storage_on = True

    # define source term in this example (comment out for zero source)
    # try source with -0.0005 m^3/s
    q_pump = -0.0005 
    sigma_p = 2000/3.0
    x_p = -1375*1e3
    y_p = -425*1e3 # site 1 = -375 //// site 2 = -425
    md.inputs.interpolate(lambda x: q_pump*np.exp(1)**(-((x[0]-x_p)**2+(x[1]-y_p)**2)/sigma_p**2) + 0*x[0])

    if md.rank==0:
        md.model_config['x_p'] = x_p
        md.model_config['y_p'] = y_p
        md.model_config['sigma_p'] = sigma_p
        
    # parameters!
    md.b_min = 1.0e-5
    md.b_max = 1.0e+1
    md.q_in = -2.0e-5            # water inflow through inflow boundary (negative for into domain)

    # define time stepping 
    days = 1*365 # subtract pump start time
    nt_per_day = 24
    t_final = (days/365)*3.154e7
    md.timesteps = np.linspace(0,t_final,int(days*nt_per_day))

    # frequency for saving files
    md.nt_save = nt_per_day
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/{md.lake_name}_test'
    
    return md