# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')
import numpy as np
from params import rho_i, rho_w, g
from pathlib import Path
from dolfinx.io import gmshio
from netCDF4 import Dataset
from load_lakes import lake_inventory
from model_setup import model_setup

def initialize(comm):
    # select lake from inventory and set geographic bounds
    lake_name = 'Cook_E2' 
    
    # Define mesh (see create_mesh.ipynb notebook for example)
    domain, *_ = gmshio.read_from_msh("../meshes/"+lake_name+"_mesh.msh", comm, gdim=2)
    
    # initialize model object
    md = model_setup(comm,domain)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    md.lake_name = lake_name
    
    # set results name for saving
    md.N_bdry = 3.7e5 # boundary condition function for N at outflow
    parent_dir = (Path(__file__).resolve()).parent.parent
    md.results_name = f'{parent_dir}/results/{md.lake_name}_{int(md.N_bdry/1e3):d}kpa'

    # lake outline from geodataframe, for defining lake boundary
    md.outline = lake_inventory.loc[lake_inventory['name']==md.lake_name]
    md.outline = md.outline.scale(xfact=1e3,yfact=1e3,origin=(0,0,0)) # convert to meters
    md.set_lake_bdry(md.outline)   # set lake boundary function
    
    # define bed geometry (BedMachine Antarctica)
    bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineAntarctica-v3.nc')
    bed = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
    x = bedmachine['x'][:].data.astype(np.float64)
    y = np.flipud(bedmachine['y'][:].data.astype(np.float64))
    bed_interp = md.interp_data("z_b", x, y, bed)
    del bedmachine, x, y, bed
    comm.barrier()  

    # define surface elevation (ICESat-2 ATL14)
    atl14 = Dataset('/Users/agstubbl/Desktop/ICESat-2/ATL14_A4_0325_100m_004_05.nc')
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
    md.b_init.x.array[:] = 0.001 + np.random.normal(scale=0.005,size=np.size(md.b_init.x.array[:])) 
    md.N_init.interpolate(lambda x:md.N_bdry+0*x[0])      
    md.q_init.sub(0).interpolate(lambda x: 0*x[0]) 
    md.q_init.sub(1).interpolate(lambda x: 0*x[0])  

    # # define outflow boundary based on minimum potenetial condition (best checked by plotting in notebook)
    potential_interp = lambda x,y: rho_i*g*h_interp((x,y)) + (rho_w-rho_i)*g*bed_interp((x,y))
    P_min, P_std = 0,0
    potential__ = comm.gather(potential_interp(md.x,md.y),root=0)
    if md.rank == 0:
        potential__ = np.concatenate(potential__)
        P_min, P_std = np.min(potential__), np.std(potential__)
    comm.barrier()    
    P_min, P_std = comm.bcast(P_min, root=0), comm.bcast(P_std, root=0)
    md.OutflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_min),0.5*P_std)
    
    # decide if outflow is allowed or not (default True)
    md.outflow_on = True

    # decide if lake is represented with a storage-type term (default True)
    md.storage_on = True

    # define moulin source term - zero (none) in this example
    md.inputs.interpolate(lambda x:  0*x[0] )

    # define time stepping 
    days = 10*365
    nt_per_day = 24
    t_final = (days/365)*3.154e7
    md.timesteps = np.linspace(0,t_final,int(days*nt_per_day))

    # frequency for saving files
    md.nt_save = nt_per_day
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    return md