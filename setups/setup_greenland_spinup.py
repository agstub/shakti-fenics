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
from model import model
from ll2xy import ll2xy
from mpi4py import MPI

def initialize(comm):
    # Store Glacier
    lake_name = 'store' 
    
    # location of supraglacial lake at Store Glacier
    lon_lake = -50.09
    lat_lake = 70.57
    x_l, y_l = ll2xy([lat_lake], [lon_lake], 1)
    x_l = x_l.mean()
    y_l = y_l.mean()
    
    # Define mesh (see create_mesh.ipynb notebook for example)
    domain, *_ = gmshio.read_from_msh("../meshes/"+lake_name+"_mesh.msh", comm, gdim=2)
    
    # initialize model object
    md = model(comm,domain)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    md.lake_name = lake_name
    
    # set results name for saving
    md.N_bdry = 1e6 # boundary condition function for N at outflow
    
    # define bed geometry (BedMachine Greenland)
    bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineGreenland-v5.nc')
    bed = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
    x = bedmachine['x'][:].data.astype(np.float64)
    y = np.flipud(bedmachine['y'][:].data.astype(np.float64))
    bed_interp = md.interp_data("z_b", x, y, bed)
    del bedmachine, x, y, bed
    comm.barrier()  

    # define surface elevation (ICESat-2 ATL14)
    atl14 = Dataset('/Users/agstubbl/Desktop/ICESat-2/ATL14_GL_0325_100m_004_05.nc')
    h = atl14['h'][:].filled()               # elevation (m)
    x = atl14['x'][:].filled()               # x coordinate array (m)
    y = atl14['y'][:].filled()               # y coordinate array (m)
    h_interp = md.interp_data("z_s", x, y, h)
    del atl14, h, x, y
    comm.barrier()  

    # Geoethermal heat flux: Colgan et al. (2022) ESSD 
    ghf_ds = Dataset('/Users/agstubbl/Desktop/GHF/geothermal_heat_flow_map_10km.nc')
    ghf = ghf_ds['GHF'][:].data.astype(np.float64) # units [mW/m^2]
    ghf = ghf / 1.0e3 # convert to W/m^2
    x = ghf_ds['X'][:].data.astype(np.float64)
    y = ghf_ds['Y'][:].data.astype(np.float64)
    
    # just take the mean over all processes since data is sparse
    x_sub = x[(x >= md.bounds[0]) & (x <= md.bounds[1])]
    y_sub = y[(y >= md.bounds[2]) & (y <= md.bounds[3])]
    ghf_sub = ghf[np.ix_(
    (y >= md.bounds[2]) & (y <= md.bounds[3]),
    (x >= md.bounds[0]) & (x <= md.bounds[1]))]
    ghf_mean = md.comm.allreduce(ghf_sub.mean(), op=MPI.SUM) / md.comm.size
    md.G.x.array[:] = ghf_mean
    md.G.x.scatter_forward()
    del ghf_ds, x, y , ghf, ghf_sub, x_sub, y_sub
    comm.barrier()  

    # define initial conditions
    md.b_init.x.array[:] = 0.01  
    md.N_init.interpolate(lambda x:md.N_bdry+0*x[0])      
    md.qx_init.interpolate(lambda x: 0*x[0]) 
    md.qy_init.interpolate(lambda x: 0*x[0])  
    
    # try initializing from reference 'spinup' results if availaible
    # (!!) note: comment this out if spinup has not been run
    #t_d = 600 # model day to initialize from
    #md.init_from_results('../results/store_spinup',t_d)

    # # define outflow boundary based on minimum potenetial condition (best checked by plotting in notebook)
    potential_interp = lambda x,y: rho_i*g*h_interp((x,y)) + (rho_w-rho_i)*g*bed_interp((x,y))
    P_min, P_max, P_std = 0,0,0
    potential__ = comm.gather(potential_interp(md.x,md.y),root=0)
    if md.rank == 0:
        potential__ = np.concatenate(potential__)
        P_min, P_max, P_std = np.min(potential__),np.max(potential__),np.std(potential__)
    comm.barrier()    
    P_min, P_max, P_std = comm.bcast(P_min, root=0),comm.bcast(P_max, root=0),comm.bcast(P_std, root=0)
    md.OutflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_min),1.5*P_std)
    md.InflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_max),0.7*P_std)
    
    # decide if outflow is allowed or not (default True)
    md.outflow_on = True

    # decide if lake is represented with a storage-type term (default True)
    md.storage_on = False # this is for subglacial lake storage

    # define source term in this example 
    q_l = 1.0e-3
    sigma_l = 1000/3.0
    
    # (!!) note: comment out this line for no-drainage "spinup":
    # md.inputs.interpolate(lambda x: q_l*np.exp(1)**(-((x[0]-x_l)**2+(x[1]-y_l)**2)/sigma_l**2) + 0*x[0])

    if md.rank==0:
        md.model_config['x_l'] = x_l
        md.model_config['y_l'] = y_l
        md.model_config['sigma_l'] = sigma_l
        
    # parameters!
    md.b_min = 1.0e-5
    md.b_max = 1.0e+1
    md.q_in = -1.0e-4            # water inflow through inflow boundary (negative for into domain)

    # define time stepping 
    days = 2*365                      # (!!) note: set to 2*365 for spinup
    nt_per_hour = 1                   # (!!) note: set to 1 for spinup
    nt_per_day = 24*nt_per_hour
    t_final = (days/365)*3.154e7
    md.timesteps = np.linspace(0,t_final,int(days*nt_per_day))

    # frequency for saving files
    md.nt_save = nt_per_day      # (!!) note: set to nt_per_day for spinup
    md.nt_check = 50*md.nt_save  # checkpoint save for real-time plotting...
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/{md.lake_name}_spinup'
    
    return md