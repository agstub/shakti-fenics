# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')

import numpy as np
from dolfinx.fem import Expression, Function
from params import rho_i, rho_w, g
from pathlib import Path
from constitutive import BackgroundPotential
from dolfinx.io import gmshio
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from load_lakes import gdf
from model import model

def initialize(comm):
    
    # select lake from inventory and set geographic bounds
    lake_name = 'Cook_E2' 
    
    # Define domain 
    domain, *_ = gmshio.read_from_msh("../meshes/"+lake_name+"_mesh.msh", comm, gdim=2)
    
    # initialize model object
    md = model(comm,domain)
    
    # set results name for saving
    N0 = 3.7e5     # effective pressure (N) initial condition / outflow condition
    md.N_bdry = N0 # boundary condition function for N at outflow

    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    md.lake_name = lake_name
    md.outline = gdf.loc[gdf['name']==md.lake_name]
    md.outline = md.outline.scale(xfact=1e3,yfact=1e3,origin=(0,0,0))
    
    # define lake boundary function
    md.set_lake_bdry(md.outline)
    x0, y0 = float(md.outline.centroid.x.iloc[0]), float(md.outline.centroid.y.iloc[0])
    L0 = 50*1000     # half-width of a bounding box surrounding the domain
    x_min, x_max, y_min, y_max = x0-L0, x0+L0, y0-L0, y0+L0
    
    parent_dir = (Path(__file__).resolve()).parent.parent
    md.results_name = f'{parent_dir}/results/{md.lake_name}_{int(N0/1e3):d}kpa'

    # define bed geometry
    bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineAntarctica-v3.nc')
    x = bedmachine['x'][:].data.astype(np.float64)
    y = np.flipud(bedmachine['y'][:].data.astype(np.float64))
    bed_bm = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
    x_sub, y_sub = x[(x>=x_min)&(x<=x_max)], y[(y>=y_min)&(y<=y_max)]
    bed_sub = bed_bm[np.ix_((y_min <= y) & (y <= y_max), (x_min <= x) & (x <= x_max))]
    bed_interp = RegularGridInterpolator((x_sub, y_sub), bed_sub.T, bounds_error=False, fill_value=None)
    md.z_b.x.array[:] = bed_interp((md.domain.geometry.x[:,0], md.domain.geometry.x[:,1]))
    md.z_b.x.scatter_forward()
    del bedmachine, x, y, bed_bm, x_sub, y_sub, bed_sub
    # comm.Barrier()  

    # define surface elevation
    # load surface elevation, make interpolation, and interpolate onto mesh nodes
    atl14 = Dataset('/Users/agstubbl/Desktop/ICESat-2/ATL14_A4_0325_100m_004_05.nc')
    h = atl14['h'][:].filled()               # elevation (m)
    x = atl14['x'][:].filled()               # x coordinate array (m)
    y = atl14['y'][:].filled()               # y coordinate array (m)
    x_sub, y_sub = x[(x>=x_min)&(x<=x_max)], y[(y>=y_min)&(y<=y_max)]
    h_sub = h[np.ix_((y_min <= y) & (y <= y_max), (x_min <= x) & (x <= x_max))]
    h_interp = RegularGridInterpolator((x_sub, y_sub), h_sub.T, bounds_error=False, fill_value=None)
    md.z_s.x.array[:] = h_interp((md.domain.geometry.x[:,0], md.domain.geometry.x[:,1]))
    md.z_s.x.scatter_forward()
    del atl14, h, x, y, x_sub, y_sub, h_sub
    # comm.Barrier()  

    # Geoethermal heat flux
    # AQ1 GHF (Stal) 
    aq1 = Dataset('/Users/agstubbl/Desktop/GHF/aq1_01_20.nc')
    x = aq1['X'][:].data
    y = aq1['Y'][:].data
    ghf = aq1['Q'][:].data
    x_sub, y_sub = x[(x>=x_min)&(x<=x_max)], y[(y>=y_min)&(y<=y_max)]
    ghf_sub = ghf[np.ix_((y_min <= y) & (y <= y_max), (x_min <= x) & (x <= x_max))]
    ghf_interp = RegularGridInterpolator((x_sub, y_sub), ghf_sub.T, bounds_error=False, fill_value=None)
    md.G.x.array[:] = ghf_interp((md.domain.geometry.x[:,0], md.domain.geometry.x[:,1]))
    md.G.x.scatter_forward()
    del aq1, x, y, ghf, x_sub, y_sub, ghf_sub
    # comm.Barrier()  

    # define initial conditions
    md.b_init.x.array[:] = 0.001 + np.random.normal(scale=0.005,size=np.size(md.b_init.x.array[:])) 
    md.N_init.interpolate(lambda x:N0+0*x[0])      
    md.q_init.sub(0).interpolate(lambda x: 0*x[0]) 
    md.q_init.sub(1).interpolate(lambda x: 0*x[0])  

    # define minimum gap height
    md.b_min = 1.0e-5

    # define outflow boundary based on minimum potenetial condition
    P_min, P_std = 0,0
    potential = Function(md.V)
    potential.interpolate(Expression(BackgroundPotential(md.z_b,md.z_s), md.V.element.interpolation_points()))
    potential__ = comm.gather(potential.x.array[md.mask],root=0)
    if md.rank == 0:
        P_min = np.min(np.concatenate(potential__))
        P_std = np.std(np.concatenate(potential__))
    comm.Barrier()    
    P_min = comm.bcast(P_min, root=0)
    P_std = comm.bcast(P_std, root=0)    
    potential_interp = lambda x,y: rho_i*g*h_interp((x,y)) + (rho_w-rho_i)*g*bed_interp((x,y))
    md.OutflowBoundary = lambda x: np.less(np.abs(potential_interp(x[0],x[1])-P_min),0.5*P_std)

    # decide if outflow is allowed or not (default True)
    md.outflow_on = True

    # decide if lake is represented with a storage-type term
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