# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys
sys.path.insert(0, '../source')

import numpy as np
from dolfinx.fem import Expression, Function, functionspace
from params import rho_i, rho_w, g
from mpi4py import MPI
from dof_helpers import ghost_mask
from pathlib import Path
from constitutive import BackgroundPotential
from dolfinx.io import gmshio
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from load_lakes import gdf
from shapely import Point
from basix.ufl import element

parent_dir = (Path(__file__).resolve()).parent.parent

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

setup_name = ''

# select lake from inventory and set geographic bounds
lake_name = 'Cook_E2' 
outline = gdf.loc[gdf['name']==lake_name]
outline = outline.scale(xfact=1e3,yfact=1e3,origin=(0,0,0))
x0 = float(outline.centroid.x.iloc[0])
y0 = float(outline.centroid.y.iloc[0])

# set results name for saving
N0 = 3.7e5 
experiment_name = f'cooke2_{int(N0/1e3):d}kpa'
resultsname = f'{parent_dir}/results/{experiment_name}'

# Define domain 
domain, cell_tags, facet_tags = gmshio.read_from_msh("../meshes/"+lake_name+"_mesh_alt.msh", MPI.COMM_WORLD, gdim=2)

# define function space (piecewise linear scalar) for inputs
V = functionspace(domain, ("CG", 1))
mask = ghost_mask(V) # mask for ghost points

# function space for water flux
P1_vec = element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))
V_flux = functionspace(domain,P1_vec) 

# L0 is the half-width  of bounding box surrounding lake
# mesh is contained within this box
L0 = 50*1000
x_min = x0-L0
x_max = x0+L0
y_min = y0-L0
y_max = y0+L0

# define bed geometry
bedmachine = Dataset('/Users/agstubbl/Desktop/bedmachine/BedMachineAntarctica-v3.nc')
x_bm = bedmachine['x'][:].data.astype(np.float64)
y_bm = np.flipud(bedmachine['y'][:].data.astype(np.float64))
H_bm = np.flipud(bedmachine['thickness'][:].data.astype(np.float64))
bed_bm = np.flipud(bedmachine['bed'][:].data.astype(np.float64))
xb_sub = x_bm[(x_bm>=x_min)&(x_bm<=x_max)]
yb_sub = y_bm[(y_bm>=y_min)&(y_bm<=y_max)]
bed_sub = bed_bm[np.ix_(np.arange(0,np.size(y_bm),1)[(y_bm>=y_min)&(y_bm<=y_max)],np.arange(0,np.size(x_bm),1)[(x_bm>=x_min)&(x_bm<=x_max)])].T
bed_interp = RegularGridInterpolator((xb_sub, yb_sub), bed_sub, bounds_error=False, fill_value=None)
z_b = Function(V) # dolfinx function for the bed elevation
z_b.x.array[:] = bed_interp( (domain.geometry.x[:,0],domain.geometry.x[:,1]) )
z_b.x.scatter_forward()
del bedmachine, x_bm, y_bm, H_bm, bed_bm, xb_sub, yb_sub, bed_sub
comm.Barrier()  

# define surface elevation
# load surface elevation, make interpolation, and interpolate onto mesh nodes
atl14 = Dataset('/Users/agstubbl/Desktop/ICESat-2/ATL14_A4_0325_100m_004_05.nc')

h = atl14['h'][:]               # elevation (m)
x = atl14['x'][:]               # x coordinate array (m)
y = atl14['y'][:]               # y coordinate array (m)

# extract the data that is inside the bounding box
x_sub = x[(x>=x_min)&(x<=x_max)].filled()
y_sub = y[(y>=y_min)&(y<=y_max)].filled()
inds_x = np.arange(0,np.size(x),1)[(x>=x_min)&(x<=x_max)]
inds_y = np.arange(0,np.size(y),1)[(y>=y_min)&(y<=y_max)]
inds_xy = np.ix_(inds_y,inds_x)
h_sub = h[inds_xy].filled().T 
h_interp = RegularGridInterpolator((x_sub, y_sub), h_sub, bounds_error=False, fill_value=None)
z_s = Function(V) # dolfinx function for the surface elevation
z_s.x.array[:] = h_interp( (domain.geometry.x[:,0],domain.geometry.x[:,1]) )
z_s.x.scatter_forward()
del atl14, h, x, y, x_sub, y_sub, inds_x, inds_y, inds_xy, h_sub
comm.Barrier()  

# inputs and initial conditions
q_dist = 0    # distributed input  [m/s]

# Geoethermal heat flux
# AQ1 GHF (Stal) 
aq1 = Dataset('/Users/agstubbl/Desktop/GHF/aq1_01_20.nc')
x = aq1['X'][:].data
y = aq1['Y'][:].data
ghf = aq1['Q'][:].data
ind_x = np.arange(0,np.size(x),1)
ind_y = np.arange(0,np.size(y),1)
x_sub = x[(x>=x_min)&(x<=x_max)]
y_sub = y[(y>=y_min)&(y<=y_max)]
inds_x = ind_x[(x>=x_min)&(x<=x_max)]
inds_y = ind_y[(y>=y_min)&(y<=y_max)]
nx = np.size(inds_x)
ny = np.size(inds_y)
inds_xy = np.ix_(inds_y,inds_x)
ghf_sub = np.zeros((ny,nx))
ghf_sub = ghf[inds_xy]
G = ghf_sub.mean() # geothermal heat flux (W/m^2)
G_mean = 0 
G__ = comm.gather(G,root=0)
if rank == 0:
    G_mean = np.mean(G__)    
comm.Barrier()    
G = comm.bcast(G_mean, root=0)

del aq1, x, y, ghf, ind_x, ind_y, x_sub, y_sub, inds_x, inds_y, nx, ny, inds_xy, ghf_sub
comm.Barrier()  

# define initial conditions
b0 = 0.001
qx0 = 0
qy0 = 0
N0 = N0 #defined above

# boundary condition for N at outflow
N_bdry = N0

# define minimum gap height
b_min = 1.0e-5

# define outflow boundary based on minimum potenetial condition
P_min, P_std = 0,0
potential = Function(V)
potential.interpolate(Expression(BackgroundPotential(z_b,z_s), V.element.interpolation_points()))
potential__ = comm.gather(potential.x.array[mask],root=0)

if rank == 0:
    P_min = np.min(np.concatenate(potential__))
    P_std = np.std(np.concatenate(potential__))
comm.Barrier()    
P_min = comm.bcast(P_min, root=0)
P_std = comm.bcast(P_std, root=0)    
    
potential_interp = lambda x,y: rho_i*g*h_interp((x,y)) + (rho_w-rho_i)*g*bed_interp((x,y))

def OutflowBoundary(x):
    return np.less(np.abs(potential_interp(x[0],x[1])-P_min),0.5*P_std)

# decide if outflow is allowed or not (default True)
outflow = True

# define initial conditions
b_init = Function(V)
N_init = Function(V)
q_init = Function(V_flux)

N_init.interpolate(lambda x:N0+0*x[0])          # initial N
q_init.sub(0).interpolate(lambda x:qx0+0*x[0])  # initial qx
q_init.sub(0).interpolate(lambda x:qy0+0*x[0])  # initial qy

# initialize gap height b=b0 plus some random noise
b_init.x.array[:] = b0 + np.random.normal(scale=0.005,size=np.size(b_init.x.array[:])) 

# define moulin source term - none defined in this example
inputs = Function(V)
inputs.interpolate(lambda x: q_dist + 0*x[0] )

# define storage function
lake_bdry = Function(V)
for j in range(lake_bdry.x.array.size):
    point = Point(domain.geometry.x[j,0],domain.geometry.x[j,1])
    lake_bdry.x.array[j] = outline.geometry.contains(point).iloc[0]
lake_bdry.x.scatter_forward()

# decide if lake is represented with a storage-type term
storage = True

stagger_solve = True

# define time stepping 
days = 10*365
nt_per_day = 24
t_final = (days/365)*3.154e7
timesteps = np.linspace(0,t_final,int(days*nt_per_day))

# frequency for saving files
nt_save = nt_per_day
nt_check = 50*nt_save # checkpoint save for real-time 