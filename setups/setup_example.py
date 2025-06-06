# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys
sys.path.insert(0, '../source')

import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import Expression, Function, functionspace
from ufl import dot
from params import rho_i,g
from mpi4py import MPI
from fem_space import mixed_space, vector_space
from pathlib import Path
from constitutive import BackgroundGradient

parent_dir = (Path(__file__).resolve()).parent.parent

setup_name = ''

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# set results name for saving
experiment_name = 'example'
resultsname = f'{parent_dir}/results/{experiment_name}'

# Define domain 
nx,ny = 128,128
H = 500            # ice thickness (m) [uniform examples]
L =  20*H          # length of domain
W =  20*H          # width of domain

p0 = [-0.5*L,-0.5*W]
p1 = [0.5*L,0.5*W]
domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [nx, ny],cell_type=CellType.triangle) 

# define function space (piecewise linear scalar) for inputs
V0 = functionspace(domain, ("CG", 1))

# define function space for full solution
V = mixed_space(domain)

# define bed geometry
bed = lambda x: 0.02*(x[0]+0.5*L) - 100*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2)) 
z_b = Function(V0)
z_b.interpolate(bed)

# define surface elevation
z_s = Function(V0)
surf = lambda x: 0.01*(x[0]+0.5*L) + H  -25*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2))
z_s.interpolate(surf)

q_dist = 2e-6    # distributed input  (m/s)

# define initial conditions
b0 = 0.01
N0 = 0.5*rho_i*g*H
qx0 = -1e-2
qy0 = 0

# set geothermal heat flux
G = 0.05        

# boundary condition for N at outflow
N_bdry = N0

# define minimum gap height
b_min = 1e-5

# define outflow boundary
def OutflowBoundary(x):
    # Left boundary (inflow/outflow)
    return np.isclose(x[0],-L/2.0)

# set initial conditions
initial = Function(V)
initial.sub(1).interpolate(lambda x: N0 + 0*x[0])       # initial N
initial.sub(2).sub(0).interpolate(lambda x:qx0+0*x[0])  # initial qx
initial.sub(2).sub(1).interpolate(lambda x:qy0+0*x[0])  # initial qy

# initialize gap height b with some random noise
b_temp = Function(V0)
b_temp.x.array[:] = b0 + np.random.normal(scale=0.005,size=np.size(b_temp.x.array[:])) 
initial.sub(0).interpolate(b_temp)

# define moulin source term
inputs = Function(V0)
inputs_ = lambda x: q_dist + 0*x[0] 
inputs.interpolate(inputs_)

# define storage function for example
lake_bdry = Function(V0)
grad_h0 = BackgroundGradient(z_b,z_s)
lake_bdry_expr = np.exp(1)**(-(150*dot(grad_h0,grad_h0)**(0.5))**8)
lake_bdry.interpolate(Expression(lake_bdry_expr, V0.element.interpolation_points()))

# decide if lake is represented with a storage-type term
storage = True

# define time stepping 
days = 325
nt_per_day = 24
t_final = (days/365)*3.154e7
timesteps = np.linspace(0,t_final,int(days*nt_per_day))

# frequency for saving files
nt_save = nt_per_day