# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import Function, FunctionSpace
from ufl import dx
from params import L,W,nx,ny,rho_i,g,H,resultsname
from mpi4py import MPI
from fem_space import mixed_space, vector_space

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define domain 
p0 = [-0.5*L,-0.5*W]
p1 = [0.5*L,0.5*W]
domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [nx, ny],cell_type=CellType.triangle) 

# define bed geometry
V0 = FunctionSpace(domain, ("CG", 1))
bed = lambda x: 0.02*(x[0]+0.5*L) - 100*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2)) 
z_b = Function(V0)
z_b.interpolate(bed)

# define surface elevation
z_s = Function(V0)
surf = lambda x: 0.01*(x[0]+0.5*L) + H  -25*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2))
z_s.interpolate(surf)

q_inflow = 1e-2  # inflow on left boundary (m^2/s)
q_dist = 2e-6    # distributed input  (m/s)

# define initial conditions
b0 = 0.01
N0 = rho_i*g*H
qx0 = -q_inflow
qy0 = 0

V = mixed_space(domain)
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

# define water flux boundary condition (Neumann)
V_q = vector_space(domain)
q_in = Function(V_q)
q_in.sub(0).interpolate(lambda x:qx0+0*x[0])
q_in.sub(1).interpolate(lambda x:qy0+0*x[0])

# define time stepping 
days = 600
nt_per_day = 100
t_final = (days/365)*3.154e7
timesteps = np.linspace(0,t_final,int(days*nt_per_day))

nt_save = nt_per_day





