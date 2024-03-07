import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import Function, FunctionSpace,assemble_scalar,form
from ufl import dx
from params import L,W,nx,ny,rho_i,g,H
from solvers import solve
from mpi4py import MPI
from fem_space import mixed_space, vector_space

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define domain
p0 = [-0.5*L,-0.5*W]
p1 = [0.5*L,0.5*W]
domain = create_rectangle(MPI.COMM_WORLD,[p0,p1], [nx, ny],cell_type=CellType.triangle) 

# define bed geometry
slope = 0.02
V0 = FunctionSpace(domain, ("CG", 1))
bed = lambda x: slope*(x[0]+0.5*L)
z_b = Function(V0)
z_b.interpolate(bed)

# define surface elevation
z_s = z_b + H

# define initial conditions
b0 = 0.01
N0 = rho_i*g*H
qx0 = 0
qy0 = 0

V = mixed_space(domain)
initial = Function(V)
initial.sub(0).interpolate(lambda x:b0+0*x[0])           # initial b
initial.sub(1).interpolate(lambda x: N0 + 0*x[0]) # initial N
initial.sub(2).sub(0).interpolate(lambda x:qx0+0*x[0])  # initial qx
initial.sub(2).sub(1).interpolate(lambda x:qy0+0*x[0])   # initial qy

# define moulin source term
moulin = Function(V0)
sigma = 1e2/3.
moulin_ = lambda x: np.exp(1)**(-(x[0]**2+x[1]**2)/sigma**2)
moulin.interpolate(moulin_)
scale = assemble_scalar(form(moulin*dx))
moulin = 5*moulin/scale 


# define water flux boundary condition (Neumann)
V_q = vector_space(domain)
q_in = Function(V_q)
q_in.sub(0).interpolate(lambda x:qx0+0*x[0])
q_in.sub(1).interpolate(lambda x:0+0*x[0])

# define time stepping 
days = 1
t_final = (days/365)*3.154e7
timesteps = np.linspace(0,t_final,int(days*100))

# solve the problem
# results are saved in a 'results' directory
solve(domain,initial,timesteps,z_b,z_s,q_in,moulin)





