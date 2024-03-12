import numpy as np
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import Function, FunctionSpace
from ufl import dx
from params import L,W,nx,ny,rho_i,g,H,resultsname
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
V0 = FunctionSpace(domain, ("CG", 1))
bed = lambda x: 0.02*(x[0]+0.5*L) - 150*np.exp(1)**(-((x[0]-0.25*W)**2+0.5*x[1]**2)/(1.333e3**2)) 
z_b = Function(V0)
z_b.interpolate(bed)

# define surface elevation
z_s = Function(V0)
surf = lambda x: 0.01*(x[0]+0.5*L) + H  -25*np.exp(1)**(-((x[0]-0.25*W)**2+0.5*x[1]**2)/(1.333e3**2))
z_s.interpolate(surf)

# define initial conditions
b0 = 0.01
N0 = rho_i*g*H
qx0 = -1e-2
qy0 = 0

V = mixed_space(domain)
initial = Function(V)
initial.sub(1).interpolate(lambda x: N0*(0.5*L-x[0])/L)       # initial N
initial.sub(2).sub(0).interpolate(lambda x:qx0+0*x[0])  # initial qx
initial.sub(2).sub(1).interpolate(lambda x:qy0+0*x[0])  # initial qy

# initialize gap height b with some random noise
b_temp = Function(V0)
b_temp.x.array[:] = b0 + np.random.normal(scale=0.005,size=np.size(b_temp.x.array[:])) 
initial.sub(0).interpolate(b_temp)

# define moulin source term
moulin = Function(V0)
sigma = 2e3/3.
moulin_ = lambda x: 2e-6 + 0*x[0] #2e-5*np.exp(1)**(-(x[0]**2+x[1]**2)/sigma**2) # 
moulin.interpolate(moulin_)

# # ~2e-5 exp input --> oscillations
# # 1e-5 constant input w/ qx0 = -5e-2 --> channels

# define water flux boundary condition (Neumann)
V_q = vector_space(domain)
q_in = Function(V_q)
q_in.sub(0).interpolate(lambda x:qx0+0*x[0])
q_in.sub(1).interpolate(lambda x:qy0+0*x[0])

# define time stepping 
days = 10
t_final = (days/365)*3.154e7
timesteps = np.linspace(0,t_final,int(days*100))

nt_save = 25

# solve the problem
# results are saved in a 'results' directory
solve(resultsname,domain,initial,timesteps,z_b,z_s,q_in,moulin,nt_save)






