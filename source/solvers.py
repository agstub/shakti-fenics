# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import dirichletbc,Function,FunctionSpace,locate_dofs_topological,Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from ufl import dx,FacetNormal, TestFunctions, split, dot,grad,ds,inner
from params import theta, rho_i, rho_w,L,g,H,nxi,nyi
from constitutive import M,C,h,Q,Re
from post_process import interp
from fem_space import mixed_space
from dolfinx.log import set_log_level, LogLevel

def LeftBoundary(x):
    # Left boundary (inflow/outflow)
    return np.isclose(x[0],-L/2.0)

def get_bcs(V,domain):
    # assign Dirichlet boundary conditions on effective pressure
    N_left = rho_i*g*H
    facets_l = locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)   
    dofs_l = locate_dofs_topological(V.sub(1), domain.topology.dim-1, facets_l)
    bc_l = dirichletbc(PETSc.ScalarType(N_left), dofs_l,V.sub(1))
    bcs = [bc_l]
    return bcs

def weak_form(V,domain,sol,sol_n,z_b,q_in,moulin,dt):
    # define functions
    (b,N,q) = split(sol)           # solution
    (b_,N_,q_) = TestFunctions(V)  # test functions
    (b_n,N_n,q_n) = split(sol_n)   # sol at previous timestep

    # define variables for time integration of db/dt equation
    b_theta = theta*b + (1-theta)*b_n
    q_theta = theta*q + (1-theta)*q_n
    N_theta = theta*N + (1-theta)*N_n
    h_theta = h(N_theta,z_b)

    n_ = FacetNormal(domain)
    
    # weak form for gap height evolution (db/dt) equation:
    F_b = (b-b_n - dt*( M(q_theta,h_theta)/rho_i - C(b_theta,N_theta)))*b_*dx

    # weak form for water flux divergence div(q) equation:
    F_N = -dot(Q(b,h(N,z_b), Re(q_n)),grad(N_))*dx + ((1/rho_i-1/rho_w)*M(q,h(N,z_b)) - C(b,N)-moulin)*N_*dx
    
    # inflow natural/Neumann BC on the water flux:
    F_bdry = dot(q_in,n_)*N_*ds 
    
    # weak form of water flux definitionL
    F_q = inner((q - Q(b,h(N,z_b),Re(q_n))),q_)*dx

    # sum all weak forms:
    F = F_b + F_N + F_q + F_bdry
    return F

def solve_pde(domain,sol_n,z_b,q_in,moulin,dt):
        # solves the hydrology problem for (b,N,q)

        # Define function spaces
        V = mixed_space(domain)

        # # Define boundary conditions 
        bcs = get_bcs(V,domain)

        # define weak form
        sol = Function(V)
        F =  weak_form(V,domain,sol,sol_n,z_b,q_in,moulin,dt)

        set_log_level(LogLevel.ERROR)

        # # set initial guess for Newton solver
        sol.sub(0).interpolate(sol_n.sub(0))
        sol.sub(1).interpolate(sol_n.sub(1))
        sol.sub(2).sub(0).interpolate( sol_n.sub(2).sub(0))
        sol.sub(2).sub(1).interpolate( sol_n.sub(2).sub(1))

        # Solve for sol = (b,N,q)
        problem = NonlinearProblem(F, sol, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        n, converged = solver.solve(sol)

        # bound gap height below by small amount (1mm)
        V0 = FunctionSpace(domain, ("CG", 1))
        b_temp = Function(V0)
        b_temp.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
        b_temp.x.array[b_temp.x.array<1e-3] = 1e-3
        sol.sub(0).interpolate(b_temp)

        assert(converged)
   
      
        return sol

def time_stepping(domain,initial,timesteps,z_b,q_in,moulin):
    # solve the hydrology problem given:
    # domain: the computational domain
    # initial: initial conditions 
    # timesteps: time array
    # z_b: bed elevation function
    # q_in: inflow conditions on domain boundary
    # moulin: water input source term

    # *see example.ipynb for an example of how to set these

    # The solver returns...

    nt = np.size(timesteps)
    dt = np.abs(timesteps[1]-timesteps[0])

    # create arrays for saving solution
    b = np.zeros((nt,nxi,nyi))
    N = np.zeros((nt,nxi,nyi))
    qx = np.zeros((nt,nxi,nyi))
    qy = np.zeros((nt,nxi,nyi))

    V = mixed_space(domain)
    sol_n = Function(V)
    sol_n.sub(0).interpolate(initial.sub(0))
    sol_n.sub(1).interpolate(initial.sub(1))
    sol_n.sub(2).sub(0).interpolate(initial.sub(2).sub(0))
    sol_n.sub(2).sub(1).interpolate(initial.sub(2).sub(1))

    V0 = FunctionSpace(domain, ("CG", 1))

    # # time-stepping loop
    for i in range(nt):
        print('time step '+str(i+1)+' out of '+str(nt)+' \r',end='')


        if i>0:
            dt = np.abs(timesteps[i]-timesteps[i-1])
    
        # solve the compaction problem for sol = N
        sol = solve_pde(domain,sol_n,z_b,q_in,moulin,dt)

        # save the solutions as numpy arrays
        b_int = Function(V0)
        N_int = Function(V0)
        qx_int = Function(V0)
        qy_int = Function(V0)

        b_int.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
        N_int.interpolate(Expression(sol.sub(1), V0.element.interpolation_points()))
        qx_int.interpolate(Expression(sol.sub(2).sub(0), V0.element.interpolation_points()))
        qy_int.interpolate(Expression(sol.sub(2).sub(1), V0.element.interpolation_points()))

        b[i,:,:] = interp(b_int.x.array,domain)
        N[i,:,:] = interp(N_int.x.array,domain)
        qx[i,:,:] = interp(qx_int.x.array,domain)
        qy[i,:,:] = interp(qy_int.x.array,domain)   

        # set solution at previous time step
        sol_n.sub(0).interpolate(sol.sub(0))
        sol_n.sub(1).interpolate(sol.sub(1))
        sol_n.sub(2).sub(0).interpolate(sol.sub(2).sub(0))
        sol_n.sub(2).sub(1).interpolate(sol.sub(2).sub(1))

    return b,N,qx,qy

