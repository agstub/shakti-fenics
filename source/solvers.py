# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,dirichletbc,Function,functionspace,locate_dofs_topological,Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from ufl import dx,FacetNormal, TestFunctions, split, dot,grad,ds,inner,sym
from params import theta, rho_i, rho_w,L,g,H,nxi,nyi, X,Y
from constitutive import M,C,h,Q,Re,potential,storage
from fem_space import mixed_space
from dolfinx.log import set_log_level, LogLevel
import sys
import os
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay


def LeftBoundary(x):
    # Left boundary (inflow/outflow)
    return np.isclose(x[0],-L/2.0)

def get_bcs(V,domain,N_bdry):
    # assign Dirichlet boundary conditions on effective pressure
    facets_l = locate_entities_boundary(domain, domain.topology.dim-1, LeftBoundary)   
    dofs_l = locate_dofs_topological(V.sub(1), domain.topology.dim-1, facets_l)
    bc_l = dirichletbc(PETSc.ScalarType(N_bdry), dofs_l,V.sub(1))
    bcs = [bc_l]
    return bcs

def weak_form(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,dt):
    # define functions
    b,N,q = split(sol)           # solution
    b_,N_,q_ = TestFunctions(V)  # test functions
    b_n,N_n,q_n = split(sol_n)   # sol at previous timestep

    # define variables for time integration of db/dt equation
    b_theta = theta*b + (1-theta)*b_n
    q_theta = theta*q + (1-theta)*q_n
    N_theta = theta*N + (1-theta)*N_n
    h_theta = h(N_theta,z_b,z_s)

    n_ = FacetNormal(domain)

    # tinkering!
    head0 = h(0*z_b,z_b,z_s)
    b0 = 0*z_b + 1e-2 #b_n 
    rey0 = 1e3 #Re(q_n) #1000
    q_in  = Q(b0,head0,rey0)

    # define storage function (0=no storage, 1=perfect storage)
    p,p_norm = potential(z_b,z_s)
    nu = storage(p_norm)

    # define term for lake activity
    lake = nu*(1/(rho_w*g*dt))*(N-N_n)
    
    # weak form for gap height evolution (db/dt) equation:
    F_b = (b-b_n - dt*( M(q_theta,h_theta)/rho_i - C(b_theta,N_theta)))*b_*dx

    # weak form for water flux divergence div(q) equation:
    F_N = -dot(Q(b,h(N,z_b,z_s), Re(q_n)),grad(N_))*dx + ((1/rho_i-1/rho_w)*M(q,h(N,z_b,z_s)) - C(b,N)-lake-inputs)*N_*dx
    
    # inflow natural/Neumann BC on the water flux:
    F_bdry = dot(q_in,n_)*N_*ds 
    
    # weak form of water flux definitionL
    F_q = inner((q - Q(b,h(N,z_b,z_s),Re(q_n))),q_)*dx

    # sum all weak forms:
    F = F_b + F_N + F_q + F_bdry
    return F

def solve_pde(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,N_bdry,dt):
        # solves the hydrology problem for (b,N,q)

        # # Define boundary conditions 
        bcs = get_bcs(V,domain,N_bdry)

        # define weak form
        F =  weak_form(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,dt)

        # # set initial guess for Newton solver
        sol.sub(0).interpolate(sol_n.sub(0))
        sol.sub(1).interpolate(sol_n.sub(1))
        sol.sub(2).sub(0).interpolate(sol_n.sub(2).sub(0))
        sol.sub(2).sub(1).interpolate(sol_n.sub(2).sub(1))

        # Solve for sol = (b,N,q)
        problem = NonlinearProblem(F, sol, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
  
        ksp = solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly" #preonly / cg?
        opts[f"{option_prefix}pc_type"] = "ksp" # ksp ?
        ksp.setFromOptions()

        return solver

def solve(resultsname,domain,initial,timesteps,z_b,z_s,q_in,inputs,N_bdry,nt_save):
    # solve the hydrology problem given:
    # domain: the computational domain
    # initial: initial conditions 
    # timesteps: time array
    # z_b: bed elevation function
    # z_s: surface elevation function
    # q_in: inflow conditions on domain boundary
    # inputs: water input source term

    # *see example.ipynb for an example of how to set these

    # The solution is saved in a directory:
    # b = subglacial gap height (m)
    # qx = subglacial water flux [x component] (m^/s)
    # qy = subglacial water flux [y component] (m^/s)
    # N = effective pressure (Pa)

    set_log_level(LogLevel.WARNING)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nt = np.size(timesteps)
    dt_ = np.abs(timesteps[1]-timesteps[0])

    dt = Constant(domain, dt_)

    points = comm.gather(domain.geometry.x[:,0:2],root=0)

    p,p_norm_ = potential(z_b,z_s)
    
    # create arrays for saving solution
    if rank == 0:
        nti = int(nt/nt_save)
        points = np.concatenate(points)
        b = np.zeros((nti,nyi,nxi))
        N = np.zeros((nti,nyi,nxi))
        qx = np.zeros((nti,nyi,nxi))
        qy = np.zeros((nti,nyi,nxi))
        store = np.zeros((nti,nyi,nxi))
        triang = Delaunay(points) 
        t_i = np.linspace(0,timesteps.max(),nti)
        os.mkdir('./'+resultsname)
        np.save('./'+resultsname+'/t.npy',t_i)
        j = 0

    V = mixed_space(domain)
    sol_n = Function(V)
    sol_n.sub(0).interpolate(initial.sub(0))
    sol_n.sub(1).interpolate(initial.sub(1))
    sol_n.sub(2).sub(0).interpolate(initial.sub(2).sub(0))
    sol_n.sub(2).sub(1).interpolate(initial.sub(2).sub(1))

    V0 = functionspace(domain, ("CG", 1))

    sol = Function(V)

    solver = solve_pde(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,N_bdry,dt)

    # # time-stepping loop
    for i in range(nt):

        if rank == 0:
            print('time step '+str(i+1)+' out of '+str(nt)+' \r',end='')
            sys.stdout.flush()

        if i>0:
            dt_ = np.abs(timesteps[i]-timesteps[i-1])
            dt.value = dt_
    
        # solve the hydrology problem for sol = b,q,N
        n, converged = solver.solve(sol)
        assert (converged)
        
        if converged == True:
            # bound gap height below by small amount (1mm)
            V0 = functionspace(domain, ("CG", 1))
            b_temp = Function(V0)
            b_temp.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
            b_temp.x.array[b_temp.x.array<1e-3] = 1e-3
            sol.sub(0).interpolate(b_temp)
        
        if converged == False:
            break


        if i % nt_save == 0:
            # save the solutions as numpy arrays
            b_int = Function(V0)
            N_int = Function(V0)
            qx_int = Function(V0)
            qy_int = Function(V0)
            storage_int = Function(V0)
    
            b_int.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
            N_int.interpolate(Expression(sol.sub(1), V0.element.interpolation_points()))
            qx_int.interpolate(Expression(sol.sub(2).sub(0), V0.element.interpolation_points()))
            qy_int.interpolate(Expression(sol.sub(2).sub(1), V0.element.interpolation_points()))
            storage_int.interpolate(Expression(storage(p_norm_), V0.element.interpolation_points()))

            b__ = comm.gather(b_int.x.array,root=0)
            N__ = comm.gather(N_int.x.array,root=0)
            qx__ = comm.gather(qx_int.x.array,root=0)
            qy__ = comm.gather(qy_int.x.array,root=0)
            storage__ = comm.gather(storage_int.x.array,root=0)
            
            
            if rank == 0:
                b__ = np.concatenate(b__).ravel()
                N__ = np.concatenate(N__).ravel()
                qx__ = np.concatenate(qx__).ravel()
                qy__ = np.concatenate(qy__).ravel()
                storage__ = np.concatenate(storage__).ravel()

                b[j,:,:] = LinearNDInterpolator(triang, b__)(X,Y)
                N[j,:,:] = LinearNDInterpolator(triang, N__)(X,Y)
                qx[j,:,:] = LinearNDInterpolator(triang, qx__)(X,Y)
                qy[j,:,:] = LinearNDInterpolator(triang, qy__)(X,Y)
                store[j,:,:] = LinearNDInterpolator(triang, storage__)(X,Y)
                
                np.save('./'+resultsname+'/b.npy',b)
                np.save('./'+resultsname+'/N.npy',N)
                np.save('./'+resultsname+'/qx.npy',qx)
                np.save('./'+resultsname+'/qy.npy',qy)
                np.save('./'+resultsname+'/storage.npy',store)
                
                j += 1

        # set solution at previous time step
        sol_n.x.array[:] = sol.x.array

    return 
