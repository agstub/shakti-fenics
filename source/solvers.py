# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,dirichletbc,Function,functionspace,locate_dofs_topological,Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from ufl import dx,FacetNormal, TestFunctions, split, dot,grad,ds,inner
from params import theta, rho_i, rho_w,g
from constitutive import M,C,h,Q,Re,potential,storage
from fem_space import mixed_space
from dolfinx.log import set_log_level, LogLevel
import sys

np.set_printoptions(threshold=sys.maxsize)

def get_bcs(V,domain,N_bdry,OutflowBoundary):
    # assign Dirichlet boundary conditions on effective pressure
    facets_outflow = locate_entities_boundary(domain, domain.topology.dim-1, OutflowBoundary)   
    dofs_outflow = locate_dofs_topological(V.sub(1), domain.topology.dim-1, facets_outflow)
    bc_outflow = dirichletbc(PETSc.ScalarType(N_bdry), dofs_outflow,V.sub(1))
    bcs = [bc_outflow]
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

    # boundary inflow tinkering! notes:
    head0 = h(0*z_b,z_b,z_s) #neglecting effective pressure at boundary
    b0 = 0*z_b + 1e-2        # using b_n doesn't converge 
    rey0 = 1000              # Re(q_n) also works, seems to give slightly lower values
    q_in  = Q(b0,head0,rey0) #approximate flux at boundary: should be small if we choose
                             # a drainage divide and N variations are small

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

def solve_pde(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,N_bdry,OutflowBoundary,dt):
        # solves the hydrology problem for (b,N,q)

        # # Define boundary conditions 
        bcs = get_bcs(V,domain,N_bdry,OutflowBoundary)

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

def solve(resultsname,domain,initial,timesteps,z_b,z_s,q_in,inputs,N_bdry,OutflowBoundary,nt_save):
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

    V0 = functionspace(domain, ("CG", 1))

    # create masks to handle ghost points to avoid saving 
    # duplicate dof's 
    ghosts = V0.dofmap.index_map.ghosts
    global_to_local = V0.dofmap.index_map.global_to_local
    ghosts_local = global_to_local(ghosts)
    size_local = V0.dofmap.index_map.size_local
    num_ghosts = V0.dofmap.index_map.num_ghosts
    mask = np.ones(size_local+num_ghosts,dtype=bool)
    mask[ghosts_local] = False

    p,p_norm_ = potential(z_b,z_s)
    
    # save nodes so that in post-processing we can create a
    # parallel-to-serial mapping between dof's for plotting
    nodes_x = comm.gather(domain.geometry.x[:,0][mask],root=0)
    nodes_y = comm.gather(domain.geometry.x[:,1][mask],root=0)

    # create arrays for saving solution
    if rank == 0:
        nodes_x = np.concatenate(nodes_x)
        nodes_y = np.concatenate(nodes_y)
        nti = int(nt/nt_save)
        nd = V0.dofmap.index_map.size_global
        b = np.zeros((nti,nd))
        N = np.zeros((nti,nd))
        qx = np.zeros((nti,nd))
        qy = np.zeros((nti,nd))
        store = np.zeros((nti,nd))
        t_i = np.linspace(0,timesteps.max(),nti)
        np.save('./'+resultsname+'/t.npy',t_i)
        np.save('./'+resultsname+'/nodes_x.npy',nodes_x)
        np.save('./'+resultsname+'/nodes_y.npy',nodes_y)

        j = 0

    V = mixed_space(domain)
    sol_n = Function(V)
    sol_n.sub(0).interpolate(initial.sub(0))
    sol_n.sub(1).interpolate(initial.sub(1))
    sol_n.sub(2).sub(0).interpolate(initial.sub(2).sub(0))
    sol_n.sub(2).sub(1).interpolate(initial.sub(2).sub(1))

    sol = Function(V)

    solver = solve_pde(V,domain,sol,sol_n,z_b,z_s,q_in,inputs,N_bdry,OutflowBoundary,dt)

    # # time-stepping loop
    j = 0
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
            b_temp = Function(V0)
            b_temp.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
            b_temp.x.array[b_temp.x.array<1e-3] = 1e-3
            sol.sub(0).interpolate(b_temp)
        
        if converged == False:
            break

        if i % nt_save == 0:
            # create piecewise linear functions for saving solution
            b_int = Function(V0)
            N_int = Function(V0)
            qx_int = Function(V0)
            qy_int = Function(V0)
            storage_int = Function(V0)

            # interpolate solution onto the piecewise linear functions
            b_int.interpolate(Expression(sol.sub(0), V0.element.interpolation_points()))
            N_int.interpolate(Expression(sol.sub(1), V0.element.interpolation_points()))
            qx_int.interpolate(Expression(sol.sub(2).sub(0), V0.element.interpolation_points()))
            qy_int.interpolate(Expression(sol.sub(2).sub(1), V0.element.interpolation_points()))
            storage_int.interpolate(Expression(storage(p_norm_), V0.element.interpolation_points()))

            # mask out the ghost points and gather
            b__ = comm.gather(b_int.x.array[mask],root=0)
            N__ = comm.gather(N_int.x.array[mask],root=0)
            qx__ = comm.gather(qx_int.x.array[mask],root=0)
            qy__ = comm.gather(qy_int.x.array[mask],root=0)
            storage__ = comm.gather(storage_int.x.array[mask],root=0)

            if rank == 0:
                # save the dof's as numpy arrays
                b[j,:] = np.concatenate(b__)
                N[j,:] = np.concatenate(N__)
                qx[j,:] = np.concatenate(qx__)
                qy[j,:] = np.concatenate(qy__)
                store[j,:] = np.concatenate(storage__)

                np.save('./'+resultsname+'/b.npy',b)
                np.save('./'+resultsname+'/N.npy',N)
                np.save('./'+resultsname+'/qx.npy',qx)
                np.save('./'+resultsname+'/qy.npy',qy)
                np.save('./'+resultsname+'/storage.npy',store)
                j += 1

        # set solution at previous time step
        sol_n.x.array[:] = sol.x.array

    return 
