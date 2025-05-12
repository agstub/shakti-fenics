# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,dirichletbc,Function,locate_dofs_topological,Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, TestFunctions, split, dot,grad,inner
from params import rho_i, rho_w,g
from constitutive import Melt,Closure,Head,WaterFlux,Reynolds
from dolfinx.log import set_log_level, LogLevel
import sys
import os
import shutil
from pathlib import Path

def get_bcs(md):
    # assign Dirichlet boundary conditions on effective pressure
    facets_outflow = locate_entities_boundary(md.domain, md.domain.topology.dim-1, md.OutflowBoundary)   
    dofs_outflow = locate_dofs_topological(md.V.sub(1), md.domain.topology.dim-1, facets_outflow)
    bc_outflow = dirichletbc(PETSc.ScalarType(md.N_bdry), dofs_outflow,md.V.sub(1))
    bcs = [bc_outflow]
    return bcs
        
def weak_form(md,sol,sol_n,melt_n,lake_bdry,dt):
    # define functions
    b,N,q = split(sol)             # solution
    b_,N_,q_ = TestFunctions(md.V) # test functions
    b_n,N_n,q_n = split(sol_n)     # sol at previous timestep    

    Re = Reynolds(q_n)
    head_n = Head(N_n,md.z_b,md.z_s)
    head = Head(N,md.z_b,md.z_s)
    water_flux = WaterFlux(b,Head(N,md.z_b,md.z_s), Re)

    # lake term is analogous to englacial storage
    lake_storage = lake_bdry*(1/(rho_w*g*dt))*(N-N_n)
    
    # weak form for gap height evolution (db/dt) equation:
    F_b = (b-b_n - dt*(Melt(q_n,head_n,md.G,b_n,melt_n)/rho_i - Closure(b_n,N_n)))*b_*dx 

    # weak form for water flux divergence div(q) equation:
    F_N = -dot(water_flux,grad(N_))*dx + ((1/rho_i-1/rho_w)*Melt(q,head,md.G,b_n,melt_n) - Closure(b,N)-lake_storage-md.inputs)*N_*dx
    
    # weak form of water flux definitionL
    F_q = inner(q - water_flux,q_)*dx

    # sum all weak form components to obtain residual:
    F = F_b + F_N + F_q 
    return F

def pde_solver(md,sol,sol_n,melt_n,lake_bdry,dt):
        # solves the hydrology problem for (b,N,q)

        # # Define boundary conditions 
        bcs = get_bcs(md)
        
        # define weak form
        F =  weak_form(md,sol,sol_n,melt_n,lake_bdry,dt)     

        # # set initial guess for Newton solver
        sol.sub(0).interpolate(sol_n.sub(0))
        sol.sub(1).interpolate(sol_n.sub(1))
        sol.sub(2).sub(0).interpolate(sol_n.sub(2).sub(0))
        sol.sub(2).sub(1).interpolate(sol_n.sub(2).sub(1))        

        # Solve for sol = (b,N,q)
        problem = NonlinearProblem(F, sol, bcs=bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        return solver

def solve(md):
    # solve the hydrology problem given:
    # domain: the computational domain
    # initial: initial conditions 
    # timesteps: time array
    # z_b: bed elevation function
    # z_s: surface elevation function
    # q_in: inflow conditions on domain boundary
    # inputs: water input source term

    # *see {repo root}/setup/setup_example.py for an example of how to set these

    # The solution is saved in a directory {repo root}/results/resultsname:
    # b = subglacial gap height (m)
    # qx = subglacial water flux [x component] (m^2/s)
    # qy = subglacial water flux [y component] (m^2/s)
    # N = effective pressure (Pa)
    
    # set dolfinx log output to desired level
    set_log_level(LogLevel.WARNING)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    error_code = 0      # code for catching io errors

    nt = np.size(md.timesteps)
    dt_ = 0.1*np.abs(md.timesteps[1]-md.timesteps[0])
    dt = Constant(md.domain, dt_)

    # create masks to handle ghost points to avoid saving 
    # duplicate dof's in parallel runs 
    mask = md.mask
    
    # save nodes so that in post-processing we can create a
    # parallel-to-serial mapping between dof's for plotting
    nodes_x = comm.gather(md.domain.geometry.x[:,0][mask],root=0)
    nodes_y = comm.gather(md.domain.geometry.x[:,1][mask],root=0)

    comm.Barrier()
    # create arrays for saving solution
    if rank == 0:
        try:
            os.makedirs(md.resultsname,exist_ok=False)
        except FileExistsError:
            print(f"Error: Directory '{md.resultsname}' already exists.\nChoose another name in setup file or delete this directory.")  
            error_code = 1
   
    comm.Barrier()    
    error_code = comm.bcast(error_code, root=0)
    
    if error_code == 1:
        sys.exit(1)

    if rank == 0:
        # some io setup
        parent_dir = str((Path(__file__).resolve()).parent.parent)
        nodes_x = np.concatenate(nodes_x)
        nodes_y = np.concatenate(nodes_y)
        nti = int(nt/md.nt_save)
        t_i = np.linspace(0,md.timesteps.max(),nti)
        nd = md.V0.dofmap.index_map.size_global
        
        # arrays for solution dof's at each timestep
        b = np.zeros((nti,nd))
        N = np.zeros((nti,nd))
        qx = np.zeros((nti,nd))
        qy = np.zeros((nti,nd))
        
        np.save(md.resultsname+'/t.npy',t_i)
        np.save(md.resultsname+'/nodes_x.npy',nodes_x)
        np.save(md.resultsname+'/nodes_y.npy',nodes_y)

        # copy setup file into results directory to for plotting/post-processing
        # and to keep record of input 
        shutil.copy(parent_dir+'/setups/{}.py'.format(md.setup_name), md.resultsname+'/{}.py'.format(md.setup_name))
        j = 0 # index for saving results at nt_save time intervals

    # define solution function at previous timestep (sol_n) 
    # and set initial conditions
    sol_n = Function(md.V)
    sol_n.sub(0).interpolate(md.initial.sub(0))
    sol_n.sub(1).interpolate(md.initial.sub(1))
    sol_n.sub(2).sub(0).interpolate(md.initial.sub(2).sub(0))
    sol_n.sub(2).sub(1).interpolate(md.initial.sub(2).sub(1))

    # define solution at current timestep (sol)
    sol = Function(md.V)
    
    # create piecewise linear functions for saving solution
    b_int = Function(md.V0)
    N_int = Function(md.V0)
    qx_int = Function(md.V0)
    qy_int = Function(md.V0)
    
    # function used for bounding gap height below
    b_bound = Function(md.V0)
    
    if md.storage == False:
        # turn off storage term by setting lake boundary function to zero
        # in the weak form if desired
        lake_bdry = Function(md.V0)
    else:
        lake_bdry = md.lake_bdry
        
    # melt rate at previous time step for Warburton et al. (2024)
    # melt rate formulation
    melt_n = Function(md.V0)
        
    # define pde solver
    solver = pde_solver(md,sol,sol_n,melt_n,lake_bdry,dt)

    # define expression for computing melt rate at previous time step
    melt_n_expr = Expression(Melt(sol_n.sub(2),Head(sol_n.sub(1),md.z_b,md.z_s),md.G,sol_n.sub(0),melt_n),md.V0.element.interpolation_points())

    # time-stepping loop
    for i in range(nt):

        if rank == 0:
            print('time step '+str(i+1)+' out of '+str(nt)+' \r',end='')
            sys.stdout.flush()

        if i>0:
            dt_ = np.abs(md.timesteps[i]-md.timesteps[i-1])
            dt.value = dt_
    
        # solve the hydrology problem for sol = (b,N,q)
        niter, converged = solver.solve(sol)
        assert (converged)
        
        if converged == True:
            # bound gap height below by small amount
            # this value influences the flood amplitude
            b_bound.interpolate(Expression(sol.sub(0), md.V0.element.interpolation_points()))
            b_bound.x.array[b_bound.x.array<md.b_min] = md.b_min
            b_bound.x.scatter_forward()
            sol.sub(0).interpolate(b_bound)
        
        if converged == False:
            break

        if i % md.nt_save == 0:
            # interpolate solution onto the piecewise linear functions
            b_int.interpolate(Expression(sol.sub(0), md.V0.element.interpolation_points()))
            N_int.interpolate(Expression(sol.sub(1), md.V0.element.interpolation_points()))
            qx_int.interpolate(Expression(sol.sub(2).sub(0), md.V0.element.interpolation_points()))
            qy_int.interpolate(Expression(sol.sub(2).sub(1), md.V0.element.interpolation_points()))

            # mask out the ghost points and gather
            b__ = comm.gather(b_int.x.array[mask],root=0)
            N__ = comm.gather(N_int.x.array[mask],root=0)
            qx__ = comm.gather(qx_int.x.array[mask],root=0)
            qy__ = comm.gather(qy_int.x.array[mask],root=0)

            if rank == 0:
                # save the dof's as numpy arrays
                b[j,:] = np.concatenate(b__)
                N[j,:] = np.concatenate(N__)
                qx[j,:] = np.concatenate(qx__)
                qy[j,:] = np.concatenate(qy__)

                np.save(md.resultsname+'/b.npy',b)
                np.save(md.resultsname+'/N.npy',N)
                np.save(md.resultsname+'/qx.npy',qx)
                np.save(md.resultsname+'/qy.npy',qy)
                j += 1
 
        # set solution at previous time step
        sol_n.x.array[:] = sol.x.array
        sol_n.x.scatter_forward()
        
        # update melt rate at previous time step
        melt_n.interpolate(melt_n_expr)        
     
    return 