# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,dirichletbc,Function,locate_dofs_topological,Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, TestFunction, dot,grad
from params import rho_i, rho_w,g
from constitutive import Melt,Closure,Head,WaterFlux,Reynolds
from dolfinx.log import set_log_level, LogLevel
import sys
import os
import shutil
from pathlib import Path

def get_bcs(md):
    # assign Dirichlet boundary conditions on effective pressure
    if md.outflow == False:
        bcs = []
    else:
        facets_outflow = locate_entities_boundary(md.domain, md.domain.topology.dim-1, md.OutflowBoundary)   
        dofs_outflow = locate_dofs_topological(md.V, md.domain.topology.dim-1, facets_outflow)
        bc_outflow = dirichletbc(PETSc.ScalarType(md.N_bdry), dofs_outflow,md.V)
        bcs = [bc_outflow]
    return bcs

def pde_solver(md,N,N_n,b,q,melt_n,lake_bdry,dt):
        # solves the hydrology problem for N

        # # Define boundary conditions 
        bcs = get_bcs(md)
        
        # define weak form
        N_ = TestFunction(md.V) # test function

        Re = Reynolds(q)
        head = Head(N,md.z_b,md.z_s)
        water_flux = WaterFlux(b,head, Re)

        # lake term is analogous to englacial storage
        lake_storage = lake_bdry*(1/(rho_w*g*dt))*(N-N_n)

        # weak form for water flux divergence div(q) equation:
        F = -dot(water_flux,grad(N_))*dx + ((1/rho_i-1/rho_w)*Melt(q,head,md.G,b,melt_n) - Closure(b,N)-lake_storage-md.inputs)*N_*dx

        # # set initial guess for Newton solver
        N.interpolate(N_n)
  
        # Solve for N
        problem = NonlinearProblem(F, N, bcs=bcs)
        solver = NewtonSolver(md.comm, problem)

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

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    
    error_code = 0      # code for catching io errors

    nt = np.size(md.timesteps)
    dt_ = 0.1*np.abs(md.timesteps[1]-md.timesteps[0])
    dt = Constant(md.domain, dt_)
    
    # save nodes so that in post-processing we can create a
    # parallel-to-serial mapping between dof's for plotting
    nodes_x = md.comm.gather(md.domain.geometry.x[:,0][md.mask],root=0)
    nodes_y = md.comm.gather(md.domain.geometry.x[:,1][md.mask],root=0)

    md.comm.Barrier()
    # create arrays for saving solution
    if md.rank == 0:
        try:
            os.makedirs(md.resultsname,exist_ok=False)
        except FileExistsError:
            print(f"Error: Directory '{md.resultsname}' already exists.\nChoose another name in setup file or delete this directory.")  
            error_code = 1
   
    md.comm.Barrier()    
    error_code = md.comm.bcast(error_code, root=0)
    
    if error_code == 1:
        sys.exit(1)

    if md.rank == 0:
        # some io setup
        parent_dir = str((Path(__file__).resolve()).parent.parent)
        nodes_x = np.concatenate(nodes_x)
        nodes_y = np.concatenate(nodes_y)
        nti = int(nt/md.nt_save)
        t_i = np.linspace(0,md.timesteps.max(),nti)
        nd = md.V.dofmap.index_map.size_global
        
        # arrays for solution dof's at each timestep
        b_arr = np.zeros((nti,nd))
        N_arr = np.zeros((nti,nd))
        qx_arr = np.zeros((nti,nd))
        qy_arr = np.zeros((nti,nd))
        
        np.save(md.resultsname+'/t.npy',t_i)
        np.save(md.resultsname+'/nodes_x.npy',nodes_x)
        np.save(md.resultsname+'/nodes_y.npy',nodes_y)

        # copy setup file into results directory to for plotting/post-processing
        # and to keep record of input 
        shutil.copy(parent_dir+'/setups/{}.py'.format(md.setup_name), md.resultsname+'/{}.py'.format(md.setup_name))
        j = 0 # index for saving results at nt_save time intervals

    # define solution function and set initial conditions
    N = Function(md.V)
    q = Function(md.V_flux)
    b = Function(md.V)
    qx = Function(md.V)
    qy = Function(md.V)
    N_n = Function(md.V) # N at previous timestep
    
    # interpolate initial conditions
    b.interpolate(md.b_init) 
    N_n.interpolate(md.N_init)
    q.sub(0).interpolate(md.q_init.sub(0))
    q.sub(1).interpolate(md.q_init.sub(1))    
    
    # create dolfinx expressions for interpolating water flux
    q_expr = Expression(WaterFlux(b,Head(N,md.z_b,md.z_s), Reynolds(q)), md.V_flux.element.interpolation_points())  
    qx_expr = Expression(q.sub(0), md.V.element.interpolation_points())
    qy_expr = Expression(q.sub(1), md.V.element.interpolation_points())
    
    if md.storage == False:
        # turn off storage term by setting lake boundary function to zero
        # in the weak form if desired
        lake_bdry = Function(md.V)
    else:
        lake_bdry = md.lake_bdry
        
    # melt rate at previous time step for Warburton et al. (2024)
    # melt rate formulation
    melt_n = Function(md.V)
            
    # define pde solver for N
    solver = pde_solver(md,N,N_n,b,q,melt_n,lake_bdry,dt)
    
    # interpolate b using expression:
    b_expr = Expression(b + dt*(Melt(q,Head(N,md.z_b,md.z_s),md.G,b,melt_n)/rho_i - Closure(b,N)),md.V.element.interpolation_points())

    # define expression for computing melt rate at previous time step
    melt_n_expr = Expression(Melt(q,Head(N,md.z_b,md.z_s),md.G,b,melt_n),md.V.element.interpolation_points())

    # time-stepping loop
    for i in range(nt):

        if md.rank == 0 and (i+1)%10==0:
            print('time step '+str(i+1)+' out of '+str(nt)+' \r',end='')
            sys.stdout.flush()

        if i>0:
            dt_ = np.abs(md.timesteps[i]-md.timesteps[i-1])
            dt.value = dt_
    
        # solve for effective pressure (N)
        niter, converged = solver.solve(N)
        assert (converged)
        
        if converged == False:
            break
        
        # update water flux (q) via interpolation 
        q.interpolate(q_expr)
        
        # update melt rate at previous time step
        melt_n.interpolate(melt_n_expr)        
        
        # update gap height (b) via interpolation
        b.interpolate(b_expr)
        
        # bound gap height below by small amount
        # note: this value influences flood amplitudes
        b.x.array[b.x.array<md.b_min] = md.b_min
        b.x.scatter_forward()
            
        if i % md.nt_save == 0:
            # interpolate water flux components for saving
            qx.interpolate(qx_expr)
            qy.interpolate(qy_expr)
            
            # mask out the ghost points and gather
            b__ = md.comm.gather(b.x.array[md.mask],root=0)
            N__ = md.comm.gather(N.x.array[md.mask],root=0)
            qx__ = md.comm.gather(qx.x.array[md.mask],root=0)
            qy__ = md.comm.gather(qy.x.array[md.mask],root=0)

            if md.rank == 0:
                # save the dof's as numpy arrays
                b_arr[j,:] = np.concatenate(b__)
                N_arr[j,:] = np.concatenate(N__)
                qx_arr[j,:] = np.concatenate(qx__)
                qy_arr[j,:] = np.concatenate(qy__)
                
                if i % md.nt_check == 0:
                    np.save(md.resultsname+f'/b.npy',b_arr)
                    np.save(md.resultsname+f'/N.npy',N_arr)
                    np.save(md.resultsname+f'/qx.npy',qx_arr)
                    np.save(md.resultsname+f'/qy.npy',qy_arr)

                j += 1
 
        # set solution at previous time step
        N_n.x.array[:] = N.x.array
        N_n.x.scatter_forward()
    
    # post-processing: put time-slices into big arrays
    if md.rank == 0:
        np.save(md.resultsname+f'/b.npy',b_arr)
        np.save(md.resultsname+f'/N.npy',N_arr)
        np.save(md.resultsname+f'/qx.npy',qx_arr)
        np.save(md.resultsname+f'/qy.npy',qy_arr)
    
    return 