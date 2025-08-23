# This file contains the functions needed for solving the subglacial hydrology problem
import numpy as np
from dolfinx.log import set_log_level, LogLevel
import sys

def solve(md):
    # solve the hydrology problem given:
    # domain: the computational domain
    # initial: initial conditions 
    # timesteps: time array
    # z_b: bed elevation function
    # z_s: surface elevation function
    # q_in: inflow conditions on domain boundary
    # inputs: water input source term
    # G: geothermal heat flux

    # *see {repo root}/setup/setup_cooke2.py for an example of how to set these

    # The solution is saved in a directory {repo root}/results/results_name:
    # b = subglacial gap height [m]
    # qx = subglacial water flux [x component] [m^2/s]
    # qy = subglacial water flux [y component] [m^2/s]
    # N = effective pressure [Pa]
    #
    # several other objects are saved (e.g. boundary coordinates, mesh nodes), 
    # see output module or plotting notebook for usage
    
    # set dolfinx log output to desired level
    set_log_level(LogLevel.WARNING)
              
    # define pde solver for N and other setup initialization
    md.solvers_setup()

    # time-stepping loop
    for i in range(md.timesteps.size):
        if md.rank == 0 and (i+1)%10==0:
            print(f"Time step {i+1} of {md.timesteps.size} completed ({(i+1)/md.timesteps.size*100:.1f}%)", end='\r')
            sys.stdout.flush()

        if i>0:
            # update timestep value 
            md.dt.value = np.abs(md.timesteps[i]-md.timesteps[i-1])
    
        # solve for effective pressure (N)
        niter, converged = md.pressure_solver.solve(md.N)
        assert (converged)
        
        if converged == False:
            break
        
        # update water flux (q) via interpolation 
        md.q.interpolate(md.q_expr)
        
        # update melt rate at previous time step
        md.melt_n.interpolate(md.melt_n_expr)        
        
        # update gap height (b) via interpolation
        # (i.e. integrating the db/dt evolution ODE element-wise)
        md.b.interpolate(md.b_expr)
        
        # bound gap height below by small amount
        md.b.x.array[md.b.x.array<md.b_min] = md.b_min
        md.b.x.scatter_forward()
        
        if i % md.nt_save == 0:
            # interpolate and put function dofs into numpy arrays
            md.output_process()
                
            if i % md.nt_check == 0:
            # checkpoint saves: e.g., to not wait until
            # the end of simulation for plotting
                md.output_save()
 
        # set solution at previous time step
        md.N_n.x.array[:] = md.N.x.array
        md.N_n.x.scatter_forward()
    
    return 