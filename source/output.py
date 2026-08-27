# this module contains functions related to processing
# and saving finite element solutions
import numpy as np
from dolfinx.fem import Expression,locate_dofs_topological
import sys, os
import json

def output_setup(md):
    error_code = 0      # code for catching io errors

    # create arrays for saving solution
    if md.rank == 0:
        try:
            os.makedirs(md.results_name,exist_ok=False)
        except FileExistsError:
            print(f"Error: Directory '{md.results_name}' already exists.\nChoose another name in setup file or delete this directory.")  
            error_code = 1
   
    error_code = md.comm.bcast(error_code, root=0)
    
    if error_code == 1:
        sys.exit(1)

    # get mesh nodes
    nodes_x = md.comm.gather(md.x[md.mask_dofs],root=0)
    nodes_y = md.comm.gather(md.y[md.mask_dofs],root=0)
    
    # save global dofmap for plotting 
    md.save_dofmap()
    md.get_boundary_coords()

    # get lake boundary function
    lake_bdry__ = md.comm.gather(md.lake_bdry.x.array[md.mask_dofs],root=0)

    # boundary coordinates
    outflow_coords__ = md.comm.gather(md.outflow_coords,root=0)
    inflow_coords__ = md.comm.gather(md.inflow_coords,root=0)
    boundary_coords__ = md.comm.gather(md.boundary_coords,root=0)
    
    # get bounding box for plotting
    x_min__ = md.comm.gather(md.bounds[0],root=0)
    x_max__ = md.comm.gather(md.bounds[1],root=0)
    y_min__ = md.comm.gather(md.bounds[2],root=0)
    y_max__ = md.comm.gather(md.bounds[3],root=0)
    
    #outflow dofs
    outflow_dofs = locate_dofs_topological(md.V, md.domain.topology.dim-1, md.facets_outflow)
    outflow_dofs__ = md.comm.gather(outflow_dofs,root=0)

    if md.rank == 0:
        # some io setup
        # get boundary coordinates from each process into one list
        boundary_coords_list = []
        for coords in boundary_coords__:
            for xy in coords:
                boundary_coords_list.append(xy)
        
        # get outflow coordinates from each process into one list
        outflow_coords_list = []
        for coords in outflow_coords__:
            for xy in coords:
                outflow_coords_list.append(xy)
        
        # get inflow coordinates from each process into one list
        inflow_coords_list = []
        for coords in inflow_coords__:
            for xy in coords:
                inflow_coords_list.append(xy)
        
        # store some basic model info
        md.model_config.update({"lake_name": md.lake_name, "storage_on": md.storage_on, 
                  "outflow_on": md.outflow_on, "N_bdry": md.N_bdry, 
                  "x_min": np.array(x_min__).min(), 
                  "x_max": np.array(x_max__).max(),
                  "y_min": np.array(y_min__).min(), 
                  "y_max": np.array(y_max__).max()})
        
        with open(md.results_name+"/model_config.json", "w") as f:
            json.dump(md.model_config, f, indent=2)
        
        # number of time steps that are saved, time array for plotting
        nti = int(np.ceil(md.timesteps.size/md.nt_save))
        t_i = np.linspace(0,md.timesteps.max(),nti)
        
        # number of global dofs for each solution
        nd = md.V.dofmap.index_map.size_global
        
        # arrays for solution dof's at each timestep
        md.b_arr = np.zeros((nti,nd))
        md.N_arr = np.zeros((nti,nd))
        md.qx_arr = np.zeros((nti,nd))
        md.qy_arr = np.zeros((nti,nd))
        
        # save some non-time-dependent objects
        np.save(md.results_name+'/t.npy',t_i)
        np.save(md.results_name+'/nodes_x.npy',np.concatenate(nodes_x))
        np.save(md.results_name+'/nodes_y.npy',np.concatenate(nodes_y))
        np.save(md.results_name+'/lake_bdry.npy',np.concatenate(lake_bdry__))
        np.save(md.results_name+'/boundary_coords.npy',boundary_coords_list)
        np.save(md.results_name+'/outflow_coords.npy',outflow_coords_list)
        np.save(md.results_name+'/inflow_coords.npy',inflow_coords_list)
        np.save(md.results_name+'/outflow_dofs.npy',np.concatenate(outflow_dofs__))

        md.j = 0 # index for saving results at nt_save time intervals
    
    # initialize expressions for saving water flux components
    md.qx_expr = Expression(md.q.sub(0), md.V.element.interpolation_points)
    md.qy_expr = Expression(md.q.sub(1), md.V.element.interpolation_points)

def output_process(md):
    # interpolate water flux components for saving
    md.qx.interpolate(md.qx_expr)
    md.qy.interpolate(md.qy_expr)
        
    # mask out the ghost dofs and gather
    b__ = md.comm.gather(md.b.x.array[md.mask_dofs],root=0)
    N__ = md.comm.gather(md.N.x.array[md.mask_dofs],root=0)
    qx__ = md.comm.gather(md.qx.x.array[md.mask_dofs],root=0)
    qy__ = md.comm.gather(md.qy.x.array[md.mask_dofs],root=0)

    if md.rank == 0:
        # save the dof's as numpy arrays
        md.b_arr[md.j,:] = np.concatenate(b__)
        md.N_arr[md.j,:] = np.concatenate(N__)
        md.qx_arr[md.j,:] = np.concatenate(qx__)
        md.qy_arr[md.j,:] = np.concatenate(qy__)

        # update time index for saving
        md.j += 1

def output_save(md):
    # save solution arrays
    if md.rank == 0:
        np.save(md.results_name+f'/b.npy',md.b_arr)
        np.save(md.results_name+f'/N.npy',md.N_arr)
        np.save(md.results_name+f'/qx.npy',md.qx_arr)
        np.save(md.results_name+f'/qy.npy',md.qy_arr)
        
        
        
