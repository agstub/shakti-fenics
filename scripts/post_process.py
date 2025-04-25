# script for mapping dofs with parallel ordering to serial ordering
# used for plotting the results
import numpy as np
# from dolfinx.fem import locate_dofs_topological
# from dolfinx.mesh import locate_entities_boundary
import matplotlib.pyplot as plt 

def dofs_to_serial(nodes_parallel,nodes_serial):
    inds = np.abs(nodes_parallel-nodes_serial)<1 
    inds = inds[:,0]*inds[:,1]
    mismatch = np.where(inds==False)[0]
    map_dofs = np.arange(nodes_parallel.shape[0])
    for j in mismatch:
        map_dofs[j] = np.where( (np.abs(nodes_parallel[:,0] - nodes_serial[j,0])<1)&(np.abs(nodes_parallel[:,1] - nodes_serial[j,1])<1) )[0]
    return map_dofs