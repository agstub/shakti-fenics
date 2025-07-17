# scripts for mapping dofs with parallel ordering to serial ordering
# and masking ghost points
import numpy as np

def dofs_to_serial(nodes_parallel,nodes_serial):
    tol = 1e-2
    inds = np.abs(nodes_parallel-nodes_serial)<1 
    inds = inds[:,0]*inds[:,1]
    mismatch = np.where(inds==False)[0]
    map_dofs = np.arange(nodes_parallel.shape[0])
    for j in mismatch:
        map_dofs[j] = np.where( (np.abs(nodes_parallel[:,0] - nodes_serial[j,0])<tol)&(np.abs(nodes_parallel[:,1] - nodes_serial[j,1])<tol) )[0]
    return map_dofs