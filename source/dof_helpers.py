# scripts for mapping dofs with parallel ordering to serial ordering
# and masking ghost points
import numpy as np

def dofs_to_serial(nodes_parallel,nodes_serial):
    inds = np.abs(nodes_parallel-nodes_serial)<1 
    inds = inds[:,0]*inds[:,1]
    mismatch = np.where(inds==False)[0]
    map_dofs = np.arange(nodes_parallel.shape[0])
    for j in mismatch:
        map_dofs[j] = np.where( (np.abs(nodes_parallel[:,0] - nodes_serial[j,0])<1)&(np.abs(nodes_parallel[:,1] - nodes_serial[j,1])<1) )[0]
    return map_dofs

def ghost_mask(V):
    ghosts = V.dofmap.index_map.ghosts
    global_to_local = V.dofmap.index_map.global_to_local
    ghosts_local = global_to_local(ghosts)
    size_local = V.dofmap.index_map.size_local
    num_ghosts = V.dofmap.index_map.num_ghosts
    mask = np.ones(size_local+num_ghosts,dtype=bool)
    mask[ghosts_local] = False
    return mask