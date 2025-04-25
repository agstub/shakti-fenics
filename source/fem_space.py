# options for finite element spaces are set here
# and related functions to determine ghost points
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace
import numpy as np

def mixed_space(domain):
    # function space for solution (b,N,q) where
    # b = gap height
    # N = effective pressure
    # q = water discharge
    P1 = element('P',domain.basix_cell(),1)
    P1_vec = element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))
    V = functionspace(domain,mixed_element([P1,P1,P1_vec]))  
    return V

def vector_space(domain):
    # function space for water input vector q_in
    P1_vec = element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))
    V = functionspace(domain,P1_vec)  
    return V

def ghost_mask(V0):
    ghosts = V0.dofmap.index_map.ghosts
    global_to_local = V0.dofmap.index_map.global_to_local
    ghosts_local = global_to_local(ghosts)
    size_local = V0.dofmap.index_map.size_local
    num_ghosts = V0.dofmap.index_map.num_ghosts
    mask = np.ones(size_local+num_ghosts,dtype=bool)
    mask[ghosts_local] = False
    return mask