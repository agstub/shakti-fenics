# options for finite element spaces are set here
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace

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