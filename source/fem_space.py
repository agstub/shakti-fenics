# options for finite element spaces are set here
from ufl import FiniteElement, MixedElement
from dolfinx.fem import FunctionSpace

def mixed_space(domain):
    # function space for solution (b,N,q) where
    # b = gap height
    # N = effective pressure
    # q = water discharge
    P1 = FiniteElement('P',domain.ufl_cell(),1)
    element = MixedElement([P1,P1,[P1,P1]])
    V = FunctionSpace(domain,element)  
    return V

def vector_space(domain):
    # function space for water input vector q_in
    P1 = FiniteElement('P',domain.ufl_cell(),1)    
    element = MixedElement([P1,P1])
    V = FunctionSpace(domain,element)  
    return V