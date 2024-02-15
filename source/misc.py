# def RightBoundary(x):
#     # Right boundary (inflow/outflow)
#     return np.isclose(x[0],L/2.0)

# def TopBoundary(x):
#     # Top boundary (inflow/outflow)
#     return np.isclose(np.abs(x[1]),W/2.0)

# def BottomBoundary(x):
#     # Top boundary (inflow/outflow)
#     return np.isclose(x[1],-W/2.0)



# facets_r = locate_entities_boundary(domain, domain.topology.dim-1, RightBoundary)   
# dofs_r = locate_dofs_topological(V.sub(2).sub(1), domain.topology.dim-1, facets_r)
# bc_r = dirichletbc(PETSc.ScalarType(0), dofs_r,V.sub(2).sub(1))

# facets_v = locate_entities_boundary(domain, domain.topology.dim-1, TopBoundary)   
# dofs_v = locate_dofs_topological(V.sub(2).sub(0), domain.topology.dim-1, facets_v)
# bc_v = dirichletbc(PETSc.ScalarType(0), dofs_v,V.sub(2).sub(0))
