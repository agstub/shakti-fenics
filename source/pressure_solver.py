# This file contains the functions needed for solving the effective pressure PDE
from dolfinx.fem import dirichletbc,locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, TestFunction, dot,grad
from constitutive import Melt,Closure,Head,WaterFlux,Reynolds

def get_bcs(md):
    # assign Dirichlet boundary conditions on effective pressure
    if md.outflow_on == False:
        # if outflow is not allowed, then we don't prescribe a Dirichlet
        # condition at the boundary (instead, we prescribe zero Neumann flux)
        bcs = []
    else:
        # if outflow is allowed, we prescribe a Dirichlet condition on the 
        # effective pressure 
        facets_outflow = locate_entities_boundary(md.domain, md.domain.topology.dim-1, md.OutflowBoundary)   
        dofs_outflow = locate_dofs_topological(md.V, md.domain.topology.dim-1, facets_outflow)
        bc_outflow = dirichletbc(PETSc.ScalarType(md.N_bdry), dofs_outflow,md.V)
        bcs = [bc_outflow]
    return bcs

def pressure_solver(md):
        # solves a PDE for effective pressure N

        # Define boundary conditions 
        bcs = get_bcs(md)
        
        # define weak form
        N_ = TestFunction(md.V) # test function

        Re = Reynolds(md.q)
        head = Head(md.N,md.z_b,md.z_s)
        water_flux = WaterFlux(md.b,head, Re)

        # lake term is analogous to englacial storage
        lake_storage = md.storage*(1/(md.rho_w*md.g*md.dt))*(md.N-md.N_n)

        # weak form for water flux divergence div(q) equation:
        F = -dot(water_flux,grad(N_))*dx + ((1/md.rho_i-1/md.rho_w)*Melt(md.q,head,md.G,md.b,md.melt_n) - Closure(md.b,md.N)-lake_storage-md.inputs)*N_*dx

        # set initial guess for Newton solver to solution from previous timestep (warm start)
        md.N.interpolate(md.N_n)
        
        petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_monitor_cancel": None,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_stol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        } 
  
        # define solver
        solver = NonlinearProblem(F, md.N, bcs=bcs,petsc_options=petsc_options,petsc_options_prefix="pressure")

        return solver