from dolfinx.fem import Function, functionspace
from dof_helpers import ghost_mask
from basix.ufl import element
from shapely import Point

# model class file
class model:
    def __init__(self, comm, domain):
        # mpi context
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Domain, mesh, function spaces
        self.domain = domain
        self.V = functionspace(domain, ("CG", 1))
        self.V_flux = functionspace(domain,element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))) 
        self.mask = ghost_mask(self.V) 
        self.OutflowBoundary = None
        
        # BC options
        self.outflow_on = None
        self.storage_on = None

        # Physical inputs
        self.z_b = Function(self.V)
        self.z_s = Function(self.V)
        self.G = Function(self.V)
        self.inputs = Function(self.V)
        self.b_init = Function(self.V)
        self.N_init = Function(self.V)
        self.q_init = Function(self.V_flux)
        self.lake_bdry = Function(self.V)
        self.N_bdry = 0.0
        self.b_min = 0.0
        
        # cosmetic / mostly for plotting
        self.outline = None

        # Output / control
        self.lake_name = None
        self.results_name = None
        self.setup_name = None
        self.nt_save = None
        self.nt_check = None
        self.timesteps = None
    
    def set_lake_bdry(self,outline):
        for j in range(self.lake_bdry.x.array.size):
            point = Point(self.domain.geometry.x[j,0],self.domain.geometry.x[j,1])
            self.lake_bdry.x.array[j] = outline.geometry.contains(point).iloc[0]
        self.lake_bdry.x.scatter_forward()