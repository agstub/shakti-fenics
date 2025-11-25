# model class for initializing, solving, and post-processing
# the subglacial hydrology model
from dolfinx.fem import Expression, Function, Constant, functionspace
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices,locate_entities, meshtags
from basix.ufl import element
from shapely import Point
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from constitutive import Melt,Closure,Head,WaterFlux,Reynolds
import numpy as np
from solver import solve
from output import output_setup,output_process,output_save
from pressure_solver import pressure_solver
import params

#--------------------------------------------------
# helper functions for interpolating various
# data sets onto the mesh:
def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def set_array_slice(obj, attr_path, values):
    arr = get_nested_attr(obj, attr_path)
    arr[:] = values
#--------------------------------------------------

# model input class file
class model:
    def __init__(self, comm, domain):
        # MPI 
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Domain, mesh, function spaces
        self.domain = domain
        self.x = domain.geometry.x[:,0]
        self.y = domain.geometry.x[:,1]
        self.V = functionspace(domain, ("CG", 1))
        self.V_flux = functionspace(domain,element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))) 
        self.mask_dofs = self.ghost_mask(self.V.dofmap.index_map) 
        self.mask_cells = self.ghost_mask(self.domain.topology.index_map(self.domain.topology.dim))
        
        # outflow boundary for prescribing Dirichlet condition
        self.OutflowBoundary = None
        
        # inflow boundary for prescribing Neumann condition
        self.InflowBoundary = None
        
        # bounding box for interpolating data onto mesh
        buffer = self.get_buffer()
        self.bounds = [self.x.min()-buffer,self.x.max()+buffer,
                       self.y.min()-buffer,self.y.max()+buffer]
        
        # BC options
        self.outflow_on = True                  # allow outflow from domain
        self.storage_on = True                  # turn on water storage in lake

        self.model_config = {}

        # Physical input functions
        self.z_b = Function(self.V)             # bed elevation [m]
        self.z_s = Function(self.V)             # surface elevation [m]
        self.G = Function(self.V)               # geothermal heat flux [W/m^2]
        self.inputs = Function(self.V)          # water inputs to bed (moulins) [m/s]
        self.b_init = Function(self.V)          # initial gap height [m]
        self.N_init = Function(self.V)          # initial effective pressure [Pa]
        self.qx_init = Function(self.V)         # initial water flux x direction [m^2/s]
        self.qy_init = Function(self.V)         # initial water flux x direction [m^2/s]
        self.lake_bdry = Function(self.V)       # lake boundary function (1=within lake, 0=outside lake)
        self.N_bdry = 0.0                       # effective pressure condition at outflow boundary [Pa]
        self.b_min = 1.0e-5                     # minimum gap height [m]     
        self.b_max = 1e3                        # maximum gap height [m]
        
        # water inflow
        self.q_in = 0.0
        
        # solution functions
        self.N = Function(self.V)               # effective pressure [Pa]
        self.q = Function(self.V_flux)          # water discharge [m^2/s]
        self.b = Function(self.V)               # gap height [m]
        self.qx = Function(self.V)              # x-component of q
        self.qy = Function(self.V)              # y-component of q
        self.N_n = Function(self.V)             # N at previous timestep
        self.storage = Function(self.V)         # storage function: 1=storage; 0=no-storage
        
        # related expressions for interpolating solutions
        self.q_expr = None
        self.qx_expr = None
        self.b_expr = None
        self.melt_n_expr = None
        
        # output arrays
        self.b_arr = None
        self.N_arr = None
        self.qx_arr = None
        self.qy_arr = None
        
        # melt rate at previous time step from Warburton et al. (2024)
        # melt rate formulation
        self.melt_n = Function(self.V)

        # lake outline GeoDataFrame for defining boundary function
        self.outline = None

        # Output names
        self.lake_name = None
        self.results_name = None
        self.setup_name = None
        
        # time stepping & frequency for saving files
        self.timesteps = None                   # number of timesteps in the model
        self.nt_save = None                     # temporal frequency of saving solution
        self.nt_check = None                    # how often to make checkpoint saves
        self.j = 0                              # time index for saving solution
        
        self.max_coldstarts = 500               # max number of time to start Newton at zero
                                                # before trying a warm start again
        
        # boundary coordinates for plotting boundaries
        self.boundary_coords = None             # whole boundary coordinates
        self.outflow_coords = None              # outflow boundary coordinates
        
        # some boundary facets
        self.facets_outflow = None              # boundary facets at outflow
        self.bdry_facets = None                 # all boundary facets

        # physical parameters:
        self.g = params.g                       # gravitational acceleration [m/s^2] 
        self.rho_i = params.rho_i               # ice density [kg/m^3] 
        self.rho_w = params.rho_w               # density of water [kg/m^3] 
        self.nu = params.nu                     # water viscosity [m^2/s]
        self.Lh = params.Lh                     # latent heat [J/kg]  
        self.omega = params.omega               # dimensionless parameter in water discharge law (laminar-turbulent transition)
        self.n = params.n                       # Glen's flow law parameter [dimensionless]
        self.A = params.A                       # Glen's flow law coefficient [P^-n s^-1]

    def set_lake_bdry(self,outline):
        # set lake boundary dolfinx Function from a GeoDataFrame (outline input)
        for j in range(self.lake_bdry.x.array.size):
            point = Point(self.domain.geometry.x[j,0],self.domain.geometry.x[j,1])
            self.lake_bdry.x.array[j] = outline.geometry.contains(point).iloc[0]
        self.lake_bdry.x.scatter_forward()

    def interp_data(self, var_name, x_d, y_d, f):
        # interpolate various data sets onto the finite element mesh
        # Subset grid and data
        x_sub = x_d[(x_d >= self.bounds[0]) & (x_d <= self.bounds[1])]
        y_sub = y_d[(y_d >= self.bounds[2]) & (y_d <= self.bounds[3])]
        f_sub = f[np.ix_(
            (y_d >= self.bounds[2]) & (y_d <= self.bounds[3]),
            (x_d >= self.bounds[0]) & (x_d <= self.bounds[1]))]

        # Interpolation
        f_interp = RegularGridInterpolator((x_sub, y_sub), f_sub.T, bounds_error=False, fill_value=None)
        points = np.column_stack((self.x, self.y))
        values = f_interp(points)

        # Dynamically assign to array and call scatter_forward
        set_array_slice(self, f"{var_name}.x.array", values)
        get_nested_attr(self, f"{var_name}.x").scatter_forward()
        return f_interp
    
    def get_buffer(self):
        # create buffer for interpolating data to ensure 
        # that domain is covered by data 
        x_bfr, y_bfr = 0, 0
        x__ = self.comm.gather(self.x[self.mask_dofs],root=0)
        y__ = self.comm.gather(self.y[self.mask_dofs],root=0)
        if self.rank == 0:
            x__ = np.unique(np.concatenate(x__))
            y__ = np.unique(np.concatenate(y__))
            x_bfr = 10*np.max(np.diff(x__)) 
            y_bfr= 10*np.max(np.diff(y__)) 
        self.comm.barrier()    
        x_bfr, y_bfr = self.comm.bcast(x_bfr, root=0), self.comm.bcast(y_bfr, root=0)
        return np.max([x_bfr, y_bfr])
    
    def ghost_mask(self, index_map):
        # mask ghosts (e.g., dofs or cells) given index map
        ghosts = index_map.ghosts
        global_to_local = index_map.global_to_local
        ghosts_local = global_to_local(ghosts)
        size_local = index_map.size_local
        num_ghosts = index_map.num_ghosts
        mask = np.ones(size_local+num_ghosts,dtype=bool)
        mask[ghosts_local] = False
        return mask
    
    def save_dofmap(self):
        # save global dofmap for reconstructing mesh
        # and plotting solutions
        
        # Extract the local geometry dofmap for owned cells
        local_geom_dofmap = self.domain.geometry.dofmap

        # Access the index map for geometry dofs
        imap = self.domain.geometry.index_map()

        # Build local-to-global mapping for coordinate dofs
        local_to_global = np.empty(imap.size_local + imap.num_ghosts, dtype=np.int32)
        local_to_global[:imap.size_local] = np.arange(*imap.local_range)
        local_to_global[imap.size_local:] = imap.ghosts

        # Map local dofs in the geometry dofmap to global indices
        global_geom_dofmap = local_to_global[local_geom_dofmap]
        all_dofmaps = self.comm.gather(global_geom_dofmap[self.mask_cells], root=0)
        if self.rank == 0:
            full_global_dofmap = np.concatenate(all_dofmaps)            
            np.save(self.results_name+'/dofmap.npy',full_global_dofmap)
    
    def solve(self):
        # solve the hydrology problem
        solve(self)
        
    def solvers_setup(self):
        # interpolate initial conditions
        self.b.interpolate(self.b_init) 
        self.N_n.interpolate(self.N_init)
        self.q.sub(0).interpolate(self.qx_init)
        self.q.sub(1).interpolate(self.qy_init)    
    
        # create dolfinx expressions for interpolating water flux
        self.q_expr = Expression(WaterFlux(self.b,Head(self.N,self.z_b,self.z_s), Reynolds(self.q)), self.V_flux.element.interpolation_points())  

        # initialize time step
        self.dt = Constant(self.domain, 0.1*np.abs(self.timesteps[1]-self.timesteps[0]))
        
        # we update b by interpolating this expression:
        self.b_expr = Expression(self.b + self.dt*(Melt(self.q,Head(self.N,self.z_b,self.z_s),self.G,self.b,self.melt_n)/self.rho_i - Closure(self.b,self.N)),self.V.element.interpolation_points())

        # we use this expression for computing melt rate at previous time step:
        self.melt_n_expr = Expression(Melt(self.q,Head(self.N,self.z_b,self.z_s),self.G,self.b,self.melt_n),self.V.element.interpolation_points())

        # define storage function based on model configuration
        if self.storage_on == False:
            # turns off storage term by setting lake boundary function to zero
            # in the weak form 
            self.storage = Function(self.V)
        else:
            # else, storage is allowed within the lake boundary
            self.storage = self.lake_bdry
        
        # define the solver for the effective pressure PDE
        self.pressure_solver = pressure_solver(self)
    
    def output_setup(self):
        # output initialization
        output_setup(self)
    
    def output_process(self):
        # put finite element solutions (dofs) into
        # numpy arrays at each time step
        output_process(self)
        
    def output_save(self):
        # save solution arrays arrays
        output_save(self)
    
    def get_boundary_coords(self):
        # obtain boundary coordinates for plotting
        self.facets_outflow = locate_entities_boundary(self.domain, self.domain.topology.dim-1, self.OutflowBoundary)
        self.facets_inflow = locate_entities_boundary(self.domain, self.domain.topology.dim-1, self.InflowBoundary)
        self.bdry_facets = exterior_facet_indices(self.domain.topology)
        self.boundary_coords = []
        self.outflow_coords = []
        self.inflow_coords = []

        for f in self.bdry_facets:
            # Get vertices of this facet
            vertices = self.domain.topology.connectivity(1, 0).links(f)
            coords = self.domain.geometry.x[vertices]
            self.boundary_coords.append(coords)
            
        for f in self.facets_outflow:
            # Get vertices of this facet
            vertices = self.domain.topology.connectivity(1, 0).links(f)
            coords = self.domain.geometry.x[vertices]
            self.outflow_coords.append(coords)
            
        for f in self.facets_inflow:
            # Get vertices of this facet
            vertices = self.domain.topology.connectivity(1, 0).links(f)
            coords = self.domain.geometry.x[vertices]
            self.inflow_coords.append(coords)
    
    def mark_boundary(self):
        # Assign markers to each boundary segment (except the upper surface).
        # "This is used at each time step to update the markers"
        # NOTE: we shouldn't need to update the markers every timesetep unless
        #       grounding line is migrating...
        # Boundary marker numbering convention:
        # 1 - Inflow boundary
        # 2 - Outflow boundary
        boundaries = [(1, lambda x: self.InflowBoundary(x)),
                      (2, lambda x: self.OutflowBoundary(x))]
        facet_indices, facet_markers = [], []
        fdim = self.domain.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = locate_entities(self.domain, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        return facet_tag
    
    def init_from_results(self,results,t):
        # results = directory name of results
        # t = time in days of results to initialize from
        nodes_x = np.load(results+'/nodes_x.npy')
        nodes_y = np.load(results+'/nodes_y.npy')
        t_ = np.load(results+'/t.npy')
        i = np.argmin(t-t_/86400)
        
        # nodes of current mesh
        points_mesh = np.column_stack((self.x, self.y))
        
        # nodes of results mesh
        points_results = np.zeros((nodes_x.size,2))
        points_results[:,0] = nodes_x
        points_results[:,1] = nodes_y
        del nodes_x, nodes_y
        
        # interpolate N initial condition
        N_0 = np.load(results+'/N.npy')
        N_interp = LinearNDInterpolator(points_results,N_0[i])
        self.N_init.x.array[:] = N_interp(points_mesh)
        self.N_init.x.scatter_forward()
        del N_0, N_interp
        self.comm.barrier()
        
        # interpolate b initial  condition
        b_0 = np.load(results+'/b.npy')
        b_interp = LinearNDInterpolator(points_results,b_0[i])
        self.b_init.x.array[:] = b_interp(points_mesh)
        self.b_init.x.scatter_forward()
        del b_0, b_interp
        self.comm.barrier()
        
        # interpolate qx initial condition
        qx_0 = np.load(results+'/qx.npy')
        qx_interp = LinearNDInterpolator(points_results,qx_0[i])
        self.qx_init.x.array[:] = qx_interp(points_mesh)
        self.qx_init.x.scatter_forward()
        del qx_0, qx_interp
        self.comm.barrier()
        
        # interpolate qy initial condition
        qy_0 = np.load(results+'/qy.npy')
        qy_interp = LinearNDInterpolator(points_results,qy_0[i])
        self.qy_init.x.array[:] = qy_interp(points_mesh)
        self.qy_init.x.scatter_forward()
        del qy_0, qy_interp
        self.comm.barrier()
        
        
        
        