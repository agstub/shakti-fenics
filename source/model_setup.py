from dolfinx.fem import Function, functionspace
from basix.ufl import element
from shapely import Point
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from solvers import solve

def get_nested_attr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj

def set_array_slice(obj, attr_path, values):
    arr = get_nested_attr(obj, attr_path)
    arr[:] = values

# model input class file
class model_setup:
    def __init__(self, comm, domain):
        # mpi 
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Domain, mesh, function spaces
        self.domain = domain
        self.x = domain.geometry.x[:,0]
        self.y = domain.geometry.x[:,1]
        self.V = functionspace(domain, ("CG", 1))
        self.V_flux = functionspace(domain,element('P',domain.basix_cell(),1,shape=(domain.geometry.dim,))) 
        self.mask = self.ghost_mask(self.V) 
        self.OutflowBoundary = None
        
        # create bounding box for interpolating data
        buffer = self.get_buffer()
        self.bounds = [self.x.min()-buffer,self.x.max()+buffer,
                       self.y.min()-buffer,self.y.max()+buffer]
        
        # BC options
        self.outflow_on = True                  # allow outflow from domain
        self.storage_on = True                  # turn on water storage in lake

        # Physical input functions
        self.z_b = Function(self.V)             # bed elevation [m]
        self.z_s = Function(self.V)             # surface elevation [m]
        self.G = Function(self.V)               # geothermal heat flux [W/m^2]
        self.inputs = Function(self.V)          # water inputs to bed (moulins) [m/s]
        self.b_init = Function(self.V)          # initial gap height [m]
        self.N_init = Function(self.V)          # initial effective pressure [Pa]
        self.q_init = Function(self.V_flux)     # initial water flux [m^2/s]
        self.lake_bdry = Function(self.V)       # lake boundary function (1=within lake, 0=outside lake)
        self.N_bdry = 0.0                       # effective pressure condition at outflow boundary [Pa]
        self.b_min = 1.0e-5                     # minimum gap height [m]     

        # lake outline GeoDataFrame for defining boundary function
        self.outline = None

        # Output names
        self.lake_name = None
        self.results_name = None
        self.setup_name = None
        
        # time stepping & frequency for saving files
        self.timesteps = None
        self.nt_save = None
        self.nt_check = None

    def set_lake_bdry(self,outline):
        for j in range(self.lake_bdry.x.array.size):
            point = Point(self.domain.geometry.x[j,0],self.domain.geometry.x[j,1])
            self.lake_bdry.x.array[j] = outline.geometry.contains(point).iloc[0]
        self.lake_bdry.x.scatter_forward()

    def interp_data(self, var_name, x_d, y_d, f):
        # Subset grid and data
        x_sub = x_d[(x_d >= self.bounds[0]) & (x_d <= self.bounds[1])]
        y_sub = y_d[(y_d >= self.bounds[2]) & (y_d <= self.bounds[3])]
        f_sub = f[np.ix_(
            (y_d >= self.bounds[2]) & (y_d <= self.bounds[3]),
            (x_d >= self.bounds[0]) & (x_d <= self.bounds[1])
        )]

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
        x__ = self.comm.gather(self.x[self.mask],root=0)
        y__ = self.comm.gather(self.y[self.mask],root=0)
        if self.rank == 0:
            x__ = np.unique(np.concatenate(x__))
            y__ = np.unique(np.concatenate(y__))
            x_bfr = 10*np.max(np.diff(x__)) 
            y_bfr= 10*np.max(np.diff(y__)) 
        self.comm.barrier()    
        x_bfr, y_bfr = self.comm.bcast(x_bfr, root=0), self.comm.bcast(y_bfr, root=0)
        return np.max([x_bfr, y_bfr])
    
    def ghost_mask(self, V):
        ghosts = V.dofmap.index_map.ghosts
        global_to_local = V.dofmap.index_map.global_to_local
        ghosts_local = global_to_local(ghosts)
        size_local = V.dofmap.index_map.size_local
        num_ghosts = V.dofmap.index_map.num_ghosts
        mask = np.ones(size_local+num_ghosts,dtype=bool)
        mask[ghosts_local] = False
        return mask
    
    def solve(self):
        solve(self)