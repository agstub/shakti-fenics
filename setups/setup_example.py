# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')
import numpy as np
from params import rho_i, rho_w, g
from pathlib import Path
from netCDF4 import Dataset
from model import model
from dolfinx.fem import Expression
from dolfinx.mesh import create_rectangle,CellType
from constitutive import BackgroundGradient
from ufl import dot

def initialize(comm):
    # select lake from inventory and set geographic bounds
    lake_name = 'example' 
    
    # Define mesh (see create_mesh.ipynb notebook for example)
    nx,ny = 128,128
    H = 500            # ice thickness (m) [uniform examples]
    L =  20*H          # length of domain
    W =  20*H          # width of domain
    p0 = [-0.5*L,-0.5*W]
    p1 = [0.5*L,0.5*W]
    domain = create_rectangle(comm,[p0,p1], [nx, ny],cell_type=CellType.triangle) 
        
    # initialize model object
    md = model(comm,domain)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    md.lake_name = lake_name
    
    # set results name for saving
    md.N_bdry = 0.5*rho_i*g*H # boundary condition function for N at outflow
     
    # define bed geometry 
    bed = lambda x: 0.02*(x[0]+0.5*L) - 100*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2))
    md.z_b.interpolate(bed)

    # define surface elevation 
    surf = lambda x: 0.01*(x[0]+0.5*L) + H  -25*np.exp(1)**(-((x[0]-0.25*W)**2+x[1]**2)/(2e3**2))
    md.z_s.interpolate(surf)
    
    # lake outline from geodataframe, for defining lake boundary
    grad_h0 = BackgroundGradient(md.z_b,md.z_s)
    lake_bdry_expr = np.exp(1)**(-(150*dot(grad_h0,grad_h0)**(0.5))**8)
    md.lake_bdry.interpolate(Expression(lake_bdry_expr, md.V.element.interpolation_points()))
       
    # Geoethermal heat flux: 
    md.G.x.array[:] = 0.05
    md.G.x.scatter_forward()

    # parameters!
    md.b_min = 1.0e-3
    # md.b_max = 1.0e2
    md.q_in = -1.0e-2            #water inflow through inflow boundary 

    # define initial conditions
    md.b_init.x.array[:] = 0.01 #+ np.random.normal(scale=0.005,size=np.size(md.b_init.x.array[:])) 
    md.b_init.x.scatter_forward()
    
    md.N_init.interpolate(lambda x:0.1*md.N_bdry+0*x[0])      
    md.q_init.sub(0).interpolate(lambda x: md.q_in+0*x[0]) 
    md.q_init.sub(1).interpolate(lambda x: 0*x[0])  

    # inflow and outflow boundaries
    md.OutflowBoundary = lambda x: np.isclose(x[0],-0.5*L)
    md.InflowBoundary = lambda x: np.isclose(x[0],0.5*L)
    
    # decide if outflow is allowed or not (default True)
    md.outflow_on = True

    # decide if lake is represented with a storage-type term (default True)
    md.storage_on = True

    # define moulin source term - zero (none) in this example
    q_dist = 2.0e-6
    md.inputs.interpolate(lambda x:  q_dist + 0*x[0] )

    # define time stepping 
    days = 100
    nt_per_day = 100
    t_final = (days/365)*3.154e7
    md.timesteps = np.linspace(0,t_final,int(days*nt_per_day))

    # frequency for saving files
    md.nt_save = 1
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/{md.lake_name}'
    
    return md