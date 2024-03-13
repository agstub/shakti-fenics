# all physical and numerical parameters are set here.
import numpy as np

# time integration:
theta = 0.0      # time integration parameter:
                 # 0 = Forward Euler, 0.5 = trapezoidal
                 # 1 = Backward Euler

resultsname = 'results'        
        
#spatial discretization:
H = 500            # ice thickness (m) [uniform examples]
L =  20*H          # length of domain
W =  20*H          # width of domain

nx,ny = 128,128    # number of elements in x and y directions

# arrays for interpolation and plotting:
nxi,nyi = nx,ny    # number of interpolation points for plotting
x_i = np.linspace(-0.5*L,0.5*L,num=nxi)
y_i = np.linspace(-0.5*W,0.5*W,num=nyi)
X,Y = np.meshgrid(x_i,y_i)

# physical parameters:
g = 9.81        # gravitational acceleration [m/s^2]
rho_i = 917     # ice density [kg/m^3]
rho_w = 1000    # density of water [kg/m^3]
G = 0.05        # geothermal heat flux (W/m^2)
nu = 1.787e-6   # water viscosity (m^2 / s)
Lh = 3.34e5     # latent heat (J/kg)

omega = 1e-3    # dimensionless parameter in water discharge 
n = 3           # Glen's flow law parameter
A = 2.24e-24    # Glen's flow law coefficient
