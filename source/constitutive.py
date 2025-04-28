# this file contains some constutitve relations for 
# solving the hydrology problem
from params import rho_i,rho_w,g,nu,omega,G,Lh,A,n
from ufl import grad, dot

def Head(N,z_b,z_s):
    # hydraulic head [m] as a function of effective pressure N, 
    # bed elevation z_b, and surface elevation z_s
    return z_b + (rho_i/rho_w)*(z_s-z_b)  - N/(rho_w*g)

def WaterFlux(b,h,Re):
    # water discharge [m^2/s] as a function of gap height b,
    # hydraulic head h, and local Reynolds number Re
    p1 = -(abs(b)**3)*g*grad(h)
    p2 = 12*nu*(1+omega*Re)
    return p1/p2

def Reynolds(q):
    # local Reynolds number [dimensionless]
    return dot(q,q)**0.5/nu

def Melt(q,h):
    # melting rate [kg/ (m^2 s)]
    p = G - rho_w*g*dot(q,grad(h))
    return p/Lh

def Closure(b,N):
    # viscous closure term [m/s]
    return A*b*N*abs(N)**(n-1)

def BackgroundGradient(z_b,z_s):
    # background hydraulic gradient [dimensionless]
    # assumes zero effective pressure (N) gradient
    return grad(Head(0*z_b,z_b,z_s))

def BackgroundPotential(z_b,z_s):
    # background hydraulic potentially [dimensionless]
    # assumes zero effective pressure (N) 
    return rho_w*g*Head(0*z_b,z_b,z_s)
