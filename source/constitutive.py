# this file contains some constutitve relations for 
# the hydrology problem
from params import rho_i,rho_w,g,nu,omega,G,Lh,A,n
from ufl import grad, dot

def h(N,z_b,z_s):
    # hydraulic head as a function of effective pressure N, 
    # bed elevation z_b, and surface elevation z_s
    return z_b + (rho_i/rho_w)*(z_s-z_b)  - N/(rho_w*g)

def Q(b,h,Re):
    # water discharge as a function of gap height b,
    # hydraulic head h, and local Reynolds number Re
    p1 = -(abs(b)**3)*g*grad(h)
    p2 = 12*nu*(1+omega*Re)
    return p1/p2

def Re(q):
    # local Reynolds number
    return dot(q,q)**0.5/nu

def M(q,h):
    # melting term
    p = G - rho_w*g*dot(q,grad(h))
    return p/Lh

def C(b,N):
    # viscous closure term
    return A*b*N*abs(N)**(n-1)

def potential(z_b,z_s):
    # basic hydraulic potential gradient
    # for plotting model setup
    p = grad(z_b + (rho_i/rho_w)*(z_s-z_b))
    return p,dot(p,p)**(0.5)