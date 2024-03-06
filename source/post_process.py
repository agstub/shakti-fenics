import numpy as np
from params import nxi, nyi
from scipy.interpolate import griddata

def interp(f,x,y):
    # interpolate a scalar function f(x,y) at the
    # spatial resolution defined by (nxi,nyi) in params
    # return a numpy array
    points = (x,y)
    x_i = np.linspace(x.min(),x.max(),num=nxi)
    y_i = np.linspace(y.min(),y.max(),num=nyi)
    X_i,Y_i = np.meshgrid(x_i,y_i)
    points_i = (X_i,Y_i)
    f_i = griddata(points=points,values=f, xi=points_i,fill_value=0)
    return f_i
