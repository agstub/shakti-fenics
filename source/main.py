# This is just a wrapper for solving the problem from command line with MPI
# See setup.py for model setup options like bed and surface geometry,
# meltwater inputs, etc...

from setup import domain,initial,timesteps,z_b,z_s,q_in,inputs,nt_save, N_bdry,OutflowBoundary,resultsname 
from solvers import solve

# solve the problem
# results are saved in a 'results' directory
solve(resultsname,domain,initial,timesteps,z_b,z_s,q_in,inputs,N_bdry,OutflowBoundary,nt_save)


