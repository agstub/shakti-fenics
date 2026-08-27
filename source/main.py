# This program solves the SHAKTI hydrology problem from command line 
# e.g., run via: mpirun -np 4 python ../source/main.py setup_cooke2
# See setup_cooke2.py for exanples of model setup options like bed and surface geometry,
# meltwater inputs, geothermal heat flux, etc...

import sys, os
import importlib
from mpi4py import rc

# note: explicit mpi init and os._exit below needed on macOS (26.5) to prevent hang on finalize...
rc.initialize = False

from mpi4py import MPI

MPI.Init()

sys.path.insert(0, '../setups')

# Set up MPI 
comm = MPI.COMM_WORLD

# import model setup module from command line argument
setup = importlib.import_module(sys.argv[1])

# initialize md with MPI context
md = setup.initialize(comm)

# setup output arrays, etc...
md.output_setup()

# solve the problem, results are saved in a 'results_name' directory
# visualize the solution with the plotting.ipynb notebook
md.solve()

# save the results
md.output_save()

comm.Barrier()

os._exit(0)