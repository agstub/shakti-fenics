# This is a wrapper for solving the SHAKTI hydrology problem from command line with MPI
# See setup.py for exanples of model setup options like bed and surface geometry,
# meltwater inputs, mesh creation, etc...

from solvers import solve
import sys
import importlib
sys.path.insert(0, '../setups')

# import model setup module from command line argument
setup_module = importlib.import_module(sys.argv[1])

# add model setup name to module
setup_module.setup_name = sys.argv[1]

# solve the problem
# results are saved in a 'results' directory
solve(setup_module)


