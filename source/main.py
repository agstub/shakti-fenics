from params import resultsname
from setup import domain,initial,timesteps,z_b,s_Z,q_in,moulin,nt_save 

# solve the problem
# results are saved in a 'results' directory
solve(resultsname,domain,initial,timesteps,z_b,z_s,q_in,moulin,nt_save)






