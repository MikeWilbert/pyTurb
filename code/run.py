import pyTurb
import time as clock

N = 1024
k_a = 0.5
k_f = 20
dk_f = 1.
c_res = 1.5
# eps = 1.
# eps = k_f**(-2)

t_end = 40.
t_print = 1.

t = 0.
t_out = 0.
out_stats = 0
num_stats = 10

out_dir = "/home/fs1/mw/Turbulence/2D_Turbulence/test"

pyTurb.init(N, k_a, k_f, dk_f, c_res, out_dir)

pyTurb.print_vtk()
start_time = clock.time()
while(t < t_end):
  
  pyTurb.step()
  
  dt = pyTurb.dt
  t  = pyTurb.t
  
  t_out += dt  
  
  # if (out_stats > num_stats):
  #   pyTurb.print_stats()
  #   out_stats -= num_stats
  
  # if(t_out>t_print):
    
  #   pyTurb.print_vtk()
  #   pyTurb.print_spectrum()
  #   t_out -= t_print
  
  #   out_stats += 1
    
duration = clock.time() - start_time
print('Duration =', duration, '[s]')
