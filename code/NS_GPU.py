import pyTurb
import time as clock

N = 2048
k_a = 0.002
k_f = 10
c_res = 1.
eps = 1.

t_end = 60.
t_print = 0.5

t = 0.
t_out = 0.
out_stats = 0
num_stats = 100

out_dir = "/home/fs1/mw/Turbulence/2D_Turbulence/test"

pyTurb.init(N, k_a, k_f, c_res, eps, out_dir)

start_time = clock.time()
while(t < t_end):
  
  pyTurb.step()
  
  dt = pyTurb.dt
  t  = pyTurb.t
  
  t_out += dt
  print('time =', t, end='\r')
  
  if (out_stats > num_stats):
    pyTurb.print_stats()
    out_stats -= num_stats
  
  if(t_out>t_print):
    
    pyTurb.print_vtk()
    pyTurb.print_spectrum()
    t_out -= t_print
  
  out_stats += 1
    
duration = clock.time() - start_time
print('Duration =', duration, '[s]')
