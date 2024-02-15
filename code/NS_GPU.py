import pyTurb
import time as clock

N = 256
k_a = 1.
k_f = 12
c_res = 1.
eps = 1.

t_end = 20.
t_print = 0.1

t = 0.
t_out = 0.
out_num = 0

out_dir = "./output"

pyTurb.init(N, k_a, k_f, c_res, eps)

start_time = clock.time()
while(t < t_end):
  
  pyTurb.step()
  
  dt = pyTurb.dt
  t  = pyTurb.t
  
  t_out += dt
  print('time =', t, end='\r')
  
  # ~ calc_energy()
  
  if(t_out>t_print):
    
    file_name = out_dir + '/step_' + str(out_num)
    out_num += 1
    
    pyTurb.print_vtk(file_name)
    t_out -= t_print
    
duration = clock.time() - start_time
print('Duration =', duration, '[s]')

# ~ print_energy()
