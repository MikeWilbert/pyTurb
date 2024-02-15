import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyTurb

N = 128
k_a = 0.5
k_f = 10
c_res = 1.
eps = 1.

pyTurb.init(N, k_a, k_f, c_res, eps)

# GRAPHIK
frames = 100000
W, W_F = pyTurb.get_fields()
fig, ((ax1,ax2)) = plt.subplots(1,2) # einzelnes Anzeigefenster
ax1.set_title(r'$W$')             
ax2.set_title(r'$\hat{W}$')                                      
im1 = ax1.pcolormesh(W.real, cmap = 'seismic')                
im1.set_clim(vmin=-2., vmax=2.)
im2 = ax2.pcolormesh(np.abs(W_F), cmap = 'gray')                
im2.set_clim(vmin=-3., vmax=3.)
ax1.invert_yaxis()
ax2.invert_yaxis()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

def anim_step(i):
  
  N_sub = 10
  for i in range(N_sub):
    pyTurb.step()
  
  t  = pyTurb.t
  dt = pyTurb.dt
  title = "time = " + str(round(t,4)) + ", dt = " + str(round(dt,4)) 
  fig.suptitle(title)

  W, W_F = pyTurb.get_fields()
  
  im1.set_array(W)
  im1.set_clim(vmin=np.amin(W.real), vmax=np.amax(W.real))
  
  im2.set_array(np.roll( np.abs(W_F) , (N//2,N//2), axis=(1,0) ) )
  im2.set_clim(vmin=np.amin(np.abs(W_F)), vmax=np.amax(np.abs(W_F)))
  
  return im1, im2
  
anim = FuncAnimation(fig, anim_step, interval = 20, frames = frames, blit = False, repeat=False)
# ~ anim2 = FuncAnimation(fig2, anim_step2, interval = 20, frames = frames, blit = False, repeat=False)

plt.show()

