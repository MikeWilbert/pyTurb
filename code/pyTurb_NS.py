import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''
ToDo:
- cupy
- Forcing
- Energie-Spektrum
'''

L = 2.*np.pi

''' --- PARAMETERS --- '''
N = 128
eps0 = 1.
k_a = 0.5
k_f = 10
c_res = 1.

k_max = float(N) / 3.
nu = c_res**2 * eps0**(1./3.) * k_f**(2./3.) * k_max**(-2.)
k_nu = k_f**(1./3.) * eps0**(1./6.) * nu**(-1./2.)
alpha = eps0**(1./3.) * k_a**(2./3.)

print('')
print('-----SCALES-----')
print('k_a     =', k_a)
print('k_f     =', k_f)
print('k_nu    =', k_nu)
print('k_max   =', k_max)
print('')
print('l_a/l_f  =', k_f/k_a)
print('l_f/l_nu =', k_nu/k_f)
print('')
print('nu      =', nu)
print('alpha   =', alpha)
print('----------------')
print('')

''' ------------------ '''

# ~ N = 128
# ~ l_f   = L/16.
# ~ dk_f  = 0
# ~ eps0  = 0.2
# ~ alpha = 0.2

# ~ k_f = np.pi/l_f
# ~ eta0 = k_f**2 * eps0
# ~ nu    = ( 4.5*eta0**(1./6.) / N )**2
# ~ Re = np.sqrt(eps0/k_f)/nu
# ~ l_nu = nu**(0.5) / eta0**(1./6.)

# ~ print('')
# ~ print('-----SCALES-----')
# ~ print('nu    =', nu)
# ~ print('l_f   =', l_f)
# ~ print('l_nu  =', l_nu)
# ~ print('k_f   =', k_f)
# ~ print('eta_0 =', eps0)
# ~ print('eta_0 =', eta0)
# ~ print('Re    =', Re)
# ~ print('----------------')
# ~ print('')

dx = L/N
dk = 2.*np.pi/L
dt = 0.01
t = 0.


W      = np.zeros((N,N), dtype=complex)
W_F    = np.zeros((N,N), dtype=complex)
A      = np.zeros((N,N), dtype=complex)
A_F    = np.zeros((N,N), dtype=complex)
force_W = np.zeros((N,N), dtype=complex)
force_A = np.zeros((N,N), dtype=complex)
force_W_F = np.zeros((N,N), dtype=complex)
force_A_F = np.zeros((N,N), dtype=complex)
f_x = np.zeros((N,N), dtype=complex)
f_y = np.zeros((N,N), dtype=complex)

energy_W = []
energy_A = []
eps      = []
time   = []

xx = np.linspace(0., L, num=N, endpoint=False)
x_val, y_val = np.meshgrid(xx,xx)

kk = dk*np.concatenate( ( np.arange(0,N//2), np.arange(-N//2,0) ))
kx, ky = np.meshgrid(kk,kk)
k2 = kx**2+ky**2
k2[0][0] = 1.
k2_inv = 1./k2
k2[0][0] = 0.

def init():
  global t
  global x_val, y_val
  global W, W_F
  global A, A_F
  
  # double shear layer
  # ~ delta = 0.05
  # ~ sigma = 15./np.pi
  
  # ~ W                = delta * np.cos(x_val) - sigma * np.cosh(sigma* ( y_val - 0.5*np.pi ) )**(-2.)
  # ~ W[y_val > np.pi] = delta * np.cos(x_val[y_val > np.pi]) + sigma * np.cosh(sigma* ( 1.5*np.pi - y_val[y_val > np.pi] ) )**(-2.)
  # ~ W_F = np.fft.fft2(W)
  
  # Taylor-Green
  W_F = ( np.cos( x_val ) + np.cos( y_val ) ) 
  
  t = 0
  
def calc_force():
  global kx, ky, k2_inv
  global nu, eta
  global force_W_F, force_A_F
  global f_x, f_y
  global x_val, y_val
  global k_f, dk_f, eps0
  
  # Alvelius
  dk_f = 0
  k_a = k_f - dk_f
  k_b = k_f + dk_f
  
  index = ( ( np.rint(np.sqrt(k2)).astype(int) < k_a ) | ( np.rint(np.sqrt(k2)).astype(int) > k_b ) )
  
  # form
  F = 1.
  
  force_W_F = np.sqrt( F * np.sqrt(k2_inv) ) * np.exp( 2.j*np.pi * np.random.randn(N,N) )
  force_W_F[index] = 0.
  
  force_W = np.fft.ifft2(force_W_F)
  force_W = force_W.real
  force_W_F = np.fft.ifft2(force_W)
  
  # strength
  Ux_F =  1.j*ky*k2_inv*force_W_F; Ux_F[0] = 0.
  Uy_F = -1.j*kx*k2_inv*force_W_F; Uy_F[0] = 0.
  force_energy = np.sqrt( 0.5*np.sum( np.abs(Ux_F)**2 + np.abs(Uy_F)**2 ) / N**4 )
  force_W_F *= np.sqrt(eps0/dt) / force_energy
  
  # random-phase Talor-Green
  # ~ F_R = k_f * np.cos(k_f*x_val + np.random.randn(1)*2.*np.pi) * np.cos(k_f*y_val + np.random.randn(1)*2.*np.pi)
  # ~ force_W_F = np.fft.fft2(F_R)
  
  # ~ Ux_F =  1.j*ky*k2_inv*force_W_F; Ux_F[0] = 0.
  # ~ Uy_F = -1.j*kx*k2_inv*force_W_F; Uy_F[0] = 0.
  # ~ force_energy = np.sqrt( 0.5* np.sum( np.abs(Ux_F)**2 + np.abs(Uy_F)**2 ) / N**4 )
  # ~ force_W_F *= np.sqrt(eps0/dt) / force_energy
  
def grad(IN):
  global kx, ky
  
  return (1.j*kx*IN), (1.j*ky*IN)
  
def curl(IN):
  global kx, ky
  
  return (1.j*ky*IN) , (-1.j*kx*IN)
  
def curl2(INx, INy):
  global kx, ky
  
  return (1.j* (kx*INy-ky*INx) )

def calc_RHS(Win_F, dt_):
  
  global kx, ky, k2_inv, k2
  global nu, eta
  global force_W_F, force_A_F
  global alpha
  
  # advection stream function
  psi_F = Win_F * k2_inv; psi_F[0] = 0.
  
  gradW_x_F, gradW_y_F = grad(Win_F)
  gradPsi_x_F, gradPsi_y_F = grad(psi_F)
  
  dealias(gradW_x_F)
  dealias(gradW_y_F)
  dealias(gradPsi_x_F)
  dealias(gradPsi_y_F)
  
  gradW_x = np.fft.ifft2(gradW_x_F)
  gradW_y = np.fft.ifft2(gradW_y_F)
  gradPsi_x = np.fft.ifft2(gradPsi_x_F)
  gradPsi_y = np.fft.ifft2(gradPsi_y_F)
  
  RHS_W_ = (gradPsi_x * gradW_y - gradPsi_y * gradW_x) 
  
  RHS_W_F_ = np.fft.fft2(RHS_W_)
  dealias(RHS_W_F_)
  
  # Forcing  
  RHS_W_F_ += force_W_F
  
  # linear fricition
  RHS_W_F_ -= alpha*Win_F
  
  # analytische Diffusion
  RHS_W_F_ *= np.exp(+nu *k2*dt_)
  
  return RHS_W_F_
  
def step():
  global t, dt
  global W_F
  global A_F
  global nu, eta
  global k2

  calc_dt()
  
  calc_force()
  
  # Euler (1. Ordnung)
  # ~ RHS1_W, RHS1_A = calc_RHS(W_F, A_F, 0.)
  # ~ W_F = (W_F + dt * RHS1_W) * np.exp(-nu *k2*dt)
  # ~ A_F = (A_F + dt * RHS1_A) * np.exp(-eta*k2*dt)
  
  # Heun (2. Ordnung)
  # ~ RHS1_U, RHS1_B = calc_RHS(W_F, A_F, 0.)
  # ~ W_1 = (W_F + dt * RHS1_U) * np.exp(-nu*k2*dt)
  # ~ A_1 = (A_F + dt * RHS1_B) * np.exp(-nu*k2*dt)
  
  # ~ RHS2_U, RHS2_B = calc_RHS(W_1, A_1, dt)
  # ~ W_F = (W_F + 0.5*dt*RHS1_U) * np.exp(-nu*k2*dt) + 0.5*dt*RHS2_U
  # ~ A_F = (A_F + 0.5*dt*RHS1_B) * np.exp(-nu*k2*dt) + 0.5*dt*RHS2_B
  
  # SSPRK3 (3. Ordnung)
  RHS1_U = calc_RHS(W_F, 0.)
  W_1 = (W_F + dt * RHS1_U) * np.exp(-nu*k2*dt)
  
  RHS2_U = calc_RHS(W_1, dt)
  W_2 = (W_F + 0.25*dt*RHS1_U) * np.exp(-0.5*nu*k2*dt) + 0.25*dt*RHS2_U * np.exp(+0.5*nu*k2*dt)
  
  RHS3_U = calc_RHS(W_2, 0.5*dt)
  W_F = (W_F + 1./6.*dt*RHS1_U) * np.exp(-nu*k2*dt) + 1./6.*dt*RHS2_U + 2./3.*dt*RHS3_U * np.exp(-0.5*nu*k2*dt)
  
  t += dt
  
def dealias(IN_F):
  global k2, kx, ky
  global N
  
  # 2/3
  # ~ IN_F[np.abs(kx) > float(N)/3.] = 0.
  # ~ IN_F[np.abs(ky) > float(N)/3.] = 0.
  
  # Filter
  exp_filter = np.exp( -36. * (np.abs(kx)/(0.5*N))**36 ) * np.exp( -36. * (np.abs(ky)/(0.5*N))**36 )
  IN_F *= exp_filter
  
def calc_dt():
  global W_F, force_W_F
  global kx, ky, k2_inv
  global nu, dt, dx
  
  Ux_F = + 1.j * ky * k2_inv * W_F; Ux_F[0] = 0.
  Uy_F = - 1.j * kx * k2_inv * W_F; Uy_F[0] = 0.
  
  Ux = np.fft.ifft2(Ux_F)
  Uy = np.fft.ifft2(Uy_F)
  
  dt = np.sqrt(3) / ( np.pi * np.amax( (1.+np.abs(Ux)/dx) + (1.+np.abs(Uy)/dx) ) )

def calc_energy():
  global W_F
  global kx, ky, k2_inv
  global nu, dt, dx, t
  
  Ux_F = + 1.j * ky * k2_inv * W_F; Ux_F[0] = 0.
  Uy_F = - 1.j * kx * k2_inv * W_F; Uy_F[0] = 0.
  
  Bx_F, By_F = curl(A_F)
  
  energy_W_ = 0.5 * np.sum( np.abs(Ux_F)**2 + np.abs(Uy_F)**2 ) / N**4 # N**2 wegen Parsevalls Theorem und N**2 wegen Mittelwert 
  energy_A_ = 0.5 * np.sum( np.abs(Bx_F)**2 + np.abs(By_F)**2 ) / N**4 # N**2 wegen Parsevalls Theorem und N**2 wegen Mittelwert 
  
  # ~ print('E =', energy_W_)
  
  return energy_W_, energy_A_
  
def calc_Dissipation():
  global W_F, A_F
  global kx, ky, k2_inv
  global nu, dt, dx, t
  
  Bx_F, By_F = curl(A_F)
  J_F = curl2(Bx_F, By_F)
  
  eps_ =  np.sum( nu*np.abs(W_F)**2) / N**4
  
  # ~ print('eps =', eps_)
  
  return eps_
  
# GRAPHIK
frames = 100000
# ~ cm = 'Spectral'
cm = 'seismic'
fig, ((ax1,ax2)) = plt.subplots(1,2) # einzelnes Anzeigefenster
ax1.set_title(r'$W$')             
ax2.set_title(r'$J$')                                      
im1 = ax1.pcolormesh(W.real, cmap = cm)                
im1.set_clim(vmin=-2., vmax=2.)
im2 = ax2.pcolormesh(np.abs(W_F), cmap = cm)                
im2.set_clim(vmin=-3., vmax=3.)
ax1.invert_yaxis()
ax2.invert_yaxis()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

fig2, ((ax3,ax4)) = plt.subplots(1,2)
line1a, = ax3.plot([], [])
line1b, = ax3.plot([], [])
line2, = ax4.plot([], [])
ax3.set_xlim(0.,40.)
ax3.set_ylim(0.,2.)  
ax3.set_title(r'Energy E')  
ax3.grid()
ax4.set_title(r'Disspiatio $\epsilon$')  
ax4.set_xlim(0.,40.)
ax4.set_ylim(0.,1.)  
ax4.grid()
  
# wird ein Mal am Anfang der Animation geladen
def anim_init():
  global W, W_F
  global A, A_F
  
  init()
  
  title =  "time = 0"
  fig.suptitle(title)
  
  # fields
  W   = np.fft.ifft2(W_F)
  
  #current
  Bx_F, By_F = curl(A_F)
  J_F = curl2(Bx_F, By_F)
  J   = np.fft.ifft2(J_F)
  
  im1.set_array(W.real)
  im2.set_array(J.real)
  im1.set_clim(vmin=np.amin(W.real), vmax=np.amax(W.real))
  im2.set_clim(vmin=np.amin(J.real), vmax=np.amax(J.real))
  
  return im1, im2

# ein Animationschritt
def anim_step(i):
  global W, W_F
  global A, A_F
  global t, N, dt
  global kx, ky, k2, k2_inv
  
  N_sub = 10
  for i in range(N_sub):
    step()
  
  title = "time = " + str(round(t,4)) + ", dt = " + str(round(dt,4)) 
  fig.suptitle(title)

  # fields
  W   = np.fft.ifft2(W_F)
  
  #current
  Bx_F, By_F = curl(A_F)
  J_F = curl2(Bx_F, By_F)
  J   = np.fft.ifft2(J_F)
  
  f_W = np.fft.ifft2(force_W_F)
  f_A = np.fft.ifft2(force_A_F)
  
  # ~ im2.set_array(J.real)
  # ~ im2.set_clim(vmin=np.amin(J.real), vmax=np.amax(J.real))
  
  im1.set_array(W.real)
  im1.set_clim(vmin=np.amin(W.real), vmax=np.amax(W.real))
  # ~ im1.set_clim(vmin=-32, vmax=32)
  
  im2.set_array(np.roll( np.abs(W_F) , (N//2,N//2), axis=(1,0) ) )
  im2.set_clim(vmin=np.amin(np.abs(W_F)), vmax=np.amax(np.abs(W_F)))
  
  # ~ im2.set_array( np.roll( force_W_F.real, (N//2,N//2), axis=(1,0) ) )
  # ~ im2.set_clim(vmin=np.amin(force_W_F.real), vmax=np.amax(force_W_F.real))
  
  return im1, im2
  
def anim_step2(i):
  global W, W_F
  global t, N
  global time, energy
  
  # ~ step()
  
  title = "time = " + str(round(t,4)) + ", dt = " + str(round(dt,4)) 
  fig2.suptitle(title)
  
  E_W, E_A = calc_energy()
  diss = calc_Dissipation()
  
  energy_W.append( E_W )
  energy_A.append( E_A )
  eps.append( diss )
  time.append(t)
  line1a.set_data(time,energy_W)
  line1b.set_data(time,energy_A)
  line2.set_data(time,eps)
  
  return line1a, line1b, line2

anim = FuncAnimation(fig, anim_step, init_func = anim_init, interval = 20, frames = frames, blit = False, repeat=False)
anim2 = FuncAnimation(fig2, anim_step2, interval = 20, frames = frames, blit = False, repeat=False)

plt.show()

# ~ init()
# ~ step()
