import numpy as np

def init(N_, k_a_, k_f_, c_res_, eps_):
  
  global N, k_a, k_f, c_res, eps
  global k_max, nu, k_nu, alpha
  global L, dx, dk, dt, t
  global W, W_F, force_W, force_W_F
  global x_val, y_val
  global kx, ky, k2, k2_inv
  
  N     = N_
  k_a   = k_a_
  k_f   = k_f_
  c_res = c_res_
  eps   = eps_
  
  # ~ k_max = float(N) / 3. # 2/3 dealiasing
  k_max = float(N) * 0.4 # filter delasing
  nu    = c_res**2 * eps**(1./3.) * k_f**(2./3.) * k_max**(-2.)
  k_nu  = k_max / c_res
  alpha = eps**(1./3.) * k_a**(2./3.)
  
  L  = 2.*np.pi
  dx = L/N
  dk = 1.
  dt = 0.
  t  = 0.
  
  W_F       = np.zeros((N,N), dtype=complex)
  force_W   = np.zeros((N,N), dtype=complex)
  force_W_F = np.zeros((N,N), dtype=complex)
  
  xx = np.linspace(0., L, num=N, endpoint=False)
  x_val, y_val = np.meshgrid(xx,xx)
  
  kk = dk*np.concatenate( ( np.arange(0,N//2), np.arange(-N//2,0) ))
  kx, ky = np.meshgrid(kk,kk)
  k2 = kx**2+ky**2
  k2[0][0] = 1.
  k2_inv = 1./k2
  k2[0][0] = 0.
  
  setup()
  print_scales()
  
def setup():
  
  global W_F, W
  global x_val, y_val
  global alpha, nu
  global force_on
  
  # Taylor-Green
  # ~ nu = 0.01
  # ~ alpha = 0.
  # ~ force_on = False
  
  # ~ W   = ( np.cos( x_val ) + np.cos( y_val ) ) 
  # ~ W_F = np.fft.fft2(W)
  
  # double shear layer
  # ~ delta = 0.05
  # ~ sigma = 15./np.pi
  # ~ nu = 0.0001
  # ~ alpha = 0.
  # ~ force_on = False
  
  # ~ W                = delta * np.cos(x_val) - sigma * np.cosh(sigma* ( y_val - 0.5*np.pi ) )**(-2.)
  # ~ W[y_val > np.pi] = delta * np.cos(x_val[y_val > np.pi]) + sigma * np.cosh(sigma* ( 1.5*np.pi - y_val[y_val > np.pi] ) )**(-2.)
  # ~ W_F = np.fft.fft2(W)

  # zero (self-evolving turbulence)
  np.zeros((N,N), dtype=complex)
  force_on = True

def dealias(IN_F):
  global k2, kx, ky
  global N
  
  # 2/3
  # ~ IN_F[np.abs(kx) > float(N)/3.] = 0.
  # ~ IN_F[np.abs(ky) > float(N)/3.] = 0.
  
  # Filter
  exp_filter = np.exp( -36. * (np.abs(kx)/(0.5*N))**36 ) * np.exp( -36. * (np.abs(ky)/(0.5*N))**36 )
  IN_F *= exp_filter

def grad(IN):
  global kx, ky
  
  return (1.j*kx*IN), (1.j*ky*IN)

def calc_RHS(Win_F, dt_):
  
  global force_W_F
  global nu, alpha
  global k2_inv, k2
  
  # advection stream function
  psi_F = Win_F * k2_inv; psi_F[0][0] = 0.
  
  gradW_x_F  , gradW_y_F   = grad(Win_F)
  gradPsi_x_F, gradPsi_y_F = grad(psi_F)
  
  dealias(gradW_x_F)
  dealias(gradW_y_F)
  dealias(gradPsi_x_F)
  dealias(gradPsi_y_F)
  
  gradW_x   = np.fft.ifft2(gradW_x_F)
  gradW_y   = np.fft.ifft2(gradW_y_F)
  gradPsi_x = np.fft.ifft2(gradPsi_x_F)
  gradPsi_y = np.fft.ifft2(gradPsi_y_F)
  
  RHS_W_ = (gradW_x * gradPsi_y - gradW_y * gradPsi_x)
  
  RHS_W_F_ = np.fft.fft2(RHS_W_)
  dealias(RHS_W_F_)
  
  # Forcing  
  RHS_W_F_ += force_W_F
  
  # linear fricition
  RHS_W_F_ -= alpha*Win_F
  
  # analytische Diffusion
  RHS_W_F_ *= np.exp(+nu *k2*dt_)
  
  return RHS_W_F_
  
def calc_dt():
  global W_F
  global kx, ky, k2_inv
  global dx, dt
  
  Ux_F = + 1.j * ky * k2_inv * W_F; Ux_F[0][0] = 0.
  Uy_F = - 1.j * kx * k2_inv * W_F; Uy_F[0][0] = 0.
  
  Ux = np.fft.ifft2(Ux_F)
  Uy = np.fft.ifft2(Uy_F)
  
  dt = np.sqrt(3) / ( np.pi * np.amax( (1.+np.abs(Ux)/dx) + (1.+np.abs(Uy)/dx) ) )
  
def calc_force():
  global kx, ky, k2_inv
  global force_W_F
  global k_f, eps
  
  dk_f = 0.5
  
  if (force_on==False):
    return
  
  # Fourier Space
  # ~ force_W_F = np.random.randn(N,N) + 1.j * np.random.randn(N,N)
  
  # ~ index = ( ( k2 > (k_f+dk_f)**2 ) | ( k2 < (k_f-dk_f)**2 )  )
  # ~ force_W_F[index] = 0.
  
  # ~ force_W = np.fft.ifft2(force_W_F)
  # ~ force_W = force_W.real
  # ~ force_W_F = np.fft.ifft2(force_W)
  
  # ~ # strength
  # ~ Ux_F =  1.j*ky*k2_inv*force_W_F; Ux_F[0][0] = 0.
  # ~ Uy_F = -1.j*kx*k2_inv*force_W_F; Uy_F[0][0] = 0.
  # ~ force_energy = np.sqrt( 0.5*np.sum( np.abs(Ux_F)**2 + np.abs(Uy_F)**2 ) / N**4 )
  # ~ force_W_F *= np.sqrt(eps/dt) / force_energy
  
  # Real Space
  F_R = np.zeros((N,N), dtype=complex)
  
  for ix in range(0, N//2+1):
    for iy in range(0, N//2+1):
    
      i2 = ix**2+iy**2
      if( ( i2 <= (k_f+dk_f)**2 ) & ( i2 >= (k_f-dk_f)**2 ) ):
        
        F_R += i2**(-0.25) * np.cos( x_val * ix + np.random.rand()*2.*np.pi ) * np.cos( y_val * iy + np.random.rand()*2.*np.pi )
  
  force_W_F = np.fft.fft2(F_R)
  
  # strength
  Ux_F =  1.j*ky*k2_inv*force_W_F; Ux_F[0][0] = 0.
  Uy_F = -1.j*kx*k2_inv*force_W_F; Uy_F[0][0] = 0.
  force_energy = np.sqrt( 0.5* np.sum( np.abs(Ux_F)**2 + np.abs(Uy_F)**2 ) / N**4 )
  force_W_F *= np.sqrt(eps/dt) / force_energy
  
def step():
  global t, dt
  global W_F
  global nu
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
  
def print_scales():
  
  global k_a, k_f
  global k_max, nu, k_nu, alpha
  
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

def get_fields():
  
  global W_F
  
  W = np.fft.ifft2(W_F).real
  
  return W, W_F
  
