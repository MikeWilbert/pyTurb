import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({'font.size':24})
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['figure.figsize'] = (11.,8.)

file_name = "/home/mike/mounts/davinci_scratch/Turbulence/2D_Turbulence/test" + "/spectra.csv"
k_f = 10

data = np.genfromtxt(file_name ,delimiter=',', skip_header=0)
k  = np.array(data[0,:])
E  = np.array(data[1:,:])

k_low  = k[k < k_f]
k_high = k[k > k_f]

fig, ax = plt.subplots()
ax.set_title('Energy Spectrum') 

# ~ ax.loglog( k_low  , 10**(11)*k_low**(-5/3) , c='k', label=r'$k^{5/3}$', ls='--')
# ~ ax.loglog( k_high , 10**(13)*k_high**(-3.) , c='k', label=r'$k^{-3}$')
ax.loglog( k_high , 10**(17)*k_high**(-3.*2) , c='k', label=r'$k^{-3}$')
# ~ ax.loglog( k_high , 10**(12)*k_high**(-3.5) , c='k', label=r'$k^{-3}$')
ax.loglog(k, E[-1], label=r'$E(k)$', c='tab:orange')

# ~ for i in range(10,E.shape[0]):
  # ~ ax.loglog(k, E[i], label=r'$E(k)$')

ax.legend()
ax.grid()
plt.show()
