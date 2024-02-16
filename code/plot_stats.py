import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update({'font.size':24})
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['figure.figsize'] = (11.,8.)

file_name = "/home/mike/Documents/pyturb_2d/output/spectrum" + "/stats.csv"

data = np.genfromtxt(file_name ,delimiter=',', skip_header=0)
t  = np.array(data[:,0])
E  = np.array(data[:,1])
D  = np.array(data[:,2])

fig, ((ax1,ax2)) = plt.subplots(1,2)

ax1.plot(t, E, label=r'$Energy$', c='tab:blue')
ax2.plot(t, D, label=r'$Dissipation$', c='tab:green')

plt.show()
