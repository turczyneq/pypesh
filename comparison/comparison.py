import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

'''IMPORTING OUR DATA'''
our_data = Path(__file__).parent / "data/our"
our_data = np.loadtxt(our_data)
#change effective surface into sherwood
for n in range(len(our_data)):
    our_data[n,1] = (our_data[n,0]*our_data[n,1])/(4*np.pi)

'''CLASSICAL CLIFT SOLUTION'''
nclift = Path(__file__).parent / "data/clift.txt"
nclift = np.loadtxt(nclift)

def clif(pe):
    return (1/2)*(1+(1+2*pe)**(1/3))

xargs = np.logspace(-1, 5, 300)
aclift = clif(xargs)

'''EXPERIMENTAL DATA OF KUTA et.al., FENG et. al. and KRAMERS et.al.'''
kuta = Path(__file__).parent / "data/kuta.txt"
kuta = np.loadtxt(kuta)

feng = Path(__file__).parent / "data/feng.txt"
feng = np.loadtxt(feng)

kramers = Path(__file__).parent / "data/kramers.txt"
kramers = np.loadtxt(kramers)

'''GIGAPLOT'''

plt.figure(figsize=(8, 5))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Cambria"
})


# Plot Clift data
plt.loglog(xargs, aclift, label='Clift approximation', color='gray', linestyle='--', linewidth=3, zorder = 1)
plt.loglog(nclift[:,0], nclift[:,1], label='Clift solution', color='g', linestyle='--', linewidth=3, zorder = 1)

# Plot our data
plt.loglog(our_data[:, 0], our_data[:, 1], label='Our solution', color='b', linewidth=2, zorder = 2)

# Plot Kuta experimental data
plt.scatter(kuta[:, 0], kuta[:, 1], label='Kuta et al.', color='r', marker='o', s=50, edgecolor='k', zorder = 3)

# Plot Feng experimental data
plt.scatter(feng[:, 0], feng[:, 1], label='Feng et al.', color='orange', marker='^', s=50, edgecolor='k', zorder = 3)

# Plot Kramers experimental data
plt.scatter(kramers[:, 0], kramers[:, 1], label='Kramers et al.', color='pink', marker='s', s=50, edgecolor='k', zorder = 3)

# Logarithmic scale
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.5,10**5)
plt.ylim(0.8,30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Labels and Title
plt.xlabel(r"Peclet Number $\left(Pe\right)$", fontsize=15)
plt.ylabel(r'Sherwood Number $\left(Sh\right)$', fontsize=15)

# Legend
plt.legend(fontsize=15)


# Show plot
plt.show()