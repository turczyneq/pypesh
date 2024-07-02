import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
U = 1.0  # Example value for U
R = 1.0  # Example value for R

# Create a grid of points
r = np.linspace(0, 5, 100)
z = np.linspace(-5, 5, 100)
R_grid, Z_grid = np.meshgrid(r, z)

# Calculate the expression on the grid
squared_dist = R_grid**2 + Z_grid**2
expression = 0.5 * U * R_grid**2 * (1 - 1.5 * R / np.sqrt(squared_dist) + 0.5 * (R / np.sqrt(squared_dist))**3)

# Create the contour plot with a single contour line at value 0.05
plt.figure(figsize=(8, 8))
toplt = plt.contour(R_grid, Z_grid, expression, [0.005])
#plt.clabel(toplt, inline=True, colors='blue')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt


# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2

# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z, [0.5])
# ax.clabel(CS, inline=True, fontsize=10)
# ax.set_title('Simplest default with labels')

# plt.show()