import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 110 # thermal diffusivity
length = 50 # mm (square plate)
time = 4 # s
nodes = 50 # nodes per side

dx = length / (nodes - 1)
dt = 0.25 * dx**2 / a # stability limit for 2D explicit scheme: a*dt/dx^2 <= 0.25
t_nodes = int(time / dt)
r = a * dt / dx**2 # = 0.25

u = np.zeros((nodes, nodes)) + 20 # plate initially at 20 °C

# Boundary conditions (held fixed)

u[0, :] = 100 # top edge
u[-1, :] = 100 # bottom edge
u[:, 0] = 100 # left edge
u[:, -1] = 100 # right edge

# Plotting
fig, axis = plt.subplots()

im = axis.imshow(u, cmap=plt.cm.inferno, vmin=0, vmax=100,
                 extent=[0, length, 0, length], origin='lower')
plt.colorbar(im, ax=axis, label='Temperature (°C)')
axis.set_xlabel('x (mm)')
axis.set_ylabel('y (mm)')

# simulation

counter = 0

while counter < t_nodes:

    w = u.copy()

    # 5-point stencil applied to all interior nodes at once
    w[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
        u[2:, 1:-1] + u[:-2, 1:-1] +
        u[1:-1, 2:] + u[1:-1, :-2] -
        4 * u[1:-1, 1:-1]
    )

    u = w
    counter += 1

    # update the plot every few steps so the animation is watchable
    if counter % 20 == 0:
        im.set_data(u)
        axis.set_title(f't = {counter * dt:.3f} s')
        plt.pause(0.01)

plt.show()