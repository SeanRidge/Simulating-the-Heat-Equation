import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 110 # thermal diffusivity
length = 50 # mm
time = 5 # s
nodes = 200

dx = length / (nodes - 1)
dt = 0.5 * dx**2 / a # stability limit for explicit scheme: a*dt/dx^2 <= 0.5
t_nodes = int(time / dt)
r = a * dt / dx**2 # = 0.5

u = np.zeros(nodes) + 20 # plate initially at 20 °C

# Boundary conditions (held fixed)

u[0] = 100
u[-1] = 100

# Plotting
fig, axis = plt.subplots()

pcm = axis.pcolormesh([u], cmap = plt.cm.inferno, vmin=0, vmax=100)
plt.colorbar(pcm, ax=axis, label='Temperature (°C)')
axis.set_xlim(0, nodes)
axis.set_yticks([])
axis.set_xlabel('Node')

# simulation

counter = 0

while counter < t_nodes:

    w = u.copy()

    for i in range(1, nodes - 1):
        w[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])

    u = w
    counter += 1

    # update the plot every few steps so the animation is watchable
    if counter % 2 == 0:
        pcm.set_array(u)
        axis.set_title(f't = {counter * dt:.2f} s')
        plt.pause(0.01)

plt.show()