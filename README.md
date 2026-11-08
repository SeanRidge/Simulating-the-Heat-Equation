# Finite Difference Heat Equation Solver

Explicit finite difference solvers for the **1D** and **2D heat equation**, with live animated plots of the temperature distribution. Written in Python with NumPy and Matplotlib.

## Contents

- [Overview](#overview)
- [Theory](#theory)
  - [The heat equation](#the-heat-equation)
  - [Discretisation](#discretisation)
  - [The explicit (FTCS) scheme](#the-explicit-ftcs-scheme)
  - [Stability](#stability)
  - [Boundary and initial conditions](#boundary-and-initial-conditions)
  - [Accuracy](#accuracy)
- [How the theory maps to the code](#how-the-theory-maps-to-the-code)


## Overview

The scripts simulate heat conduction through a thin plate (1D: a rod or strip, 2D: a square plate) that starts at a uniform temperature, while its edges are held at a fixed, higher temperature. Heat diffuses inward from the edges, and the animation shows the temperature field evolving in time.

| Script | Problem | Plot |
|--------|---------|------|
| 1D solver | Temperature `u(x, t)` along a strip of length `L` | `pcolormesh` colour strip |
| 2D solver | Temperature `u(x, y, t)` over an `L × L` plate | `imshow` heat map |

## Theory

### The heat equation

Heat conduction in a homogeneous material with no internal heat sources is described by the heat (diffusion) equation. In one dimension:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

and in two dimensions:

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

where

- $u$ is the temperature (°C),
- $t$ is time (s),
- $\alpha$ is the **thermal diffusivity** (mm²/s), defined as $\alpha = k / (\rho c_p)$, with thermal conductivity $k$, density $\rho$ and specific heat capacity $c_p$.

In the code, $\alpha$ is the variable `a`. A value of 110 mm²/s is in the range of a highly conductive metal such as aluminium.

Physically, the equation says that the temperature at a point rises when it is colder than the average of its neighbours (positive curvature) and falls when it is hotter than the average (negative curvature). Sharp temperature differences smooth out over time.

### Discretisation

Analytical solutions exist only for simple geometries, so we approximate the equation on a grid.

**Space.** The domain is split into evenly spaced nodes. For a strip of length $L$ with $N$ nodes (including both end nodes):

$$
\Delta x = \frac{L}{N - 1}, \qquad x_i = i\,\Delta x, \quad i = 0, 1, \dots, N-1
$$

**Time.** Time advances in fixed steps $\Delta t$, so $t^n = n\,\Delta t$. We write $u_i^n$ for the temperature at node $i$ and time step $n$.

**Derivatives.** Using Taylor series, the time derivative is approximated with a *forward difference*:

$$
\frac{\partial u}{\partial t} \approx \frac{u_i^{n+1} - u_i^{n}}{\Delta t}
$$

and the second spatial derivative with a *central difference*:

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}^{n} - 2u_i^{n} + u_{i-1}^{n}}{\Delta x^2}
$$

### The explicit (FTCS) scheme

Substituting these into the 1D heat equation and solving for the unknown $u_i^{n+1}$ gives the **Forward-Time Central-Space (FTCS)** scheme:

$$
u_i^{n+1} = u_i^{n} + r\left(u_{i+1}^{n} - 2u_i^{n} + u_{i-1}^{n}\right), \qquad r = \frac{\alpha\,\Delta t}{\Delta x^2}
$$

It is called *explicit* because every new value depends only on known values from the previous time step, so no linear system has to be solved.

In 2D, with a square grid ($\Delta x = \Delta y$), the central differences in both directions give the **5-point stencil**:

$$
u_{i,j}^{n+1} = u_{i,j}^{n} + r\left(u_{i+1,j}^{n} + u_{i-1,j}^{n} + u_{i,j+1}^{n} + u_{i,j-1}^{n} - 4u_{i,j}^{n}\right)
$$

with the same definition $r = \alpha\,\Delta t / \Delta x^2$.

A useful way to read the 2D update is to rearrange it:

$$
u_{i,j}^{n+1} = (1 - 4r)\,u_{i,j}^{n} + r\left(u_{i+1,j}^{n} + u_{i-1,j}^{n} + u_{i,j+1}^{n} + u_{i,j-1}^{n}\right)
$$

The new temperature is a weighted average of the node and its four neighbours. This only behaves physically (no new maxima or minima appear) if all weights are non-negative, which leads directly to the stability condition.

### Stability

The explicit scheme is only **conditionally stable**. If $\Delta t$ is too large relative to $\Delta x^2$, small errors grow exponentially and the solution blows up with oscillations.

A von Neumann stability analysis substitutes a Fourier mode $u_i^n = G^n e^{\mathrm{i}k x_i}$ into the scheme. In 1D the amplification factor is

$$
G = 1 - 4r\sin^2\!\left(\frac{k\,\Delta x}{2}\right)
$$

Errors stay bounded only if $|G| \le 1$ for all wavenumbers $k$, which requires

$$
r = \frac{\alpha\,\Delta t}{\Delta x^2} \le \frac{1}{2} \quad \text{(1D)}
$$

In 2D (with $\Delta x = \Delta y$), the amplification factor picks up a second term and the condition becomes stricter:

$$
r = \frac{\alpha\,\Delta t}{\Delta x^2} \le \frac{1}{4} \quad \text{(2D)}
$$

This is why the scripts choose

```python
dt = 0.5  * dx**2 / a   # 1D
dt = 0.25 * dx**2 / a   # 2D
```

i.e. they run exactly at the stability limit. In practice a slightly smaller factor (e.g. 0.4 in 1D, 0.2 in 2D) gives a smoother solution with a small cost in run time.

### Boundary and initial conditions

The problem is only fully defined once we specify the starting temperature and what happens at the edges.

- **Initial condition:** the plate starts uniformly at 20 °C.
- **Boundary condition:** **Dirichlet** (fixed temperature). The outer nodes are held at 100 °C for all time, as if the plate edges are in contact with a thermostatically controlled heat source.

Dirichlet conditions are enforced simply by never updating the boundary nodes. Only interior nodes are computed, and the edge values set before the loop stay constant.


### Accuracy

The FTCS scheme is **first-order accurate in time** and **second-order accurate in space**. The leading truncation error is

$$
\mathcal{O}(\Delta t) + \mathcal{O}(\Delta x^2)
$$

Because $\Delta t$ is tied to $\Delta x^2$ by the stability condition, the overall error is effectively $\mathcal{O}(\Delta x^2)$.

Note that the fixed 100 °C edge and the 20 °C interior create a discontinuity at $t = 0$. Discontinuities are smoothed out immediately by real diffusion, but a coarse grid cannot resolve this, so expect small inaccuracies in the first few time steps near the edges.

## How the theory maps to the code

| Theory | Code |
|--------|------|
| Thermal diffusivity $\alpha$ | `a` |
| Plate length $L$ | `length` |
| Number of nodes $N$ | `nodes` |
| Grid spacing $\Delta x = L/(N-1)$ | `dx = length / (nodes - 1)` |
| Time step (stability-limited) | `dt = 0.5 * dx**2 / a` (1D), `0.25 * ...` (2D) |
| Number of steps | `t_nodes = int(time / dt)` |
| Diffusion number $r$ | `r = a * dt / dx**2` |
| Temperature field $u^n$ | `u` |
| Next time level $u^{n+1}$ | `w` (a copy, so every node updates from the same time level) |
| Dirichlet boundaries | `u[0] = 100`, `u[-1] = 100` (never overwritten in the loop) |

**1D update:**

```python
for i in range(1, nodes - 1):
    w[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
```

**2D update (vectorised 5-point stencil):**

```python
w[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
    u[2:, 1:-1] + u[:-2, 1:-1] +
    u[1:-1, 2:] + u[1:-1, :-2] -
    4 * u[1:-1, 1:-1]
)
```

The slices `u[2:, 1:-1]`, `u[:-2, 1:-1]`, `u[1:-1, 2:]` and `u[1:-1, :-2]` are the four neighbours (down, up, right, left) of every interior node, evaluated for the whole grid at once. This performs exactly the same arithmetic as nested `for` loops but runs far faster.

