import numpy as np
import taichi as ti
import taichi.math as tm
import torch
import cupy as cp
from cupyx.scipy.fft import dctn, idctn   # Discrete cosine transformation (n-dimensional)

ti.init(arch=ti.gpu)

#-- Simulation parameters --#
M = 500            # Number of horizontal cells
N = 500            # Number of vertical cells
side = 1.0 / M     # Side length of each cell (in metres)
dt = 0.001         # Time between two consecutive frames (in seconds)
viscosity = 3e-4
density = 1e3
# Boundary velocities, in the order of top, right, bottom, and left.
boundaries_np = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 0.]], dtype=np.float32)
dye_radius = 0.005 # The radius of the added dye when the GUI is clicked
left_click_dye_colour = ti.Vector([0.75, 0.01, 0.02])
right_click_dye_colour = ti.Vector([0.13, 0.32, 0.86])
physical_steps_per_frame = 4

#-- Computed parameters --#
boundaries = ti.ndarray(dtype=ti.f32, shape=(4, 2))
boundaries.from_numpy(boundaries_np)
width = M * side
height = N * side

#-- State arrays --#
# The solver uses MAC grid -- `u` values are placed on vertical cell edges, `v` values are placed on horizontal cell edges,
# and `p` are placed on cell centres. Let the grid shape be (M, N), then the shape of `u` is (M-1, N), `v` is (M, N-1), and
# `p` is (M, N). The velocities on the boundaries are specified in the `boundaries` variable.
u = ti.field(dtype=ti.f32, shape=(M - 1, N))
v = ti.field(dtype=ti.f32, shape=(M, N - 1))
dye = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))   # Dye field for visualisation.

#-- Poisson solver --#
class NeumannPoissonSolver:
    """
    Solve A p = b, where A is the unscaled five-point negative Laplacian with homogeneous Neumann boundary conditions.

    The pressure is fixed to have zero mean.
    """

    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        k = cp.arange(M, dtype=cp.float32)
        l = cp.arange(N, dtype=cp.float32)
        eigenvalue_x = 4 * cp.sin(cp.pi * k / (2 * M))**2
        eigenvalue_y = 4 * cp.sin(cp.pi * l / (2 * N))**2
        # Eigenvalue of the mode `mode[i, j] = cos(pi*k*(i+0.5)/M)*cos(pi*l*(j+0.5)/N)`
        eigenvalues = eigenvalue_x[:, None] + eigenvalue_y[None, :]
        eigenvalues[0, 0] = 1.0  # Set the constant mode's denominator to 1.0 to avoid division by 0
        self.eigenvalues = eigenvalues

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        assert b.shape == (self.M, self.N)
        b_hat = dctn(b, type=2, norm="ortho")
        p_hat = b_hat / self.eigenvalues
        p_hat[0, 0] = 0.0   # Manually remove the constant mode
        return idctn(p_hat, type=2, norm="ortho")

neumann_solver = NeumannPoissonSolver(M, N)

#-- Kernel functions --#
u_augmented_x = ti.field(dtype=ti.f32, shape=(M + 1, N))
u_augmented_y = ti.field(dtype=ti.f32, shape=(M - 1, N + 2))
v_augmented_x = ti.field(dtype=ti.f32, shape=(M + 2, N - 1))
v_augmented_y = ti.field(dtype=ti.f32, shape=(M, N + 1))

@ti.kernel
def compute_augmented_arrays(boundaries: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(M + 2, N + 2):
        if i < u_augmented_x.shape[0] and j < u_augmented_x.shape[1]:
            u_augmented_x[i, j] = boundaries[3, 0] if i == 0 else boundaries[1, 0] if i == M else u[i - 1, j]
        if i < u_augmented_y.shape[0] and j < u_augmented_y.shape[1]:
            if j == 0:
                u_augmented_y[i, j] = boundaries[2, 0] * 2 - u[i, 0]
            elif j == N + 1:
                u_augmented_y[i, j] = boundaries[0, 0] * 2 - u[i, N - 1]
            else:
                u_augmented_y[i, j] = u[i, j - 1]
        if i < v_augmented_x.shape[0] and j < v_augmented_x.shape[1]:
            if i == 0:
                v_augmented_x[i, j] = boundaries[3, 1] * 2 - v[0, j]
            elif i == M + 1:
                v_augmented_x[i, j] = boundaries[1, 1] * 2 - v[M - 1, j]
            else:
                v_augmented_x[i, j] = v[i - 1, j]
        if i < v_augmented_y.shape[0] and j < v_augmented_y.shape[1]:
            v_augmented_y[i, j] = boundaries[2, 1] if j == 0 else boundaries[0, 1] if j == N else v[i, j - 1]

u_shifted = ti.field(dtype=ti.f32, shape=(M, N - 1))
v_shifted = ti.field(dtype=ti.f32, shape=(M - 1, N))

@ti.kernel
def compute_shifted_arrays():
    for i, j in ti.ndrange(M, N):
        if i < u_shifted.shape[0] and j < u_shifted.shape[1]:
            u_shifted[i, j] = 0.25 * (u_augmented_x[i, j] + u_augmented_x[i + 1, j]
                                      + u_augmented_x[i, j + 1] + u_augmented_x[i + 1, j + 1])
        if i < v_shifted.shape[0] and j < v_shifted.shape[1]:
            v_shifted[i, j] = 0.25 * (v_augmented_y[i, j] + v_augmented_y[i + 1, j]
                                      + v_augmented_y[i, j + 1] + v_augmented_y[i + 1, j + 1])

du_dx = ti.field(dtype=ti.f32, shape=u.shape)
du_dy = ti.field(dtype=ti.f32, shape=u.shape)
dv_dx = ti.field(dtype=ti.f32, shape=v.shape)
dv_dy = ti.field(dtype=ti.f32, shape=v.shape)
advection_u = ti.field(dtype=ti.f32, shape=u.shape)
advection_v = ti.field(dtype=ti.f32, shape=v.shape)

@ti.kernel
def compute_advection():
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            # Compute du_dx and du_dy
            if u[i, j] > 0:
                du_dx[i, j] = (u[i, j] - u_augmented_x[i, j]) / side
            else:
                du_dx[i, j] = (u_augmented_x[i + 2, j] - u[i, j]) / side
            if v_shifted[i, j] > 0:
                du_dy[i, j] = (u[i, j] - u_augmented_y[i, j]) / side
            else:
                du_dy[i, j] = (u_augmented_y[i, j + 2] - u[i, j]) / side
            advection_u[i, j] = u[i, j] * du_dx[i, j] + v_shifted[i, j] * du_dy[i, j]
        if i < v.shape[0] and j < v.shape[1]:
            # Compute dv_dx and dv_dy
            if v[i, j] > 0:
                dv_dy[i, j] = (v[i, j] - v_augmented_y[i, j]) / side
            else:
                dv_dy[i, j] = (v_augmented_y[i, j + 2] - v[i, j]) / side
            if u_shifted[i, j] > 0:
                dv_dx[i, j] = (v[i, j] - v_augmented_x[i, j]) / side
            else:
                dv_dx[i, j] = (v_augmented_x[i + 2, j] - v[i, j]) / side
            advection_v[i, j] = v[i, j] * dv_dy[i, j] + u_shifted[i, j] * dv_dx[i, j]

@ti.kernel
def update_advection():
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            u[i, j] -= dt * advection_u[i, j]
        if i < v.shape[0] and j < v.shape[1]:
            v[i, j] -= dt * advection_v[i, j]

d2u_dx2 = ti.field(dtype=ti.f32, shape=u.shape)
d2u_dy2 = ti.field(dtype=ti.f32, shape=u.shape)
d2v_dx2 = ti.field(dtype=ti.f32, shape=v.shape)
d2v_dy2 = ti.field(dtype=ti.f32, shape=v.shape)

@ti.kernel
def compute_viscosity():
    side2 = side * side
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            d2u_dx2[i, j] = (u_augmented_x[i + 2, j] - 2 * u_augmented_x[i + 1, j] + u_augmented_x[i, j]) / side2
            d2u_dy2[i, j] = (u_augmented_y[i, j + 2] - 2 * u_augmented_y[i, j + 1] + u_augmented_y[i, j]) / side2
        if i < v.shape[0] and j < v.shape[1]:
            d2v_dx2[i, j] = (v_augmented_x[i + 2, j] - 2 * v_augmented_x[i + 1, j] + v_augmented_x[i, j]) / side2
            d2v_dy2[i, j] = (v_augmented_y[i, j + 2] - 2 * v_augmented_y[i, j + 1] + v_augmented_y[i, j]) / side2

@ti.kernel
def update_viscosity():
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            u[i, j] += viscosity * dt * (d2u_dx2[i, j] + d2u_dy2[i, j])
        if i < v.shape[0] and j < v.shape[1]:
            v[i, j] += viscosity * dt * (d2v_dx2[i, j] + d2v_dy2[i, j])

poisson_rhs = torch.empty((M, N), dtype=torch.float32, device="cuda")

@ti.kernel
def assemble_poisson_rhs(poisson_rhs: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    """Assemble the right hand side vector of the Poisson equation to solve for pressure field."""
    for i, j in poisson_rhs:
        du_dx_times_side = u_augmented_x[i + 1, j] - u_augmented_x[i, j]
        dv_dy_times_side = v_augmented_y[i, j + 1] - v_augmented_y[i, j]
        poisson_rhs[i, j] = -(side * density / dt) * (du_dx_times_side + dv_dy_times_side)

@ti.kernel
def update_pressure(pressure: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            u[i, j] -= dt / (side * density) * (pressure[i + 1, j] - pressure[i, j])
        if i < v.shape[0] and j < v.shape[1]:
            v[i, j] -= dt / (side * density) * (pressure[i, j + 1] - pressure[i, j])

velocity = ti.Vector.field(2, dtype=ti.f32, shape=(M, N))

@ti.kernel
def compute_velocity_field():
    for i, j in velocity:
        velocity[i, j].x = 0.5 * (u_augmented_x[i, j] + u_augmented_x[i + 1, j])
        velocity[i, j].y = 0.5 * (v_augmented_y[i, j] + v_augmented_y[i, j + 1])

new_dye = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))

@ti.kernel
def advect_dye():
    """Use semi-Lagrangian advection scheme to advect the dye."""
    for i, j in dye:
        centre = ti.Vector([(i + 0.5) * side, (j + 0.5) * side])
        prev = centre - dt * velocity[i, j]
        prev = tm.clamp(prev / side - ti.Vector([0.5, 0.5]), ti.Vector([0, 0]), ti.Vector([M - 1, N - 1]))
        # Bilinear interpolation of the dye colour
        i1 = ti.i32(tm.floor(prev.x))
        if i1 == M - 1:
            i1 -= 1
        j1 = ti.i32(tm.floor(prev.y))
        if j1 == N - 1:
            j1 -= 1
        s = prev.x - i1
        t = prev.y - j1
        new_dye[i, j] = dye[i1, j1] * (1 - s) * (1 - t) + dye[i1, j1 + 1] * (1 - s) * t \
                + dye[i1 + 1, j1] * s * (1 - t) + dye[i1 + 1, j1 + 1] * s * t

@ti.kernel
def copy_and_inject_dye(click_x: float, click_y: float, radius: float, colour: ti.types.vector(3, ti.f32)):
    for i, j in dye:
        dye[i, j] = new_dye[i, j]
        x = (i + 0.5) * side
        y = (j + 0.5) * side
        if not tm.isnan(click_x) and not tm.isnan(click_y):
            dye[i, j] += colour * tm.exp(-((x - click_x * width) ** 2 + (y - click_y * height) ** 2) / (2 * radius ** 2))

image = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))

@ti.kernel
def shader(pressure: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    RED = ti.Vector([0.75, 0.01, 0.02])
    BLUE = ti.Vector([0.13, 0.32, 0.86])
    GRAY = ti.Vector([0.90, 0.90, 0.90])
    RANGE = 1.0
    for i, j in image:
        val = pressure[i, j]
        if val > 0:
            t = tm.clamp(val / RANGE, 0.0, 1.0)
            image[i, j] = RED * t + GRAY * (1 - t)
        else:
            t = tm.clamp(-val / RANGE, 0.0, 1.0)
            image[i, j] = BLUE * t + GRAY * (1 - t)

#-- Main code --#
def physical_simulation(click_x: float, click_y: float, colour: ti.Vector) -> torch.Tensor:
    """Performs one iteration of the physical simulation steps. Returns the pressure field."""
    compute_augmented_arrays(boundaries_np)
    compute_shifted_arrays()
    compute_advection()
    update_advection()
    compute_augmented_arrays(boundaries_np)
    compute_viscosity()
    update_viscosity()
    compute_augmented_arrays(boundaries_np)
    assemble_poisson_rhs(poisson_rhs)
    ti.sync()  # Ensure that pytorch and cupy sees the latest data
    poisson_rhs_cupy = cp.from_dlpack(poisson_rhs)
    pressure_cupy = neumann_solver.solve(poisson_rhs_cupy)
    pressure_cupy -= cp.mean(pressure_cupy)
    pressure_torch = torch.from_dlpack(pressure_cupy)
    update_pressure(pressure_torch)
    ti.sync()
    compute_augmented_arrays(boundaries_np)
    compute_velocity_field()
    advect_dye()
    copy_and_inject_dye(click_x, click_y, dye_radius, colour)
    return pressure_torch

gui = ti.GUI("Fluid simulation", res=(M, N))

frame = 0
while gui.running:
    click_x, click_y = tm.nan, tm.nan
    dye_colour = ti.Vector([0., 0., 0.])
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            break
        elif e.key == gui.LMB:
            click_x, click_y = e.pos
            dye_colour = left_click_dye_colour
        elif e.key == gui.RMB:
            click_x, click_y = e.pos
            dye_colour = right_click_dye_colour
    for _ in range(physical_steps_per_frame):
        pressure = physical_simulation(click_x, click_y, dye_colour)
    # shader(pressure)
    ti.sync()
    gui.set_image(dye)
    gui.text(f"frame {frame}", pos=[0.1, 0.1], font_size=14)
    gui.text(f"time: {frame * physical_steps_per_frame * dt:.4f}", pos=[0.1, 0.05], font_size=14)
    gui.show()
    frame += 1
