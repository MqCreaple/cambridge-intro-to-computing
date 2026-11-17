import numpy as np
import taichi as ti
import taichi.math as tm
import torch
import cupy as cp
from cupyx.scipy.fft import dctn, idctn   # Discrete cosine transformation (n-dimensional)

ti.init(arch=ti.gpu)

#-- Constants --#
# Directions
TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3
# Shader options
SHADER_DYE = 0
SHADER_PRESSURE = 1
SHADER_U = 2
SHADER_V = 3
# Brush options
BRUSH_DYE = 0
BRUSH_FORCE = 1

#-- Simulation parameters --#
M: int = 500            # Number of horizontal cells
N: int = 500            # Number of vertical cells
side = 1.0 / M     # Side length of each cell (in metres)
dt = 0.001         # Time between two consecutive frames (in seconds)
viscosity = 3e-4
density = 1e3
# Boundary velocities, in the order of top, right, bottom, and left.
boundaries_np = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 0.]], dtype=np.float32)
dye_radius = 0.005 # The radius of the added dye when the GUI is clicked
dye_scale = 0.5
left_click_dye_colour = ti.Vector([0.75, 0.01, 0.02])
right_click_dye_colour = ti.Vector([0.13, 0.32, 0.86])
external_force_scale = 1e4
external_force_radius = 0.02  # In metres
physical_steps_per_frame = 4

#-- UI Parameters --#
CONTROL_PANEL_WIDTH = 300  # Width of the control panel, in pixels

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
dye = ti.Vector.field(3, dtype=ti.f32, shape=(M, N))   # Dye field for visualisation

#-- Low-pass Filter --#
class LowPassSignalFilter:
    def __init__(self, init_value: ti.types.vector(2, dtype=ti.f32) = np.nan, decay_factor: float = 0.25):
        """
        Initialize the low-pass signal filter.
        
        `value` specifies the initial value set to the filter. `decay_factor` describes the desired decay in one real-world second.
        """
        self.value = init_value
        self.gamma = -np.log(decay_factor)

    def update(self, new_value: ti.types.vector(2, dtype=ti.f32), dt: float):
        if np.any(np.isnan(self.value)):
            self.value = new_value
        else:
            a = np.exp(-self.gamma * dt)
            self.value = a * self.value + (1 - a) * new_value

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
@ti.func
def external_force_function(mouse_x: float, mouse_y: float, last_x: float, last_y: float, x: float, y: float) -> ti.types.vector(2, dtype=ti.f32):
    drag_dir = ti.Vector([(mouse_x - last_x) * width, (mouse_y - last_y) * height])
    scale = ti.exp(-((x - mouse_x)**2 + (y - mouse_y)**2) / (2 * external_force_radius**2))
    return external_force_scale * scale * drag_dir

@ti.kernel
def update_external_force(mouse_x: float, mouse_y: float, last_x: float, last_y: float):
    """
    Update the velocity field based on the clicked position and drag direction.
    
    `mouse_x`, `mouse_y`, `last_x`, and `last_y` are floating point numbers between 0 and 1, representing
    the fraction of width or height.
    """
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            x = (i + 1) * side
            y = (j + 0.5) * side
            force = external_force_function(mouse_x, mouse_y, last_x, last_y, x, y)
            u[i, j] += dt * force.x
        if i < v.shape[0] and j < v.shape[1]:
            x = (i + 0.5) * side
            y = (j + 1) * side
            force = external_force_function(mouse_x, mouse_y, last_x, last_y, x, y)
            v[i, j] += dt * force.y

u_augmented_x = ti.field(dtype=ti.f32, shape=(M + 1, N))
u_augmented_y = ti.field(dtype=ti.f32, shape=(M - 1, N + 2))
v_augmented_x = ti.field(dtype=ti.f32, shape=(M + 2, N - 1))
v_augmented_y = ti.field(dtype=ti.f32, shape=(M, N + 1))

@ti.kernel
def compute_augmented_arrays(boundaries: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i, j in ti.ndrange(M + 2, N + 2):
        if i < u_augmented_x.shape[0] and j < u_augmented_x.shape[1]:
            u_augmented_x[i, j] = boundaries[LEFT, 0] if i == 0 else boundaries[RIGHT, 0] if i == M else u[i - 1, j]
        if i < u_augmented_y.shape[0] and j < u_augmented_y.shape[1]:
            if j == 0:
                u_augmented_y[i, j] = boundaries[BOTTOM, 0] * 2 - u[i, 0]
            elif j == N + 1:
                u_augmented_y[i, j] = boundaries[TOP, 0] * 2 - u[i, N - 1]
            else:
                u_augmented_y[i, j] = u[i, j - 1]
        if i < v_augmented_x.shape[0] and j < v_augmented_x.shape[1]:
            if i == 0:
                v_augmented_x[i, j] = boundaries[LEFT, 1] * 2 - v[0, j]
            elif i == M + 1:
                v_augmented_x[i, j] = boundaries[RIGHT, 1] * 2 - v[M - 1, j]
            else:
                v_augmented_x[i, j] = v[i - 1, j]
        if i < v_augmented_y.shape[0] and j < v_augmented_y.shape[1]:
            v_augmented_y[i, j] = boundaries[BOTTOM, 1] if j == 0 else boundaries[TOP, 1] if j == N else v[i, j - 1]

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

@ti.func
def interpolate_velocity(boundaries: ti.types.ndarray(dtype=ti.f32, ndim=2), pos: ti.types.vector(2, dtype=ti.f32)) -> ti.types.vector(2, dtype=ti.f32):
    pos = tm.clamp(pos, 0, ti.Vector([width, height]))
    i, j = pos.x / side, pos.y / side
    iu, ju = int(tm.floor(i - 1)), int(tm.floor(j + 0.5))
    su, tu = i - 1 - iu, j + 0.5 - ju
    u1 = boundaries[LEFT, 0] if iu < 0 else u_augmented_y[iu, ju]
    u2 = boundaries[LEFT, 0] if iu < 0 else u_augmented_y[iu, ju + 1]
    u3 = boundaries[RIGHT, 0] if iu + 1 >= M - 1 else u_augmented_y[iu + 1, ju]
    u4 = boundaries[RIGHT, 0] if iu + 1 >= M - 1 else u_augmented_y[iu + 1, ju + 1]
    u = u1 * (1 - su) * (1 - tu) + u2 * (1 - su) * tu + u3 * su * (1 - tu) + u4 * su * tu
    iv, jv = int(tm.floor(i + 0.5)), int(tm.floor(j - 1))
    sv, tv = i + 0.5 - iv, j - 1 - jv
    v1 = boundaries[BOTTOM, 1] if jv < 0 else v_augmented_x[iv, jv]
    v2 = boundaries[BOTTOM, 1] if jv < 0 else v_augmented_x[iv + 1, jv]
    v3 = boundaries[TOP, 1] if jv + 1 >= N - 1 else v_augmented_x[iv, jv + 1]
    v4 = boundaries[TOP, 1] if jv + 1 >= N - 1 else v_augmented_x[iv + 1, jv + 1]
    v = v1 * (1 - sv) * (1 - tv) + v2 * sv * (1 - tv) + v3 * (1 - sv) * tv + v4 * sv * tv
    return ti.Vector([u, v])

@ti.kernel
def advect_dye(boundaries: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    """Use semi-Lagrangian advection scheme to advect the dye."""
    for i, j in dye:
        centre = ti.Vector([(i + 0.5) * side, (j + 0.5) * side])
        mid = centre - 0.5 * dt * velocity[i, j]
        mid_velocity = interpolate_velocity(boundaries, mid)
        prev = centre - dt * mid_velocity
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
def copy_and_inject_dye(mouse_x: float, mouse_y: float, radius: float, colour: ti.types.vector(3, ti.f32)):
    """
    Copy the `new_dye` field to `dye` and inject new dye in the clicked position.
    
    `mouse_x` and `mouse_y` are floating point numbers between 0 and 1, which represent the fraction of width/height.
    """
    for i, j in dye:
        dye[i, j] = new_dye[i, j]
        x = (i + 0.5) * side
        y = (j + 0.5) * side
        if not tm.isnan(mouse_x) and not tm.isnan(mouse_y):
            dye[i, j] += colour * dye_scale * tm.exp(-((x - mouse_x * width) ** 2 + (y - mouse_y * height) ** 2) / (2 * radius ** 2))

image = ti.Vector.field(3, dtype=ti.f32, shape=(M + CONTROL_PANEL_WIDTH, N))

@ti.kernel
def shader(pressure: ti.types.ndarray(dtype=ti.f32, ndim=2), shader_option: int):
    RED = ti.Vector([0.75, 0.01, 0.02])
    BLUE = ti.Vector([0.13, 0.32, 0.86])
    GRAY = ti.Vector([0.90, 0.90, 0.90])
    for i, j in image:
        if i >= M:
            image[i, j] = ti.Vector([0.5, 0.5, 0.5])
            continue
        if shader_option == SHADER_DYE:
            image[i, j] = dye[i, j]
        else:
            val = 0.0
            value_range = 1.0
            if shader_option == SHADER_PRESSURE:
                val = pressure[i, j]
                value_range = 100.0
            elif shader_option == SHADER_U:
                val = velocity[i, j].x
                value_range = 1.0
            elif shader_option == SHADER_V:
                val = velocity[i, j].y
                value_range = 1.0

            if val > 0:
                t = tm.clamp(val / value_range, 0.0, 1.0)
                image[i, j] = RED * t + GRAY * (1 - t)
            else:
                t = tm.clamp(-val / value_range, 0.0, 1.0)
                image[i, j] = BLUE * t + GRAY * (1 - t)

#-- Main code --#
def physical_simulation(mouse_x: float, mouse_y: float, last_x: float, last_y: float, colour: ti.Vector, brush_option: int) -> torch.Tensor:
    """Performs one iteration of the physical simulation steps. Returns the pressure field."""
    if brush_option == BRUSH_FORCE and not np.isnan(mouse_x) and not np.isnan(mouse_y) and not np.isnan(last_x) and not np.isnan(last_y) \
        and (mouse_x != last_x or mouse_y != last_y):
        # print(mouse_x, mouse_y, last_x, last_y)
        update_external_force(mouse_x, mouse_y, last_x, last_y)
    compute_augmented_arrays(boundaries)
    compute_shifted_arrays()
    compute_advection()
    update_advection()
    compute_augmented_arrays(boundaries)
    compute_viscosity()
    update_viscosity()
    compute_augmented_arrays(boundaries)
    assemble_poisson_rhs(poisson_rhs)
    ti.sync()  # Ensure that pytorch and cupy sees the latest data
    poisson_rhs_cupy = cp.from_dlpack(poisson_rhs)
    pressure_cupy = neumann_solver.solve(poisson_rhs_cupy)
    pressure_cupy -= cp.mean(pressure_cupy)
    pressure_torch = torch.from_dlpack(pressure_cupy)
    update_pressure(pressure_torch)
    ti.sync()
    compute_augmented_arrays(boundaries)
    compute_velocity_field()
    advect_dye(boundaries)
    if brush_option == BRUSH_DYE:
        copy_and_inject_dye(mouse_x, mouse_y, dye_radius, colour)
    else:
        copy_and_inject_dye(tm.nan, tm.nan, dye_radius, colour)
    return pressure_torch

WINDOW_WIDTH = M + CONTROL_PANEL_WIDTH
window = ti.ui.Window(name="Fluid Simulation", res = (WINDOW_WIDTH, N), fps_limit=200)
canvas = window.get_canvas()
gui = window.get_gui()
shader_option = SHADER_DYE
brush_option = BRUSH_DYE
mouse_filter = LowPassSignalFilter()
last_x, last_y = tm.nan, tm.nan

frame = 0
while window.running:
    # Event handling
    mouse_x_target, mouse_y_target = tm.nan, tm.nan
    dye_colour = ti.Vector([0., 0., 0.])
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            break
    if window.is_pressed(ti.ui.LMB):
        x, y = window.get_cursor_pos()
        if x < M / WINDOW_WIDTH:
            mouse_x_target, mouse_y_target = x * WINDOW_WIDTH / M, y
            dye_colour += left_click_dye_colour
    if window.is_pressed(ti.ui.RMB):
        x, y = window.get_cursor_pos()
        if x < M / WINDOW_WIDTH:
            mouse_x_target, mouse_y_target = x * WINDOW_WIDTH / M, y
            dye_colour += right_click_dye_colour

    # Physical steps
    for _ in range(physical_steps_per_frame):
        mouse_filter.update(ti.Vector([mouse_x_target, mouse_y_target]), 0.05)
        mouse_x, mouse_y = mouse_filter.value
        pressure = physical_simulation(mouse_x, mouse_y, last_x, last_y, dye_colour, brush_option)
        last_x, last_y = mouse_x, mouse_y

    # Shading
    shader(pressure, shader_option)
    ti.sync()
    canvas.set_image(image)
    with gui.sub_window(
        "Simulation Controls",
        x = M / WINDOW_WIDTH,
        y = 0.0,
        width = CONTROL_PANEL_WIDTH / WINDOW_WIDTH,
        height = 1.0,
    ):
        gui.text("Fluid Simulation")
        gui.text(f"frame: {frame}")
        gui.text(f"time: {frame * physical_steps_per_frame * dt:.4f}")
        
        gui.text("Shader options:")
        if gui.checkbox("Dye##shader", shader_option == SHADER_DYE):
            shader_option = SHADER_DYE
        if gui.checkbox("Pressure", shader_option == SHADER_PRESSURE):
            shader_option = SHADER_PRESSURE
        if gui.checkbox("X Velocity (U)", shader_option == SHADER_U):
            shader_option = SHADER_U
        if gui.checkbox("Y Velocity (V)", shader_option == SHADER_V):
            shader_option = SHADER_V

        gui.text("Brush options:")
        if gui.checkbox("Dye##brush", brush_option == BRUSH_DYE):
            brush_option = BRUSH_DYE
        if gui.checkbox("Force", brush_option == BRUSH_FORCE):
            brush_option = BRUSH_FORCE

    window.show()
    frame += 1
