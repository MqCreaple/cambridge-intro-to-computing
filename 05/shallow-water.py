# Shallow water simulation
import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch="gpu")

#-- Constants --#
# Directions
TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3
# Shader options
SHADER_H = 0
SHADER_ELEVATION = 1
SHADER_U = 2
SHADER_V = 3
# Brush options
BRUSH_FORCE = 0
BRUSH_DIVERGENCE = 1

def default_shader_value_range(shader_option: int) -> tuple[float, float, float]:
    if shader_option == SHADER_H:
        return (0.05, 0.005, 0.1)
    elif shader_option == SHADER_ELEVATION:
        return (0.005, 0.001, 0.05)
    elif shader_option == SHADER_U or shader_option == SHADER_V:
        return (0.2, 0.05, 5.0)
    return (1.0, 0.1, 5.0)

def default_brush_force_range(brush_option: int) -> tuple[float, float, float]:
    if brush_option == BRUSH_FORCE:
        return (300, 0, 500)
    elif brush_option == BRUSH_DIVERGENCE:
        return (100, 0, 500)
    return (500, 0, 1000)

#-- Simulation parameters --#
M: int = 600            # Number of horizontal cells
N: int = 600            # Number of vertical cells
side = 1.0 / M     # Side length of each cell (in metres)
dt = 0.001         # Time between two consecutive frames (in seconds)
density = 1e3
g = 9.8            # Gravitational constant
initial_water_elevation = 0.0
physical_steps_per_frame = 5
external_force_radius = 0.02  # In metres

@ti.func
def bed_height(x: float, y: float):
    """Defines the height field of the water bed."""
    return -0.02 + 0.017 * x

#-- UI Parameters --#
CONTROL_PANEL_WIDTH = 300  # Width of the control panel, in pixels

#-- Computed parameters --#
width = M * side
length = N * side

#-- State arrays --#
# The solver uses MAC grid -- `u` values are placed on vertical cell edges, `v` values are placed on horizontal cell edges,
# and `h` are placed on cell centres. Let the grid shape be (M, N), then the shape of `u` is (M-1, N), `v` is (M, N-1), and
# `h` is (M, N).
u = ti.field(dtype=ti.f32, shape=(M - 1, N))
v = ti.field(dtype=ti.f32, shape=(M, N - 1))
h = ti.field(dtype=ti.f32, shape=(M, N))

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

#-- Kernel functions --#
bed = ti.field(dtype=ti.f32, shape=(M, N))

@ti.kernel
def precompute_bed_and_height():
    for i, j in bed:
        x = (i + 0.5) * side
        y = (j + 0.5) * side
        bed[i, j] = bed_height(x, y)
        h[i, j] = initial_water_elevation - bed[i, j]

@ti.func
def external_force_function(
    mouse_x: float,
    mouse_y: float,
    last_x: float,
    last_y: float,
    brush_option: int,
    brush_force: float,
    x: float,
    y: float,
) -> ti.types.vector(2, dtype=ti.f32):
    ret = ti.Vector([0., 0.])
    if brush_option == BRUSH_FORCE:
        drag_dir = ti.Vector([mouse_x - last_x, mouse_y - last_y])
        drag_dir_len = tm.length(drag_dir)
        if drag_dir_len > 0:
            drag_dir /= drag_dir_len
        scale = ti.exp(-((x - mouse_x)**2 + (y - mouse_y)**2) / (2 * external_force_radius**2))
        ret = brush_force * scale * drag_dir
    elif brush_option == BRUSH_DIVERGENCE:
        unit_dir = ti.Vector([x - mouse_x, y - mouse_y])
        unit_dir_len = tm.length(unit_dir)
        if unit_dir_len > 0:
            unit_dir /= unit_dir_len
        scale = ti.exp(-((x - mouse_x)**2 + (y - mouse_y)**2) / (2 * external_force_radius**2))
        ret = brush_force * scale * unit_dir
    return ret

@ti.kernel
def update_external_force(
    mouse_x_normalised: float,
    mouse_y_normalised: float,
    last_x_normalised: float,
    last_y_normalised: float,
    brush_option: int,
    brush_force: float
):
    """
    Update the velocity field based on the clicked position and drag direction.
    
    `mouse_x_normalised`, `mouse_y_normalised`, `last_x_normalised`, and `last_y_normalised` are floating point numbers
    between 0 and 1, representing the fraction of width or length.
    """
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            x = (i + 1) * side
            y = (j + 0.5) * side
            force = external_force_function(
                mouse_x_normalised * width,
                mouse_y_normalised * length,
                last_x_normalised * width,
                last_y_normalised * length,
                brush_option, brush_force, x, y
            )
            height = 0.5 * (h[i, j] + h[i + 1, j])
            u[i, j] += dt * force.x / (density * height)
        if i < v.shape[0] and j < v.shape[1]:
            x = (i + 0.5) * side
            y = (j + 1) * side
            force = external_force_function(
                mouse_x_normalised * width,
                mouse_y_normalised * length,
                last_x_normalised * width,
                last_y_normalised * length,
                brush_option, brush_force, x, y
            )
            height = 0.5 * (h[i, j] + h[i, j + 1])
            v[i, j] += dt * force.y / (density * height)

u_augmented_x = ti.field(dtype=ti.f32, shape=(M + 1, N))
u_augmented_y = ti.field(dtype=ti.f32, shape=(M - 1, N + 2))
v_augmented_x = ti.field(dtype=ti.f32, shape=(M + 2, N - 1))
v_augmented_y = ti.field(dtype=ti.f32, shape=(M, N + 1))

# The simulation uses slip boundary conditions on all 4 boundaries. That is, the velocity component normal to the boundaries is
# zero, and the gradient of the velocity component parallel to the boundary is 0.
@ti.kernel
def compute_augmented_velocities():
    for i, j in ti.ndrange(M + 2, N + 2):
        if i < u_augmented_x.shape[0] and j < u_augmented_x.shape[1]:
            u_augmented_x[i, j] = 0.0 if i == 0 or i == M else u[i - 1, j]
        if i < u_augmented_y.shape[0] and j < u_augmented_y.shape[1]:
            if j == 0:
                u_augmented_y[i, j] = u[i, j]
            elif j == N + 1:
                u_augmented_y[i, j] = u[i, j - 2]
            else:
                u_augmented_y[i, j] = u[i, j - 1]
        if i < v_augmented_x.shape[0] and j < v_augmented_x.shape[1]:
            if i == 0:
                v_augmented_x[i, j] = v[i, j]
            elif i == M + 1:
                v_augmented_x[i, j] = v[i - 2, j]
            else:
                v_augmented_x[i, j] = v[i - 1, j]
        if i < v_augmented_y.shape[0] and j < v_augmented_y.shape[1]:
            v_augmented_y[i, j] = 0.0 if j == 0 or j == N else v[i, j - 1]

u_shifted = ti.field(dtype=ti.f32, shape=(M, N - 1))
v_shifted = ti.field(dtype=ti.f32, shape=(M - 1, N))

@ti.kernel
def compute_shifted_velocities():
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

@ti.kernel
def update_elevation_force():
    for i, j in ti.ndrange(M, N):
        if i < u.shape[0] and j < u.shape[1]:
            elevation_left = bed[i, j] + h[i, j]
            elevation_right = bed[i + 1, j] + h[i + 1, j]
            u[i, j] -= dt * g * (elevation_right - elevation_left) / side
        if i < v.shape[0] and j < v.shape[1]:
            elevation_bottom = bed[i, j] + h[i, j]
            elevation_top = bed[i, j + 1] + h[i, j + 1]
            v[i, j] -= dt * g * (elevation_top - elevation_bottom) / side

h_augmented_x = ti.field(dtype=ti.f32, shape=(M + 1, N))
h_augmented_y = ti.field(dtype=ti.f32, shape=(M, N + 1))

@ti.kernel
def compute_augmented_height():
    for i, j in ti.ndrange(M + 1, N + 1):
        if i < h_augmented_x.shape[0] and j < h_augmented_x.shape[1]:
            h_augmented_x[i, j] = h[0, j] if i == 0 else h[M - 1, j] if i == M else 0.5 * (h[i - 1, j] + h[i, j])
        if i < h_augmented_y.shape[0] and j < h_augmented_y.shape[1]:
            h_augmented_y[i, j] = h[i, 0] if j == 0 else h[i, N - 1] if j == N else 0.5 * (h[i, j - 1] + h[i, j])

@ti.kernel
def update_height():
    for i, j in h:
        dhu_dx = (h_augmented_x[i + 1, j] * u_augmented_x[i + 1, j] - h_augmented_x[i, j] * u_augmented_x[i, j]) / side
        dhv_dy = (h_augmented_y[i, j + 1] * v_augmented_y[i, j + 1] - h_augmented_y[i, j] * v_augmented_y[i, j]) / side
        h[i, j] -= dt * (dhu_dx + dhv_dy)

velocity = ti.Vector.field(2, dtype=ti.f32, shape=(M, N))

@ti.kernel
def compute_velocity_field():
    for i, j in velocity:
        velocity[i, j].x = 0.5 * (u_augmented_x[i, j] + u_augmented_x[i + 1, j])
        velocity[i, j].y = 0.5 * (v_augmented_y[i, j] + v_augmented_y[i, j + 1])

image = ti.Vector.field(3, dtype=ti.f32, shape=(M + CONTROL_PANEL_WIDTH, N))

@ti.kernel
def shader(shader_option: int, shader_range: float):
    RED = ti.Vector([0.75, 0.01, 0.02])
    BLUE = ti.Vector([0.13, 0.32, 0.86])
    GRAY = ti.Vector([0.90, 0.90, 0.90])
    for i, j in image:
        if i >= M:
            image[i, j] = ti.Vector([0.5, 0.5, 0.5])
            continue
        val = 0.0
        if shader_option == SHADER_H:
            val = h[i, j]
        elif shader_option == SHADER_ELEVATION:
            val = h[i, j] + bed[i, j]
        elif shader_option == SHADER_U:
            val = velocity[i, j].x
        elif shader_option == SHADER_V:
            val = velocity[i, j].y

        if val > 0:
            t = tm.clamp(val / shader_range, 0.0, 1.0)
            image[i, j] = RED * t + GRAY * (1 - t)
        else:
            t = tm.clamp(-val / shader_range, 0.0, 1.0)
            image[i, j] = BLUE * t + GRAY * (1 - t)

#-- Main Code --#
def physical_simulation(mouse_x: float, mouse_y: float, last_x: float, last_y: float, brush_option: int, brush_force: float):
    if not np.isnan(mouse_x) and not np.isnan(mouse_y) and not np.isnan(last_x) and not np.isnan(last_y):
        update_external_force(mouse_x, mouse_y, last_x, last_y, brush_option, brush_force)
    compute_augmented_velocities()
    compute_shifted_velocities()
    compute_advection()
    update_advection()
    update_elevation_force()
    compute_augmented_velocities()
    compute_augmented_height()
    update_height()
    compute_velocity_field()

WINDOW_WIDTH = M + CONTROL_PANEL_WIDTH
window = ti.ui.Window(name="Shallow Water Simulation", res = (WINDOW_WIDTH, N), fps_limit=200)
canvas = window.get_canvas()
gui = window.get_gui()
shader_option = SHADER_ELEVATION
shader_range, shader_range_min, shader_range_max = default_shader_value_range(shader_option)
brush_option = BRUSH_FORCE
brush_force, brush_force_min, brush_force_max = default_brush_force_range(brush_option)
mouse_filter = LowPassSignalFilter()
last_x, last_y = tm.nan, tm.nan

precompute_bed_and_height()
frame = 0
while window.running:
    # Event handling
    mouse_x_target, mouse_y_target = tm.nan, tm.nan
    for e in window.get_events(ti.ui.PRESS):
        if e.key == ti.ui.ESCAPE:
            break
    if window.is_pressed(ti.ui.LMB):
        x, y = window.get_cursor_pos()
        if x < M / WINDOW_WIDTH:
            mouse_x_target, mouse_y_target = x * WINDOW_WIDTH / M, y
    if window.is_pressed(ti.ui.RMB):
        x, y = window.get_cursor_pos()
        if x < M / WINDOW_WIDTH:
            mouse_x_target, mouse_y_target = x * WINDOW_WIDTH / M, y

    # Physical steps
    for _ in range(physical_steps_per_frame):
        mouse_filter.update(ti.Vector([mouse_x_target, mouse_y_target]), 0.05)
        mouse_x, mouse_y = mouse_filter.value
        physical_simulation(mouse_x, mouse_y, last_x, last_y, brush_option, brush_force)
        last_x, last_y = mouse_x, mouse_y

    # Shading
    shader(shader_option, shader_range)
    ti.sync()
    canvas.set_image(image)
    with gui.sub_window(
        "Simulation Controls",
        x = M / WINDOW_WIDTH,
        y = 0.0,
        width = CONTROL_PANEL_WIDTH / WINDOW_WIDTH,
        height = 1.0,
    ):
        gui.text("Shallow Water Simulation")
        gui.text(f"frame: {frame}")
        gui.text(f"time: {frame * physical_steps_per_frame * dt:.4f}")
        
        gui.text("Shader options:")
        for current_shader_option, description in zip(
            [SHADER_H, SHADER_ELEVATION, SHADER_U, SHADER_V],
            ["Water height", "Elevation (water height + bed height)", "X velocity (U)", "Y velocity (V)"],
        ):
            if gui.checkbox(description, shader_option == current_shader_option):
                if shader_option != current_shader_option:
                    shader_range, shader_range_min, shader_range_max = default_shader_value_range(current_shader_option)
                shader_option = current_shader_option

        shader_range = gui.slider_float("Value range", shader_range, shader_range_min, shader_range_max)

        gui.text("Brush options:")
        for current_brush_option, description in zip(
            [BRUSH_FORCE, BRUSH_DIVERGENCE],
            ["Drag to apply force", "Click to create ripple"]
        ):
            if gui.checkbox(description, brush_option == current_brush_option):
                if brush_option != current_brush_option:
                    brush_force, brush_force_min, brush_force_max = default_brush_force_range(current_brush_option)
                brush_option = current_brush_option

        brush_force = gui.slider_float("Brush force", brush_force, brush_force_min, brush_force_max)

    window.show()
    frame += 1
