import numpy as np
import cupy as cp
from cupyx.scipy.fft import dctn, idctn   # Discrete cosine transformation (n-dimensional)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

class NeumannPoissonSolver:
    """
    Solve A p = b, where A is the unscaled five-point negative Laplacian with homogeneous Neumann boundary conditions.

    The pressure is fixed to have zero mean.
    """

    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        k = cp.arange(M, dtype=cp.float64)
        l = cp.arange(N, dtype=cp.float64)
        eigenvalue_y = 4 * cp.sin(cp.pi * k / (2 * M))**2
        eigenvalue_x = 4 * cp.sin(cp.pi * l / (2 * N))**2
        # Eigenvalue of the mode `mode[i, j] = cos(pi*l*(i+0.5)/N)*cos(pi*k*(j+0.5)/M)`
        eigenvalues = eigenvalue_y[:, None] + eigenvalue_x[None, :]
        eigenvalues[0, 0] = 1.0  # Set the constant mode's denominator to 1.0 to avoid division by 0
        self.eigenvalues = eigenvalues

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        assert b.shape == (self.M, self.N)
        b_hat = dctn(b, type=2, norm="ortho")
        p_hat = b_hat / self.eigenvalues
        p_hat[0, 0] = 0.0   # Manually remove the constant mode
        return idctn(p_hat, type=2, norm="ortho")


# Proceed the Navier-Stokes simulation by time `dt`.
#
# Equation: ∂𝐮/∂t = -(𝐮·∇)𝐮 − (1/ρ)∇p + ν∇²𝐮 + 𝐟
#
# The solver uses MAC grid -- `u` values are placed on vertical cell edges, `v` values are placed on horizontal cell edges,
# and `p` are placed on cell centres. Let the grid shape be (M, N), then the shape of `u` is (M, N-1), `v` is (M-1, N), and
# `p` is (M, N), where increasing the first index increases y and increasing the second index increases x. The velocities
# on the boundaries are specified by the `boundaries` variable.
#
# The cell (0, 0) is at the top left of the grid. X axis goes from left to right, and y axis goes from top to bottom. The x
# axis is divided into N divisions, and the y axis is divided into M divisions.
#
# `boundaries` is a 4 by 2 array specifying the u and v velocities on the top, right, bottom, and left boundaries. It is
# assumed that for each boundary, the velocity at any point on that boundary is the same.
#
# `h` is the grid's side length in metres, and `dt` is the time step in seconds.
def proceed_simulation(
        u: cp.ndarray,
        v: cp.ndarray,
        dt: float,
        h: float,
        viscosity: float,
        density: float,
        boundaries: np.ndarray,
        solver: NeumannPoissonSolver,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    M, N = u.shape
    N += 1
    assert np.all(v.shape == np.array([M-1, N]))

    # Helper functions
    def augment_u_y(u1: cp.ndarray) -> cp.ndarray:
        # The spaces between the first and last row and their closest boundaries are only half a grid length.
        top = cp.full((1, N-1), boundaries[0][0] * 2) - u1[0, :]
        bottom = cp.full((1, N-1), boundaries[2][0] * 2) - u1[-1, :]
        return cp.vstack([top, u1, bottom])

    def augment_v_x(v1: cp.ndarray) -> cp.ndarray:
        left = cp.full((M-1, 1), boundaries[3][1] * 2) - v1[:, 0].reshape((M-1, 1))
        right = cp.full((M-1, 1), boundaries[1][1] * 2) - v1[:, -1].reshape((M-1, 1))
        return cp.hstack([left, v1, right])

    # Compute velocities on different grid positions
    u_augmented_x = cp.hstack([cp.full((M, 1), boundaries[3][0]), u, cp.full((M, 1), boundaries[1][0])])
    u_augmented_y = augment_u_y(u)
    u_shifted = (u_augmented_x[:-1, :-1] + u_augmented_x[1:, :-1] + u_augmented_x[:-1, 1:] + u_augmented_x[1:, 1:]) * 0.25
    v_augmented_x = augment_v_x(v)
    v_augmented_y = cp.vstack([cp.full((1, N), boundaries[0][1]), v, cp.full((1, N), boundaries[2][1])])
    v_shifted = (v_augmented_y[:-1, :-1] + v_augmented_y[1:, :-1] + v_augmented_y[:-1, 1:] + v_augmented_y[1:, 1:]) * 0.25

    # Advection (updated to upwind discretisation)
    du_dx_backward = (u - u_augmented_x[:, :-2]) / h
    du_dx_forward = (u_augmented_x[:, 2:] - u) / h
    du_dx = cp.where(u > 0, du_dx_backward, du_dx_forward)

    du_dy_backward = (u - u_augmented_y[:-2, :]) / h
    du_dy_forward = (u_augmented_y[2:, :] - u) / h
    du_dy = cp.where(v_shifted > 0, du_dy_backward, du_dy_forward)

    dv_dy_backward = (v - v_augmented_y[:-2, :]) / h
    dv_dy_forward  = (v_augmented_y[2:, :] - v) / h
    dv_dy = cp.where(v > 0, dv_dy_backward, dv_dy_forward)

    dv_dx_backward = (v - v_augmented_x[:, :-2]) / h
    dv_dx_forward  = (v_augmented_x[:, 2:] - v) / h
    dv_dx = cp.where(u_shifted > 0, dv_dx_backward, dv_dx_forward)

    advection_u = u * du_dx + v_shifted * du_dy
    advection_v = v * dv_dy + u_shifted * dv_dx
    u -= dt * advection_u
    v -= dt * advection_v

    # Viscosity
    h2 = h * h
    d2u_dx2 = (u_augmented_x[:, 2:] - 2 * u_augmented_x[:, 1:-1] + u_augmented_x[:, :-2]) / h2
    d2u_dy2 = (u_augmented_y[2:, :] - 2 * u_augmented_y[1:-1, :] + u_augmented_y[:-2, :]) / h2
    d2v_dx2 = (v_augmented_x[:, 2:] - 2 * v_augmented_x[:, 1:-1] + v_augmented_x[:, :-2]) / h2
    d2v_dy2 = (v_augmented_y[2:, :] - 2 * v_augmented_y[1:-1, :] + v_augmented_y[:-2, :]) / h2

    u += (viscosity * dt) * (d2u_dx2 + d2u_dy2)
    v += (viscosity * dt) * (d2v_dx2 + d2v_dy2)

    # Pressure (solve a Poisson equation)
    u_augmented_x = cp.hstack([cp.full((M, 1), boundaries[3][0]), u, cp.full((M, 1), boundaries[1][0])])
    v_augmented_y = cp.vstack([cp.full((1, N), boundaries[0][1]), v, cp.full((1, N), boundaries[2][1])])
    du_dx_times_h = cp.diff(u_augmented_x, axis=1)
    dv_dy_times_h = cp.diff(v_augmented_y, axis=0)
    poisson_rhs = -(h * density / dt) * (du_dx_times_h + dv_dy_times_h)   # need to negate the divergence because the Poisson matrix is also negated
    pressure = solver.solve(poisson_rhs)
    pressure -= cp.mean(pressure)

    u -= dt / (h * density) * cp.diff(pressure, axis=1)
    v -= dt / (h * density) * cp.diff(pressure, axis=0)

    # #! DEBUG: check that the fluid is indeed incompressible after pressure correction. Can be removed.
    # u_augmented_x = cp.hstack([cp.full((M, 1), boundaries[3][0]), u, cp.full((M, 1), boundaries[1][0])])
    # v_augmented_y = cp.vstack([cp.full((1, N), boundaries[0][1]), v, cp.full((1, N), boundaries[2][1])])
    # du_dx = cp.diff(u_augmented_x, axis=1) / h
    # dv_dy = cp.diff(v_augmented_y, axis=0) / h
    # poisson_rhs = -(du_dx + dv_dy)
    # # print("-" * 50)
    # # print(u)
    # assert cp.allclose(poisson_rhs, 0, atol=1e-8)
    # #! END DEBUG

    return u, v, pressure

# Compute the velocities at grid centres
#
# `u`, `v`, and `boundaries` follow the same definition as before
def velocities_at_grid_centres(u: cp.ndarray, v: cp.ndarray, boundaries: cp.ndarray, M: int, N: int) -> tuple[cp.ndarray, cp.ndarray]:
    u_augmented_x = cp.hstack([cp.full((M, 1), boundaries[3][0]), u, cp.full((M, 1), boundaries[1][0])])
    v_augmented_y = cp.vstack([cp.full((1, N), boundaries[0][1]), v, cp.full((1, N), boundaries[2][1])])
    u_centred = 0.5 * (u_augmented_x[:, 1:] + u_augmented_x[:, :-1])
    v_centred = 0.5 * (v_augmented_y[1:, :] + v_augmented_y[:-1, :])
    return u_centred, v_centred

# Compute the average of each k times k block of the array.
def block_average(arr: cp.ndarray, k: int) -> cp.ndarray:
    M, N = arr.shape
    assert M % k == 0 and N % k == 0
    return arr.reshape(M // k, k, N // k, k).mean(axis=(1, 3))

# Visualise the lid-driven cavity problem with matplotlib animation
def visualise_matplotlib(
    M: int = 300,
    N: int = 300,
    dt: float = 0.001,
    physical_steps_per_frame: int = 18,
    block_size: int = 5,  # Number of simulation grids per on-screen block
):
    u = cp.zeros((M, N-1))
    v = cp.zeros((M-1, N))
    p = cp.zeros((M, N))
    h = 1.0 / M
    viscosity = 3e-4   # A fictional viscous liquid (about the value of kerosene)
    density = 1e3
    boundaries = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 0.]])
    solver = NeumannPoissonSolver(M, N)

    # Plot the vector field of velocity
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(0.5, N // block_size + 0.5) * block_size * h, np.arange(0.5, M // block_size + 0.5) * block_size * h)
    u_centred, v_centred = velocities_at_grid_centres(u, v, boundaries, M, N)
    q = ax.quiver(
        X, Y,
        block_average(u_centred, block_size).get(), block_average(v_centred, block_size).get(),
        block_average(p, block_size).get(),
        cmap="viridis",
        norm=Normalize(vmin=-4.0, vmax=4.0),
        angles="xy",
        scale_units="xy",
        scale=100.0 / block_size,
    )
    fig.colorbar(q, ax=ax, label="Pressure")
    ax.set_xlim(0, N * h)
    ax.set_ylim(0, M * h)
    ax.set_aspect("equal")
    title = ax.set_title("Frame 0; t = 0.0000")

    # Animation callback
    def update(frame):
        nonlocal u, v, p
        for _ in range(physical_steps_per_frame):
            u, v, p = proceed_simulation(u, v, dt, h, viscosity, density, boundaries, solver)
        u_centred, v_centred = velocities_at_grid_centres(u, v, boundaries, M, N)
        q.set_UVC(block_average(u_centred, block_size).get(), block_average(v_centred, block_size).get(), block_average(p, block_size).get())
        title.set_text(f"Frame {frame}; t = {frame * physical_steps_per_frame * dt:.4f}")

    ani = FuncAnimation(
        fig,
        update,
        frames=None,
        interval=30,
        blit=False,
    )
    plt.show()

if __name__ == "__main__":
    visualise_matplotlib(300, 300, 0.001, 18, 5)
