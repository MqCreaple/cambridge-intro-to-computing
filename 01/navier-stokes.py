import numpy as np
import scipy.sparse as sp
import pyamg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# Build a multigrid solver to solve the Poisson equation with Neumann boundary condition.
#
# Note that the matrix computes the negative Laplacian with the kernel [0, -1, 0; -1, 4, -1; 0, -1, 0] to ensure diagonal dominance.
def build_multigrid_solver(M: int, N: int):
    rows = []
    cols = []
    values = []
    def push_entry(row: int, col: int, value: float):
        rows.append(row)
        cols.append(col)
        values.append(value)
    for i in range(M):
        for j in range(N):
            current = i * N + j
            cnt = 0
            if i != 0:
                push_entry(current, (i - 1) * N + j, -1)
                cnt += 1
            if i != M - 1:
                push_entry(current, (i + 1) * N + j, -1)
                cnt += 1
            if j != 0:
                push_entry(current, current - 1, -1)
                cnt += 1
            if j != N - 1:
                push_entry(current, current + 1, -1)
                cnt += 1
            push_entry(current, current, cnt)
    A = sp.csr_matrix((values, (rows, cols)), shape=(M * N, M * N))
    nullspace = np.ones((M * N, 1))
    solver = pyamg.smoothed_aggregation_solver(A, B=nullspace)
    return A, solver

# Proceed the Navier-Stokes simulation by time `dt`.
#
# The solver uses MAC grid -- `u` values are placed on vertical cell edges, `v` values are placed on horizontal cell edges,
# and `p` are placed on cell centres. Let the grid shape be (M, N), then the shape of `u` is (M, N-1), `v` is (M-1, N), and
# `p` is (M, N). The velocities on the boundaries are specified in the `boundaries` variable.
#
# The cell (0, 0) is at the bottom left of the grid. X axis goes from left to right, and y axis goes from top to bottom.
#
# `boundaries` is a 4 by 2 array specifying the u and v velocities on the top, right, bottom, and left boundaries. It is
# assumed that for each boundary, the velocity at any point on that boundary is the same.
#
# `h` is the grid's side length in metres, and `dt` is the time step in seconds.
def proceed_simulation(
        u: np.ndarray,
        v: np.ndarray,
        dt: float,
        h: float,
        viscosity: float,
        density: float,
        boundaries: np.ndarray,
        solver: pyamg.MultilevelSolver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, N = u.shape
    N += 1
    assert np.all(v.shape == np.array([M-1, N]))

    # Helper functions
    def augment_u_y(u1: np.ndarray) -> np.ndarray:
        # The spaces between the first and last row and their closest boundaries are only half a grid length.
        top = np.full((1, N-1), boundaries[0][0] * 2) - u1[0, :]
        bottom = np.full((1, N-1), boundaries[2][0] * 2) - u1[-1, :]
        return np.vstack([top, u1, bottom])

    def augment_v_x(v1: np.ndarray) -> np.ndarray:
        left = np.full((M-1, 1), boundaries[3][1] * 2) - v1[:, 0].reshape((M-1, 1))
        right = np.full((M-1, 1), boundaries[1][1] * 2) - v1[:, -1].reshape((M-1, 1))
        return np.hstack([left, v1, right])

    # Compute velocities on different grid positions
    u_augmented_x = np.hstack([np.full((M, 1), boundaries[3][0]), u, np.full((M, 1), boundaries[1][0])])
    u_augmented_y = augment_u_y(u)
    u_shifted = (u_augmented_x[:-1, :-1] + u_augmented_x[1:, :-1] + u_augmented_x[:-1, 1:] + u_augmented_x[1:, 1:]) * 0.25
    v_augmented_x = augment_v_x(v)
    v_augmented_y = np.vstack([np.full((1, N), boundaries[0][1]), v, np.full((1, N), boundaries[2][1])])
    v_shifted = (v_augmented_y[:-1, :-1] + v_augmented_y[1:, :-1] + v_augmented_y[:-1, 1:] + v_augmented_y[1:, 1:]) * 0.25

    # Advection
    du_dx = np.diff(u_augmented_x, axis=1) / h
    du_dx = 0.5 * (du_dx[:, 1:] + du_dx[:, :-1])
    du_dy = np.diff(u_augmented_y, axis=0) / h
    du_dy = 0.5 * (du_dy[1:, :] + du_dy[:-1, :])

    dv_dx = np.diff(v_augmented_x, axis=1) / h
    dv_dx = 0.5 * (dv_dx[:, 1:] + dv_dx[:, :-1])
    dv_dy = np.diff(v_augmented_y, axis=0) / h
    dv_dy = 0.5 * (dv_dy[1:, :] + dv_dy[:-1, :])

    u -= (u * du_dx + v_shifted * du_dy) * dt
    v -= (v * dv_dy + u_shifted * dv_dx) * dt

    # Viscosity
    h2 = h * h
    d2u_dx2 = (u_augmented_x[:, 2:] - 2 * u_augmented_x[:, 1:-1] + u_augmented_x[:, :-2]) / h2
    d2u_dy2 = (u_augmented_y[2:, :] - 2 * u_augmented_y[1:-1, :] + u_augmented_y[:-2, :]) / h2
    d2v_dx2 = (v_augmented_x[:, 2:] - 2 * v_augmented_x[:, 1:-1] + v_augmented_x[:, :-2]) / h2
    d2v_dy2 = (v_augmented_y[2:, :] - 2 * v_augmented_y[1:-1, :] + v_augmented_y[:-2, :]) / h2

    u += (viscosity * dt) * (d2u_dx2 + d2u_dy2)
    v += (viscosity * dt) * (d2v_dx2 + d2v_dy2)

    # Pressure (solve a Poisson equation)
    u_augmented_x = np.hstack([np.full((M, 1), boundaries[3][0]), u, np.full((M, 1), boundaries[1][0])])
    v_augmented_y = np.vstack([np.full((1, N), boundaries[0][1]), v, np.full((1, N), boundaries[2][1])])
    du_dx = np.diff(u_augmented_x, axis=1) / h
    dv_dy = np.diff(v_augmented_y, axis=0) / h
    poisson_rhs = -(h2 * density / dt) * (du_dx + dv_dy)   # need to negate the divergence because the Poisson matrix is also negated
    pressure = solver.solve(poisson_rhs.ravel(), tol=1e-9).reshape((M, N))

    u -= dt / (h * density) * np.diff(pressure, axis=1)
    v -= dt / (h * density) * np.diff(pressure, axis=0)

    #! DEBUG: check that the fluid is indeed incompressible after pressure correction. Can be removed.
    u_augmented_x = np.hstack([np.full((M, 1), boundaries[3][0]), u, np.full((M, 1), boundaries[1][0])])
    v_augmented_y = np.vstack([np.full((1, N), boundaries[0][1]), v, np.full((1, N), boundaries[2][1])])
    du_dx = np.diff(u_augmented_x, axis=1) / h
    dv_dy = np.diff(v_augmented_y, axis=0) / h
    poisson_rhs = -(du_dx + dv_dy)
    # print("-" * 50)
    # print(u)
    assert np.allclose(poisson_rhs, 0)
    #! END DEBUG

    return u, v, pressure

# Compute the velocities at grid centres
#
# `u`, `v`, and `boundaries` follow the same definition as before
def velocities_at_grid_centres(u: np.ndarray, v: np.ndarray, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_augmented_x = np.hstack([np.full((M, 1), boundaries[3][0]), u, np.full((M, 1), boundaries[1][0])])
    v_augmented_y = np.vstack([np.full((1, N), boundaries[0][1]), v, np.full((1, N), boundaries[2][1])])
    u_centred = 0.5 * (u_augmented_x[:, 1:] + u_augmented_x[:, :-1])
    v_centred = 0.5 * (v_augmented_y[1:, :] + v_augmented_y[:-1, :])
    return u_centred, v_centred

# Compute the average of each k times k block of the array.
def block_average(arr: np.ndarray, k: int) -> np.ndarray:
    M, N = arr.shape
    assert M % k == 0 and N % k == 0
    return arr.reshape(M // k, k, N // k, k).mean(axis=(1, 3))

if __name__ == "__main__":
    M, N = 300, 300
    u = np.zeros((M, N-1))
    v = np.zeros((M-1, N))
    p = np.zeros((M, N))
    dt = 0.005
    h = 0.01
    viscosity = 1e-6  # Water at 20 degrees Celsius
    density = 1e3
    boundaries = np.array([[1., 0.], [0., 0.], [0., 0.], [0., 0.]])
    A, solver = build_multigrid_solver(M, N)

    # Plot the vector field of velocity
    K = 10   # Block size
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.arange(0.5, N // K + 0.5) * K * h, np.arange(0.5, M // K + 0.5) * K * h)
    u_centred, v_centred = velocities_at_grid_centres(u, v, boundaries)
    q = ax.quiver(
        X, Y,
        block_average(u_centred, K), block_average(v_centred, K),
        block_average(p, K),
        cmap="viridis",
        norm=Normalize(vmin=-4.0, vmax=4.0),
        angles="xy",
        scale_units="xy",
        scale=2.0 / K,
    )
    fig.colorbar(q, ax=ax, label="Pressure")
    ax.set_xlim(0, N * h)
    ax.set_ylim(0, M * h)
    ax.set_aspect("equal")
    title = ax.set_title("Frame 0")

    # Animation callback
    def update(frame):
        global u, v, p
        u, v, p = proceed_simulation(u, v, dt, h, viscosity, density, boundaries, solver)
        u_centred, v_centred = velocities_at_grid_centres(u, v, boundaries)
        q.set_UVC(block_average(u_centred, K), block_average(v_centred, K), block_average(p, K))
        title.set_text(f"Frame {frame}")

    ani = FuncAnimation(
        fig,
        update,
        frames=None,
        interval=30,
        blit=False,
    )
    plt.show()
