import networkx as nx

import DynamicalSystemSim as dss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.linalg import expm
import plotting_functions as pf

#Several functions to create binary matrices of different types. E.g. Scale free, ring lattice, hierachical, etc., modular, etc.

def generate_mat(n_inputs, n_outputs, dist, dist_params, sign=1):
    """
    Generates a random matrix with elements drawn from a specified distribution.

    Parameters:
    - n_rows (int): Number of rows in the matrix.
    - n_cols (int): Number of columns in the matrix.
    - dist (callable): Probability distribution function from numpy.random.*.
    - dist_params (dict): Parameters to pass to the distribution function.
    - sign (int): If 1, the matrix is positive. If -1, the matrix is negative.

    Returns:
    - np.ndarray: Generated matrix.
    """
    # Generate the matrix with random values based on the specified distribution and parameters
    print(dist_params)
    matrix = dist(size=(n_outputs, n_inputs), **dist_params)

    # Apply the sign to the matrix
    matrix *= sign

    return matrix


def gen_large_matrix(cell_counts, submats):
    # Calculate the total size of the matrix
    total_size = sum(cell_counts.values())
    # Initialize the large matrix with zeros
    large_matrix = np.zeros((total_size, total_size))

    # Keep track of the starting index for each cell type
    start_indices = {}
    current_index = 0
    for cell_type, count in cell_counts.items():
        start_indices[cell_type] = current_index
        current_index += count

    # Generate and place each submatrix
    for (cell_type_from, cell_type_to), specs in submats.items():
        # Ensure n_rows is for postsynaptic and n_cols for presynaptic
        n_rows = cell_counts[cell_type_to]
        n_cols = cell_counts[cell_type_from]
        submatrix = generate_mat(n_cols, n_rows, specs['dist'], specs['params'], specs['sign'])

        start_row = start_indices[cell_type_to]
        start_col = start_indices[cell_type_from]
        # Ensure the dimensions match when placing the submatrix
        large_matrix[start_row:start_row + n_rows, start_col:start_col + n_cols] = submatrix

    return large_matrix



def random_binary(N, sparsity=0.1):
    matrix = np.zeros((N, N))
    num_ones = int(sparsity * N * N)
    indices = np.random.choice(N * N, num_ones, replace=False)
    np.put(matrix, indices, 1)
    return matrix


def Block_Matrix(N_E, N_I, randfuncs, sparsity=0.1):
    pass;


# def feed_forward(layer_sizes, noise, size = None):
#     N = np.sum(layer_sizes)
#     matrix = np.zeros((N, N))
#
#     start = 0
#     for i, size in enumerate(layer_sizes[:-1]):
#         end = start + size
#         next_start = end
#         next_end = next_start + layer_sizes[i + 1]
#         matrix[next_start:next_end, start:end] = 1  # Swapped indices
#         start = end
#
#     num_noise = int(noise * N * N)
#     noise_indices = np.random.choice(N * N, num_noise, replace=False)
#     np.put(matrix, noise_indices, 1)
#     return matrix






# response_times = range(0, T)
# respones_times_to_plot = [response_times[1], response_times[int(T / 2)], response_times[-1]]

# matrix_types = [scale_free, ring_lattice, hierarchical, modular, random]
# weight_matrices = [f(N, 2) for f in matrix_types]

#visualize each matrix, the connectivity, and then the dynamics
# Define matrix generating functions and their parameters

# Fixed code snippet with customized parameters for each matrix type
layer_sizes = [10,10,10]
T = 5
N = sum(layer_sizes)
cell_counts = {'L5': 10, 'L4':20, 'Mini': 10}
N = sum(cell_counts.values())
x0 = np.abs(np.random.randn(N))



submats = {('L5', 'L4'): {'dist': np.random.exponential, 'params': {'scale' :0.1}, 'sign': 1},
           ('L4','L5'): {'dist': np.random.exponential, 'params': {'scale' :0.1}, 'sign': 1},
           ('Mini', 'L4'): {'dist': np.random.exponential, 'params': {'scale':0.1}, 'sign': -1}}

matrices = [gen_large_matrix(cell_counts, submats)]
# matrix_parameters = [{'func':feed_forward, 'params': {'layer_sizes': layer_sizes,
#                                                        'noise': i * 0.1}}
#                      for i in range(10)]



fig, axes = plt.subplots(4, np.max([len(matrices),2]), figsize=(20, len(matrices)))
axes = np.atleast_2d(axes)
# Adjusted loop to generate matrices with correct parameters
for i, A in enumerate(matrices):

    # Continue with visualization and simulation as before
# Setup the figure for plotting

    # Visualize the weight matrix
    pf.plot_weight_mat(A, title=f'experiment', axis=axes[0][i])

    # Simulate dynamics
    times, x = dss.dynamical_simulator(T, x0, A, dss.BFS, dt='step')

    # Visualize dynamics
    pf.plot_dynamical_sim(times, x, axis=axes[2][i])

    # Scatterplot eigenvalues
    pf.scatterplot_eigenvalues(A, ax=axes[1][i])

plt.tight_layout()
plt.show()

