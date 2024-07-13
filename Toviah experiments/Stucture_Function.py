import networkx as nx

import DynamicalSystemSim as dss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.linalg import expm
import plotting_functions as pf
import helper_functions as hf

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



cell_counts_1 = {'L5': 10, 'L4':20, 'Mini': 10}
submats_1 = {('L5', 'L4'): {'dist': np.random.exponential, 'params': {'scale' :0.1}, 'sign': 1},
           ('L4','L5'): {'dist': np.random.exponential, 'params': {'scale' :0.1}, 'sign': 1},
           ('Mini', 'L4'): {'dist': np.random.exponential, 'params': {'scale':0.1}, 'sign': -1}}

cell_counts_layered = {'L1': 20, 'L2': 40, 'L2I':20, 'L3': 20, }
submats_layered = {
    ('L1', 'L2'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
    ('L2I', 'L2'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': -1},
    ('L2', 'L2I'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
    ('L2', 'L3'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
    #('L3', 'L1'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
}

cell_counts_balanced = {'E': 100, 'I': 130}
submats_balanced = {('E', 'I'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
                    ('I', 'E'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': -1},
                    ('E', 'E'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
                    ('I', 'I'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': -1}}


cell_counts = cell_counts_layered
cell_bounds = hf.generate_cell_bounds(cell_counts)
print(cell_bounds)
submats = submats_layered
T = 25
N = sum(cell_counts.values())
x0 = np.abs(np.random.uniform(0,1, N))

matrices = [gen_large_matrix(cell_counts, submats)]
# matrix_parameters = [{'func':feed_forward, 'params': {'layer_sizes': layer_sizes,
#                                                        'noise': i * 0.1}}
#                      for i in range(10)]



fig, axes = plt.subplots(4, len(matrices), figsize = (6,5))
# Adjusted loop to generate matrices with correct parameters
for i, A in enumerate(matrices):
    # Continue with visualization and simulation as before
# Setup the figure for plotting

    # Visualize the weight matrix
    pf.plot_weight_mat(A, title=f'experiment', cell_bounds = cell_bounds, axis=axes[0])

    # Simulate dynamics
    times, x, spikes, input_by_type = dss.dynamical_simulator(T, x0, A, cell_bounds = cell_bounds, tau_leak= 5, activation = dss.threshold_activation)

    # Visualize dynamics
    pf.plot_dynamical_sim(times, x, axis=axes[2], cell_counts=cell_counts)
    pf.plot_dynamical_sim_raster(times, spikes, axis=axes[3], cell_counts=cell_counts)

    # Scatterplot eigenvalues
    pf.scatterplot_eigenvalues(A, ax=axes[1])
    # axes[1,1].hist(A.flatten(), bins = 20)
    # axes[2,1].hist(A.T)
    # axes[3,1].hist(np.sum(A, axis = 1), bins = 20, fill = False)
    plt.tight_layout()
    pf.all_inputs_by_type(times, input_by_type, cell_bounds)
    print(x)


plt.tight_layout()
plt.show()

