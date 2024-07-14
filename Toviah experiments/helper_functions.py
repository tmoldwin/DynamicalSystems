import numpy as np


def generate_cell_bounds(cell_counts):
    cell_bounds = {}
    start = 0
    for cell_type, count in cell_counts.items():
        end = start + count
        cell_bounds[cell_type] = (start, end)
        start = end
    return cell_bounds


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


def gen_large_matrix(cell_bounds, submats):
    # Calculate the total size of the matrix
    total_size = sum(bound[1] - bound[0] for bound in cell_bounds.values())
    # Initialize the large matrix with zeros
    large_matrix = np.zeros((total_size, total_size))

    # Generate and place each submatrix
    for (cell_type_from, cell_type_to), specs in submats.items():
        # Use cell_bounds to determine the dimensions and starting positions for submatrices
        start_row, end_row = cell_bounds[cell_type_to]
        start_col, end_col = cell_bounds[cell_type_from]
        n_rows = end_row - start_row
        n_cols = end_col - start_col

        # Generate the submatrix
        submatrix = generate_mat(n_cols, n_rows, specs['dist'], specs['params'], specs['sign'])

        # Place the submatrix in the large matrix
        large_matrix[start_row:end_row, start_col:end_col] = submatrix

    return large_matrix


def random_binary(N, sparsity=0.1):
    matrix = np.zeros((N, N))
    num_ones = int(sparsity * N * N)
    indices = np.random.choice(N * N, num_ones, replace=False)
    np.put(matrix, indices, 1)
    return matrix
