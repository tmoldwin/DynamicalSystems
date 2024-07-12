import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import expm
from matplotlib.ticker import FuncFormatter

def format_tick(value, tick_number):
    if abs(value) >= 1000:
        return f'{int(value):.1e}'
    else:
        return f'{round(value,2)}'

formatter = FuncFormatter(format_tick)




def BFS(x, A, **kwargs):
    return np.dot(A, x)
def decaying_cascade(x, A, Tau = 2, **kwargs):
    return np.dot(A, x) - x / Tau

def cascade(x, A):
    return np.dot(A,x)





def r_BFS(A,t):
    return np.linalg.matrix_power(A,t)

def r_diffusion(A, t, diffusion_vec):
    return A**t

def r_random_walk(A, t):
    P = A / A.sum(axis=0)
    return np.linalg.matrix_power(P,t)

def r_cascade(A, t):
    return expm(A * t)

def r_decaying_cascade(A, t, Tau=2):
    # J = A - np.eye(A.shape[0]) / Tau
    # J0 = -1/Tau
    # return expm(J * t) - np.exp(J0 * t)
    B = A - (np.eye(A.shape[0]) / Tau)
    return expm(B * t)


def dynamical_simulator(T, x0, W, func, dt='step', plasticity_func = None, sim_params = {}, plasticity_params={}, asymptotic = false):
    N = len(x0)
    W_tensor = []
    W_tensor.append(W)
    if dt != 'step':
        # Continuous dynamics approximation case (not used here but kept for completeness)
        times = np.arange(0, T-1 + dt, dt)
        num_steps = len(times)
        X = np.zeros((N, num_steps))
        X[:, 0] = x0
        for i in range(1, num_steps):
            X[:, i] = X[:, i - 1] + dt * func(X[:, i - 1], W, **kwargs)
            if not plasticity_func is None:
                if not asymptotic:
                    W = W + dt * plasticity_func()
                else:
                    pass
            W_tensor.append(W)
    else:
        # Discrete dynamics case
        times = np.arange(T)  # Corrected to use arange for integer steps
        X = np.zeros((N, T))
        X[:, 0] = x0
        for i in range(1, T):
            X[:, i] = func(X[:, i - 1], A, **kwargs)

    return times, X

def plot_dynamical_sim(times, x, title='', axis = None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[0]))  # Assuming x_dif is 2D
    for i in range(x.shape[0]):
        ax.plot(times, x[i,:], **plot_kwargs, c = colors[i])
    return ax

def plot_weight_mat(A, title='', axis = None):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
    ax.imshow(A, cmap='coolwarm', norm = mcolors.TwoSlopeNorm(vcenter = 0))
    return ax

def plot_rs(times, rfuncs, axes=None, title=False):
    # Calculate global min and max
    global_min = np.min(rfuncs)
    global_max = np.max(rfuncs)

    if axes is None:
        fig, axes = plt.subplots(len(times))
    for i in range(len(times)):
        # Pass global_min and global_max to imshow
        axes[i].imshow(rfuncs[i], cmap='bwr', norm=mcolors.TwoSlopeNorm(vcenter=1e-100))
        if title:
            axes[i].set_title(f'R^{times[i]}')
    return axes

def plot_sum_rs(times, rfuncs, A, title=None, axes=None):
    if axes is None:
        fig, axes = plt.subplots(len(times))
    for i in range(len(times)):
        axes[i].imshow(rfuncs[i], cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vcenter=0))
        if title:
            axes[i].set_title(f'R^{times[i]}')
    return axes

def difs_plot(x, x_alt, rfuncs, times, r_times, ax,  **plot_kwargs):
    x_dif = x_alt - x
    colors = plt.cm.viridis(np.linspace(0, 1, x_dif.shape[0]))  # Assuming x_dif is 2D
    x_dif_init = x_dif[:,0]
    predicted = np.array([np.dot(rfuncs[r_time], x_dif_init) for r_time in r_times]).T
    for i in range(x_dif.shape[0]):
        x_dif_row = x_dif[i,:]
        predicted_row = predicted[i,:]
        # Plot x_dif for this row
        ax.plot(times, x_dif_row, color=colors[i], **plot_kwargs)
        # Plot predicted for this row with the same color
        ax.plot(r_times, predicted_row, linestyle='--', color=colors[i], **plot_kwargs)

# response_times, x0, A, rfuncs[key]
def plot_analytical_solution(response_times, x0, rfuncs, axes):
    colors = plt.cm.viridis(np.linspace(0, 1, x0.shape[0]))  # Assuming x_dif is 2D
    prediction = np.zeros((len(x0), len(response_times)))
    for i in range(len(response_times)):
        prediction[:,i] = np.dot(rfuncs[i], x0)
    for i in range(len(x0)):
        axes.plot(response_times, prediction[i,:], linestyle = ':', c = colors[i])

def plot_eigenvalues(response_matrices, times, ax=None):
    # Initialize a list to store eigenvalues
    eigenvalues_list = []
    # Calculate eigenvalues for each response matrix
    eigenvector_list = []
    for matrix in response_matrices:
        # Check if the matrix contains any NaN or infinity values
        if not np.isfinite(matrix).all():
            print("Warning: Matrix contains NaN or infinity values. Replacing eigenvalues with NaN.")
            eigenvalues = np.full(matrix.shape[0], np.nan)  # Create an array of NaNs with the same size as the matrix
        else:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
        print(eigenvectors)

        eigenvalues_list.append(eigenvalues)
        eigenvector_list.append(eigenvectors/np.linalg.norm(
            eigenvectors, axis=0))  # Normalize eigenvectors to have unit length
        print(eigenvectors)

    eig_fig, eig_ax = plt.subplots(2,len(eigenvector_list))
    for i in range(len(eigenvector_list)):
        eig_ax[0,i].imshow(np.real(eigenvector_list[i]))
        eig_ax[1,i].imshow(np.imag(eigenvector_list[i]))

    # Plot eigenvalues over time
    for i in range(len(eigenvalues_list[0])):  # Assuming all matrices have the same size
        eigenvalue_real_parts = [eigenvalues[i].real for eigenvalues in eigenvalues_list]
        eigenvalue_imag_parts = [eigenvalues[i].imag for eigenvalues in eigenvalues_list]
        ax.plot(eigenvalue_real_parts, eigenvalue_imag_parts, label=f'eigenvalue {i+1}')

        # Add arrowheads to indicate direction of trajectory over time
        for j in range(len(times) - 1):
            start_x = eigenvalue_real_parts[j]
            start_y = eigenvalue_imag_parts[j]
            end_x = eigenvalue_real_parts[j+1] - start_x
            end_y = eigenvalue_imag_parts[j+1] - start_y
            ax.quiver(start_x, start_y, end_x, end_y, angles='xy', scale_units='xy', scale=1, width=0.008)


def create_dale_matrix(size, e_i_balance):
    """
    Create a Dale's law-respecting matrix with a given size and E-I balance.

    Parameters:
    size (int): The size of the matrix.
    e_i_balance (float): The balance between excitatory and inhibitory columns.
                         This should be a value between 0 and 1, where 0 means all inhibitory,
                         1 means all excitatory, and 0.5 means half excitatory and half inhibitory.

    Returns:
    numpy.ndarray: The generated Dale's law-respecting matrix.
    """
    # Determine the number of excitatory and inhibitory columns
    num_excitatory = int(size * e_i_balance)
    num_inhibitory = size - num_excitatory

    # Generate excitatory and inhibitory columns
    excitatory_columns = np.random.rand(num_excitatory, size)
    inhibitory_columns = -np.random.rand(num_inhibitory, size)

    # Concatenate excitatory and inhibitory columns to form the final matrix
    matrix = np.concatenate((excitatory_columns, inhibitory_columns), axis=0).T

    return matrix

def scatterplot_eigenvalues(A, ax):
    """
    Scatterplot the eigenvalues of the matrix A on the complex plane.

    Parameters:
    A (numpy.ndarray): The matrix whose eigenvalues to plot.
    """
    # Calculate the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(A)

    # Separate the real and imaginary parts of the eigenvalues
    real_parts = eigenvalues.real
    imaginary_parts = eigenvalues.imag

    # Create the scatterplot
    ax.scatter(real_parts, imaginary_parts)
    ax.axhline(0, color='black', linewidth=0.1, linestyle = '--')
    ax.axvline(0, color='black', linewidth=0.1, linestyle = '--')

