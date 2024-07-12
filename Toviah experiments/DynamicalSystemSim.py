import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import expm
from matplotlib.ticker import FuncFormatter
import plotting_functions as pf

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

def diffusion(x, A, diffusion_vec = None):
    D = np.diag(np.sum(A, axis=1))
    L = A - D
    return np.dot(L, x)
def random_walk(x, A, **kwargs):
    # x contains the number of agents on each node
    # Divide each column of A by the sum of the column.
    # This ensures that the probability of moving from one node to the next adds to one.
    epsilon = 1e-10  # small constant to prevent division by zero
    P = A / (A.sum(axis=0) + epsilon)
    x_next = np.dot(P, x)
    return x_next

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

def r_diffusion(A, t):
    D = np.diag(np.sum(A, axis=1))
    L = A - D
    return expm(L * t)


def dynamical_simulator(T, x0, A, func, dt='step', **kwargs):
    N = len(x0)
    if dt != 'step':
        # Continuous dynamics approximation case (not used here but kept for completeness)
        times = np.arange(0, T-1 + dt, dt)
        num_steps = len(times)
        X = np.zeros((N, num_steps))
        X[:, 0] = x0
        for i in range(1, num_steps):
            X[:, i] = X[:, i - 1] + dt * func(X[:, i - 1], A, **kwargs)
    else:
        # Discrete dynamics case
        times = np.arange(T)  # Corrected to use arange for integer steps
        X = np.zeros((N, T))
        X[:, 0] = x0
        for i in range(1, T):
            X[:, i] = func(X[:, i - 1], A, **kwargs)

    return times, X





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

def simulate_all(simulations, T, num_nodes, x0, A, response_times):
    fig, axes = plt.subplots(len(simulations), 3 + len(response_times_to_plot), figsize=(20, 20))

    for i, (key, sim) in enumerate(simulations.items()):
        colors = plt.cm.viridis(np.linspace(0, 1, len(response_times)))
        plot_weight_mat(A, title=key, axis=axes[i][0])
        print(key)
        times, x = dynamical_simulator(T, x0, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
        rfuncs = [sim['rfunc'](A, t, **sim['kwargs']) for t in response_times]
        plot_dynamical_sim(times, x, title=key, axis=axes[i][1])
        plot_analytical_solution(response_times, x0, rfuncs, axes=axes[i][1])

        times, x_alt = dynamical_simulator(T, x0_alt, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
        # Uncomment the next line if you want to plot the dynamical simulation for x_alt
        # plot_dynamical_sim(times_alt, x_alt, title=key + " alt", axis=axes[i][1], linestyle='--')
        print('plotrs')
        plot_rs(respones_times_to_plot, [rfuncs[i] for i in respones_times_to_plot], axes=axes[i][2:])
        difs_plot(x, x_alt, rfuncs, times, response_times, ax=axes[i][-1])

    plt.tight_layout()
    plt.show()

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



if __name__ == "__main__":
    # # T = 10
    # # num_nodes = 4
    # # x0 = 1 * np.ones((num_nodes,)) +np.random.randn(num_nodes)
    # # # x0_alt = x0.copy()
    # # # x0_alt[0] += 1
    # # A = np.random.randn(num_nodes, num_nodes)
    # # response_times = range(0, T)
    # # respones_times_to_plot = [response_times[1], response_times[int(T/2)], response_times[-1]]
    # #
    # # # Consolidate all information into a single dictionary
    # # simulations = {
    # #     'BFS': {'func': BFS, 'rfunc': r_BFS, 'kwargs': {}, 'dt': 'step'},
    # #     'random_walk': {'func': random_walk, 'rfunc': r_random_walk, 'kwargs': {}, 'dt': 'step'},
    # #     'cascade': {'func': cascade, 'rfunc': r_cascade, 'kwargs': {}, 'dt': 0.001},
    # #     'decaying_cascade': {'func': decaying_cascade, 'rfunc': r_decaying_cascade, 'kwargs': {'Tau': 0.55}, 'dt': 0.0001},
    # #     'diffusion': {'func': diffusion, 'rfunc': r_diffusion, 'kwargs': {},
    # #                   'dt': 0.001}
    # #     # Add continuous cascade case
    # # }
    # #
    # # fig, axes = plt.subplots(len(simulations), 3 + len(respones_times_to_plot), figsize=(20, 20))
    # #
    # # for i, (key, sim) in enumerate(simulations.items()):
    # #     colors = plt.cm.viridis(np.linspace(0, 1, len(response_times)))
    # #     plot_weight_mat(A, title=key, axis=axes[i][0])
    # #     print(key)
    # #     times, x = dynamical_simulator(T, x0, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
    # #     rfuncs = [sim['rfunc'](A, t, **sim['kwargs']) for t in response_times]
    # #     plot_dynamical_sim(times, x, title=key, axis=axes[i][1])
    # #     plot_analytical_solution(response_times, x0, rfuncs, axes=axes[i][1])
    # #
    # #     #times, x_alt = dynamical_simulator(T, x0_alt, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
    # #     # Uncomment the next line if you want to plot the dynamical simulation for x_alt
    # #     # plot_dynamical_sim(times_alt, x_alt, title=key + " alt", axis=axes[i][1], linestyle='--')
    # #     print('plotrs')
    # #     #plot_rs(respones_times_to_plot, [rfuncs[i] for i in respones_times_to_plot], axes=axes[i][2:])
    # #     #difs_plot(x, x_alt, rfuncs, times, response_times, ax=axes[i][-1])
    # #
    # # plt.tight_layout()
    # # plt.show()
    # T = 10
    # num_nodes = 4
    # x0 = 1 * np.ones((num_nodes,)) +np.random.randn(num_nodes)
    # # x0_alt = x0.copy()
    # # x0_alt[0] += 1
    # A = np.random.randn(num_nodes, num_nodes)
    # response_times = range(0, T)
    # respones_times_to_plot = [response_times[1], response_times[int(T/2)], response_times[-1]]
    #
    # # Consolidate all information into a single dictionary
    # simulations = {
    #     'BFS': {'func': BFS, 'rfunc': r_BFS, 'kwargs': {}, 'dt': 'step'},
    #     'random_walk': {'func': random_walk, 'rfunc': r_random_walk, 'kwargs': {}, 'dt': 'step'},
    #     'cascade': {'func': cascade, 'rfunc': r_cascade, 'kwargs': {}, 'dt': 0.001},
    #     'decaying_cascade': {'func': decaying_cascade, 'rfunc': r_decaying_cascade, 'kwargs': {'Tau': 0.55}, 'dt': 0.0001},
    #     'diffusion': {'func': diffusion, 'rfunc': r_diffusion, 'kwargs': {},
    #                   'dt': 0.001}
    #     # Add continuous cascade case
    # }
    #
    # fig, axes = plt.subplots(len(simulations), 3 + len(respones_times_to_plot), figsize=(20, 20))
    #
    # for i, (key, sim) in enumerate(simulations.items()):
    #     colors = plt.cm.viridis(np.linspace(0, 1, len(response_times)))
    #     plot_weight_mat(A, title=key, axis=axes[i][0])
    #     print(key)
    #     times, x = dynamical_simulator(T, x0, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
    #     rfuncs = [sim['rfunc'](A, t, **sim['kwargs']) for t in response_times]
    #     plot_dynamical_sim(times, x, title=key, axis=axes[i][1])
    #     plot_analytical_solution(response_times, x0, rfuncs, axes=axes[i][1])
    #     plot_eigenvalues(rfuncs, response_times, ax = axes[i][2])
    #
    #     #times, x_alt = dynamical_simulator(T, x0_alt, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
    #     # Uncomment the next line if you want to plot the dynamical simulation for x_alt
    #     # plot_dynamical_sim(times_alt, x_alt, title=key + " alt", axis=axes[i][1], linestyle='--')
    #     print('plotrs')
    #     #plot_rs(respones_times_to_plot, [rfuncs[i] for i in respones_times_to_plot], axes=axes[i][2:])
    #     #difs_plot(x, x_alt, rfuncs, times, response_times, ax=axes[i][-1])
    #
    # plt.tight_layout()
    # plt.show()
    T = 5
    num_nodes = 10
    x0 = np.abs(np.random.randn(num_nodes))
    response_times = range(0, T)
    respones_times_to_plot = [response_times[1], response_times[int(T / 2)], response_times[-1]]

    # Define the weight matrices
    weight_matrices = [
        np.abs(np.random.randn(num_nodes, num_nodes)),  # All-positive random matrix
        -np.abs(np.random.randn(num_nodes, num_nodes)),  # All-negative random matrix
        np.random.randn(num_nodes, num_nodes),  # Balanced matrix        # np.abs(np.random.randn(num_nodes, num_nodes)),  # All-positive random matrix
        create_dale_matrix(num_nodes, 0.5),
        create_dale_matrix(num_nodes, 0.8),
        create_dale_matrix(num_nodes, 0.2),
     ]

    # Consolidate all information into a single dictionary
    simulations = {
        'BFS': {'func': BFS, 'rfunc': r_BFS, 'kwargs': {}, 'dt': 'step'},
        'random_walk': {'func': random_walk, 'rfunc': r_random_walk, 'kwargs': {}, 'dt': 'step'},
        'cascade': {'func': cascade, 'rfunc': r_cascade, 'kwargs': {}, 'dt': 0.001},
        'decaying_cascade': {'func': decaying_cascade, 'rfunc': r_decaying_cascade, 'kwargs': {'Tau': 0.2},
                             'dt': 0.0001},
        'diffusion': {'func': diffusion, 'rfunc': r_diffusion, 'kwargs': {}, 'dt': 0.001}
    }

    fig, axes = plt.subplots(len(simulations) + 1, len(weight_matrices), figsize = (20, 22))
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    for j, A in enumerate(weight_matrices):
        pf.plot_weight_mat(A, title=f'Weight Matrix {j + 1}', axis=axes[0][j])
        #scatterplot_eigenvalues(A, ax=axes[0][2 * j + 1])
        for i, (key, sim) in enumerate(simulations.items()):
            colors = plt.cm.viridis(np.linspace(0, 1, len(response_times)))
            times, x = dynamical_simulator(T, x0, A, sim['func'], dt=sim['dt'], **sim['kwargs'])
            rfuncs = [sim['rfunc'](A, t, **sim['kwargs']) for t in response_times]
            pf.plot_dynamical_sim(times, x, title=key, axis=axes[i + 1][j])
            #plot_analytical_solution(response_times, x0, rfuncs, axes=axes[i + 1][j])
            #plot_eigenvalues(rfuncs, response_times, ax=axes[i + 1][2 * j + 1])

    # Assuming 'axes' is a list of Axes objects and 'simulations' is your dictionary of simulations

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, left = 0.1, bottom = 0.08)
    for i, (key, sim) in enumerate(simulations.items()):
        # Your existing code...
        # Add a ylabel to the plot
        axes[i+1][0].set_ylabel(key, rotation=90, verticalalignment='center', labelpad = 20,
                              fontsize=10)
    plt.savefig('test.png')
    plt.show()




