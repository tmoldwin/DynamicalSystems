import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

def plot_weight_mat(A, title='', axis = None):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
    ax.imshow(A, cmap='coolwarm', norm = mcolors.TwoSlopeNorm(vcenter = 0))
    return ax

def plot_dynamical_sim(times, x, title='', axis = None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
    colors = plt.cm.viridis(np.linspace(0, 1, x.shape[0]))  # Assuming x_dif is 2D
    for i in range(x.shape[0]):
        ax.plot(times, x[i,:], **plot_kwargs, c = colors[i])
    return ax

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