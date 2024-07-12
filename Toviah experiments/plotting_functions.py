import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

def plot_weight_mat(A, cell_counts = None, title='', axis=None):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    tick_positions = []
    tick_labels = []
    # Calculate tick positions and labels
    if not cell_counts is None:
        start_indices = {}
        current_index = 0
        for cell_type, count in cell_counts.items():
            start_indices[cell_type] = current_index
            tick_positions.append(current_index)
            tick_labels.append(cell_type)
            current_index += count

        # Adjust the first tick position to be at the half of the first count

        # Set the ticks and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90)  # Rotate labels if needed
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    # Plot the matrix
    ax.matshow(A, cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vcenter=0))
    ax.set_title(title)

    return ax

def plot_dynamical_sim(times, x, title='', axis=None, cell_counts=None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Calculate the total number of cells and types for color mapping
    total_cells = x.shape[0]
    cell_types = list(cell_counts.keys())
    num_types = len(cell_types)

    # Generate a color for each cell type
    colors = plt.cm.viridis(np.linspace(0, 1, num_types))

    # Create a mapping from cell type to color
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}

    # Initialize a variable to keep track of the current cell's index
    current_index = 0

    # Plot each cell's data with the corresponding color
    for cell_type, count in cell_counts.items():
        for i in range(current_index, current_index + count):
            ax.plot(times, x[i, :], c=type_to_color[cell_type], label=cell_type if i == current_index else "", **plot_kwargs)
        current_index += count

    # Add a legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5), title="Cell Types")

    ax.set_title(title)
    return ax

def plot_dynamical_sim_spike(times, x, title='', axis=None, cell_counts=None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Calculate the total number of cells and types for color mapping
    total_cells = x.shape[0]
    cell_types = list(cell_counts.keys())
    num_types = len(cell_types)

    # Generate a color for each cell type
    colors = plt.cm.viridis(np.linspace(0, 1, num_types))

    # Create a mapping from cell type to color
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}

    # Initialize a variable to keep track of the current cell's index
    current_index = 0

    # Plot each cell's data with the corresponding color
    for cell_type, count in cell_counts.items():
        for i in range(current_index, current_index + count):
            ax.scatter(times, x[i, :], c = type_to_color[cell_type], s = 0.1, label=cell_type if i == current_index else "", **plot_kwargs)
            #plt.setp(stem_container.markerline, 'color', type_to_color[cell_type])
        current_index += count

    # Add a legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5), title="Cell Types")

    ax.set_title(title)
    return ax

def plot_dynamical_sim_raster(times, x, title='', axis=None, cell_counts=None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    total_cells = x.shape[0]
    cell_types = list(cell_counts.keys())
    num_types = len(cell_types)

    colors = plt.cm.viridis(np.linspace(0, 1, num_types))
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}

    current_index = 0
    for cell_type, count in cell_counts.items():
        for i in range(current_index, current_index + count):
            # Find indices where the cell is active
            active_indices = np.where(x[i, :] > 0.5)[0]
            # Plot vertical lines for each active index
            ax.vlines(times[active_indices], i, i + 0.8, color=type_to_color[cell_type], **plot_kwargs)
        current_index += count

    ax.set_ylim(0, total_cells)
    ax.legend(handles=[plt.Line2D([0], [0], color=type_to_color[cell_type], lw=4) for cell_type in cell_types],
              labels=cell_types, loc='center left', bbox_to_anchor=(-0.2, 0.5), title="Cell Types")
    ax.set_title(title)

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