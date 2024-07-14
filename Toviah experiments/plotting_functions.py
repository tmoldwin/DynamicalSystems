import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['legend.fontsize'] = 6  # You can adjust the size as needed

def plot_weight_mat(A, cell_bounds=None, title='', axis=None):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    bound_positions = [bound[0] for bound in cell_bounds.values()]
    tick_positions = [0.5*(bound[0] + bound[1]) for bound in cell_bounds.values()]
    tick_labels = [cell_type for cell_type in cell_bounds.keys()]

    # Set the ticks and labels for both axes
    ax.set_xticks(tick_positions)
    for position in bound_positions:
        ax.axhline(position, color='black', linewidth=0.1)
        ax.axvline(position, color='black', linewidth=0.1)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.xaxis.tick_bottom()  # Place x ticks at the bottom
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Plot the matrix
    ax.imshow(A, cmap='coolwarm', norm=mcolors.TwoSlopeNorm(vcenter=0))
    ax.set_title(title)

    return ax


def plot_dynamical_sim(times, x, title='', axis=None, cell_bounds=None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Calculate the total number of cells and types for color mapping
    total_cells = x.shape[0]
    cell_types = list(cell_bounds.keys())
    num_types = len(cell_types)

    # Generate a color for each cell type
    colors = plt.cm.viridis(np.linspace(0, 1, num_types))

    # Create a mapping from cell type to color
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}

    # Initialize a variable to keep track of the current cell's index

    # Plot each cell's data with the corresponding color
    for cell_type, bound in cell_bounds.items():
        for i in range(bound[0], bound[1]):
            ax.plot(times, x[i, :], c=type_to_color[cell_type], label=cell_type if i == bound[0] else "",
                    **plot_kwargs)

    # Add a legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5))

    ax.set_title(title)
    return ax


def hist_mat(A, title='', axis=None):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.hist(A.flatten(), bins=100)
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
            ax.scatter(times, x[i, :], c=type_to_color[cell_type], s=0.1, label=cell_type if i == current_index else "",
                       **plot_kwargs)
            # plt.setp(stem_container.markerline, 'color', type_to_color[cell_type])
        current_index += count

    # Add a legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(-0.2, 0.5))

    ax.set_title(title)
    return ax


def plot_dynamical_sim_raster(times, x, title='', axis=None, cell_bounds=None, **plot_kwargs):
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    total_cells = x.shape[0]
    cell_types = list(cell_bounds.keys())
    num_types = len(cell_types)

    colors = plt.cm.viridis(np.linspace(0, 1, num_types))
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}

    current_index = 0
    for cell_type, bound in cell_bounds.items():
        for i in range(bound[0], bound[1]):
            # Find indices where the cell is active
            active_indices = np.where(x[i, :] > 0)[0]  # Assuming threshold for activity is > 0
            # Use scatterplot for each active index
            ax.scatter(times[active_indices], [i + 0.4] * len(active_indices), s = 0.1, color=type_to_color[cell_type],
                       **plot_kwargs)

    # Set x-axis to cover the full length of times
    ax.set_xlim(times[0], times[-1])

    ax.set_ylim(0, total_cells)
    ax.legend(handles=[plt.Line2D([0], [0], color=type_to_color[cell_type], lw=4) for cell_type in cell_types],
              labels=cell_types, loc='center left', bbox_to_anchor=(-0.3, 0.5))
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
    ax.axhline(0, color='black', linewidth=0.1, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.1, linestyle='--')


def show_inputs_by_type(times, input_by_type, indices_to_show, ax=None, legend=0):
    if ax is None:
        fig, ax = plt.subplots()
    cell_types = list(input_by_type.keys())
    num_types = len(cell_types)
    colors = plt.cm.viridis(np.linspace(0, 1, num_types))
    type_to_color = {cell_type: colors[i] for i, cell_type in enumerate(cell_types)}
    # Calculate the total number of cells and types for color mapping
    sm = 0
    for type in cell_types:
        type_input = np.atleast_1d(input_by_type[type])
        for ind in np.atleast_1d(indices_to_show):
            current = type_input[ind]
            ax.plot(times, current, c=type_to_color[type], label=type)
            sm += current
    ax.plot(times, sm, c='k', label="sum")
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.3, 0.5))
    return ax


def all_inputs_by_type(times, input_by_type, type_bounds, num_examples = 3, axes=None):
    cell_types = list(input_by_type.keys())
    num_types = len(cell_types)
    if axes is None:
        fig, axes = plt.subplots(num_types, num_examples, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, num_types))
    axes = np.atleast_2d(axes)
    for i, type in enumerate(cell_types):
        for j in range(num_examples):
            ax = axes[i][j]
            lower_bound = type_bounds[type][0]
            index = lower_bound + j
            show_inputs_by_type(times, input_by_type, index, ax=ax, legend= i == j == 0)
    return axes
