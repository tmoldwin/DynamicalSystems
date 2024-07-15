import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import expm
from matplotlib.ticker import FuncFormatter
import plotting_functions as pf
import helper_functions as hf


# def format_tick(value, tick_number):
#     if abs(value) >= 1000:
#         return f'{int(value):.1e}'
#     else:
#         return f'{round(value, 2)}'
#
#
# formatter = FuncFormatter(format_tick)
#

def sigmoid_activation(x, threshold):
    return 1 / (1 + np.exp(-x-threshold))


def tanh_activation(x, threshold):
    return np.tanh(x-threshold)


def threshold_activation(x, threshold):
    return (x > threshold).astype(int)


def linear_activation(x, threshold):
    return x


class Neural_Network:
    def __init__(self, cell_counts, submats, tau_leak=1, v_rest=-70, v_thresh=-40, activation=threshold_activation):
        self.cell_counts = cell_counts
        self.submats = submats
        self.tau_leak = tau_leak
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.activation = activation
        self.cell_bounds = hf.generate_cell_bounds(cell_counts)
        self.A = hf.gen_large_matrix(cell_bounds, submats)
        self.times = None
        self.V = None
        self.spikes = None
        self.input_by_type = None
        print(self.v_rest)

    def plot(self, **plot_kwargs):
        fig, axes = plt.subplots(4, 1, figsize=(6, 5))

        # Visualize the weight matrix
        pf.plot_weight_mat(self.A, title='experiment', cell_bounds=self.cell_bounds, axis=axes[0])

        # Assuming simulate is a method that updates self.times, self.V, self.spikes, and self.input_by_type

        # Visualize dynamics
        pf.plot_dynamical_sim(self.times, self.V, axis=axes[2], threshold = self.v_thresh, cell_bounds=self.cell_bounds)
        pf.plot_dynamical_sim_raster(self.times, self.spikes,  axis=axes[3], cell_bounds=self.cell_bounds)

        # Scatterplot eigenvalues
        pf.scatterplot_eigenvalues(self.A, ax=axes[1])
        plt.tight_layout()
        pf.all_inputs_by_type(self.times, self.input_by_type, self.cell_bounds)
        plt.tight_layout()
        plt.show()

    def plot_weights(self, **plot_kwargs):
        pf.plot_weight_mat(self.A, cell_bounds=self.cell_bounds, **plot_kwargs)

    def simulate(self, T, x0, dt):
        N = len(x0)
        times = np.arange(0, T - 1 + dt, dt)
        num_steps = len(times)
        V = np.zeros((N, num_steps))
        spikes = np.zeros_like(V)
        V[:, 0] = x0
        spikes[:, 0] = self.activation(V[:, 0], threshold=self.v_thresh)
        input_by_type = {cell_type: np.zeros((N, num_steps)) for cell_type, bound in self.cell_bounds.items()}
        for t in range(1, num_steps):
            for cell_type, bound in self.cell_bounds.items():
                input_by_type[cell_type][:, t] = np.dot(self.A[:, bound[0]:bound[1]], spikes[bound[0]:bound[1], t - 1])
            current = np.dot(self.A, spikes[:, t - 1])
            leak = (1.0 / self.tau_leak) * (V[:, t - 1] - self.v_rest)
            dV = dt * (-leak + current)
            V[:, t] = V[:, t - 1] + dV
            spike_inds = np.where(spikes[:, t-1])
            #V[spike_inds, t] = self.v_rest
            spikes[:, t] = self.activation(V[:, t], threshold=self.v_thresh)
        self.times = times
        self.V = V
        self.spikes = spikes
        self.input_by_type = input_by_type


if __name__ == '__main__':
    cell_counts_1 = {'L5': 10, 'L4': 20, 'Mini': 10}
    submats_1 = {('L5', 'L4'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
                 ('L4', 'L5'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
                 ('Mini', 'L4'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': -1}}

    cell_counts_layered = {'L1': 20, 'L2': 40, 'L2I': 20, 'L3': 20, }
    submats_layered = {
        ('L1', 'L2'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
        ('L2I', 'L2'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': -1},
        ('L2', 'L2I'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
        ('L2', 'L3'): {'dist': np.random.exponential, 'params': {'scale': 0.1}, 'sign': 1},
    }

    cell_counts_balanced = {'E': 100, 'I': 100}
    submats_balanced = {('E', 'I'): {'dist': np.random.exponential, 'params': {'scale': 1}, 'sign': 1},
                        ('I', 'E'): {'dist': np.random.exponential, 'params': {'scale': 1}, 'sign': -1},
                        ('E', 'E'): {'dist': np.random.exponential, 'params': {'scale': 1}, 'sign': 1},
                        ('I', 'I'): {'dist': np.random.exponential, 'params': {'scale': 1}, 'sign': -1}}

    cell_counts_Reem = {'L3': 360}
    submats_Reem = {('L3', 'L3'): {'dist': np.random.exponential, 'params': {'scale': 0.5}, 'sign': 1},
                    }

    cell_counts_Maor = {'P_E':20, 'P_I':20, 'R_E': 20, 'R_I':20}

    submats_Maor = {
        ('P_E', 'R_I'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': 1},
        ('P_E', 'R_E'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': 1},
        ('P_I', 'R_E'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': -1},
        ('P_I', 'R_I'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': -1},
        ('R_E', 'R_I'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': 1},
        ('R_I', 'R_E'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': -1},
        ('R_E', 'R_E'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': 1},
        ('R_I', 'R_I'): {'dist': np.random.exponential, 'params': {'scale': 5}, 'sign': -1},
    }
    cell_counts = cell_counts_balanced
    submats = submats_balanced
    cell_bounds = hf.generate_cell_bounds(cell_counts)
    v_rest = -70
    v_thresh = -65
    T = 25
    N = sum(cell_counts.values())
    active_ratio = 0.5
    x0 = np.random.randn(N) + v_thresh
    print(x0)
    dt = 0.01
    nn = Neural_Network(cell_counts, submats, v_rest=v_rest, tau_leak=1, v_thresh=v_thresh,
                        activation=threshold_activation)
    nn.simulate(T, x0, dt=dt)
    print(nn.spikes)
    nn.plot()
