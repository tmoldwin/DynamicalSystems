import numpy as np
from matplotlib import pyplot as plt
import scipy

def uniform_tuning_curve(num_stimuli, baseline=1, tuned_frac = 0, tuned_value = 2, noise = 0.01):
    x, tuning = np.arange(num_stimuli), baseline * np.ones(num_stimuli)
    tuning[:int(tuned_frac*num_stimuli)] = tuned_value
    tuning += noise * np.random.randn(num_stimuli)
    np.random.shuffle(tuning)
    return tuning

def gaussian_tuning_curve(num_stimuli, sigma, center, baseline=1, peak=2, noise = 0.01):
    x = np.linspace(0, 1, num_stimuli)
    tuning = baseline + (peak - baseline) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    #np.random.shuffle(tuning)
    tuning += noise * np.random.randn(num_stimuli)
    return tuning

def plot_tuning_curve(ax, x, tuning_curve, title='', xlabel='', ylabel=''):
    ax.stem(x, tuning_curve)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def Z_score(tuning_curve):
    return (tuning_curve - tuning_curve.mean()) / tuning_curve.std()

def divide_by_mean(tuning_curve):
    return tuning_curve/tuning_curve.mean()

def plot_z_distribution(ax, tuning_curve, title='', xlabel='', ylabel=''):
    z_scores = Z_score(tuning_curve)
    print(tuning_curve)
    print(z_scores)
    try:
        ax.hist(z_scores, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    except:
        pass
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def normalize(x, norm):
    if norm == 'min_max':
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif norm == 'Z_score':
        return (x - np.mean(x)) / np.std(x)
    else:  # No normalization
        return x

def range_over_mean(tuning_curve, norm='None'):
    tuning_curve = normalize(tuning_curve, norm)
    return (tuning_curve.max() - tuning_curve.min()) / tuning_curve.mean()

def CV_score(tuning_curve, norm = None):
    tuning_curve = normalize(tuning_curve, norm)
    return np.std(tuning_curve)/np.mean(tuning_curve)
def rolls1995(tuning_curve, norm='None'):
    tuning_curve = normalize(tuning_curve, norm)
    return np.mean(tuning_curve)**2 / np.mean(tuning_curve**2)

def Vinje2000(tuning_curve, norm='None'):
    tuning_curve = normalize(tuning_curve, norm)
    A = rolls1995(tuning_curve)
    return (1 - A)/(1-1/len(tuning_curve))

def gini(tuning_curve, norm='None'):
    tuning_curve = normalize(tuning_curve, norm)
    n = len(tuning_curve)
    return np.sum(np.abs(np.subtract.outer(tuning_curve, tuning_curve))) / (2 * n**2 * np.mean(tuning_curve))



num_stimuli = 100
fracs_uniform = [i * 0.01 for i in range(100)]
sigmas_bell = [i * 0.005 for i in range(100)]
tunings_uniform = [uniform_tuning_curve(100, baseline = 0.1, tuned_frac=frac, tuned_value=1) for frac in fracs_uniform]
tunings_bell = [gaussian_tuning_curve(num_stimuli=100, sigma = sigma, baseline=0.1, peak = 1, center = 0.5) for sigma in sigmas_bell]
conditions_dict = {'Uniform': tunings_uniform, 'Bell': tunings_bell}
example_nums = [0,1,25,75,99]

# Define a new subplot layout with additional rows for histograms
fig, axs = plt.subplot_mosaic([['A1', 'A2', 'A3', 'A4', 'A5', 'C1'],
                               ['H1', 'H2', 'H3', 'H4', 'H5', 'C2'],
                               ['B1', 'B2', 'B3', 'B4', 'B5', 'C3'],
                               ['I1', 'I2', 'I3', 'I4', 'I5', 'C4']], figsize=(12, 10))

# fig, axs = plt.subplot_mosaic([['A1', 'A2', 'A3', 'A4', 'A5', 'C1'],
#
#                                ['B1', 'B2', 'B3', 'B4', 'B5', 'C2'],
#                                ], figsize=(12, 5))

for it, example_num in enumerate(example_nums):
    frac_uniform = fracs_uniform[example_num]
    axs['A' + str(it + 1)].stem(tunings_uniform[example_num], linefmt = 'blue')
    axs['A' + str(it + 1)].set_title('Density = ' + str(frac_uniform))
    axs['A' + str(it + 1)].set_ylim([0, 1.5])
    axs['A' + str(it + 1)].set_xlabel('Stim #')

    # Plot histogram of firing rates for uniform tuning curve
    axs['H' + str(it + 1)].hist(tunings_uniform[example_num], bins='auto', color='blue')
    axs['H' + str(it + 1)].set_title('Density = ' + str(frac_uniform))
    axs['H' + str(it + 1)].set_xlabel('Firing rate')
    mean_val = np.mean(tunings_uniform[example_num])
    std_val = np.std(tunings_uniform[example_num])
    axs['H' + str(it + 1)].axvline(mean_val, color='green', linestyle='--')
    axs['H' + str(it + 1)].axvspan(mean_val - std_val, mean_val + std_val, alpha=0.3, color='gray')
    axs['H' + str(it + 1)].set_xlim([0,1.1])



    sigma_bell = sigmas_bell[example_num]
    axs['B' + str(it + 1)].stem(tunings_bell[example_num], linefmt='r')
    axs['B' + str(it + 1)].set_title('Sigma = ' + str(sigma_bell))
    axs['B' + str(it + 1)].set_ylim([0, 1.5])
    axs['B' + str(it + 1)].set_xlabel('Stim #')

    # Plot histogram of firing rates for bell tuning curve
    axs['I' + str(it + 1)].hist(tunings_bell[example_num], bins='auto', color='red')
    axs['I' + str(it + 1)].set_title('Sigma = ' + str(sigma_bell))
    axs['I' + str(it + 1)].set_xlabel('Firing rate')
    mean_val = np.mean(tunings_bell[example_num])
    axs['I' + str(it + 1)].axvline(mean_val, color='green', linestyle='--')
    axs['I' + str(it + 1)].axvspan(mean_val - std_val, mean_val + std_val, alpha=0.3, color='gray')
    axs['I' + str(it + 1)].set_xlim([0,1.1])


axs['A1'].set_ylabel('Firing rate')
axs['B1'].set_ylabel('Firing rate')
# axs['I1'].set_ylabel('Count')
# axs['H1'].set_ylabel('Count')


def calculate_and_plot(params, score_funcs_dict, condition_values, subplot):
    for i, (score_name, score_data) in enumerate(score_funcs_dict.items()):
        norm = score_data['norm']
        scores = [score_data['func'](x, norm) for x in condition_values]
        color = score_data.get('color', 'b')  # Get color from dictionary or default to 'b'
        linestyle = score_data.get('linestyle', '-')  # Get linestyle from dictionary or default to '-'
        subplot.plot(params, scores, label=f'{score_name}', color=color, linestyle=linestyle)
    subplot.set_title('Scores')

score_funcs_dict = {
    'Vinje2000': {'func': Vinje2000, 'norm': 'None', 'color': 'b', 'linestyle': '-'},
    'Vinje2000_minmax': {'func': Vinje2000, 'norm': 'min_max', 'color': 'b', 'linestyle': ':'},
    'CV': {'func': CV_score, 'norm': 'None', 'color': 'g', 'linestyle': '-'},
    'CV_minmax': {'func': CV_score, 'norm': 'min_max', 'color': 'g', 'linestyle': ':'}
}

# Plot the scores on the new subplots
calculate_and_plot(fracs_uniform, score_funcs_dict, tunings_uniform, axs['C2'])
calculate_and_plot(sigmas_bell, score_funcs_dict, tunings_bell, axs['C4'])


axs['C1'].set_title('Uniform')
axs['C2'].set_title('Bell')
axs['C1'].set_xlabel('Fraction preferred stimuli')
axs['C2'].set_xlabel('Sigma')
axs['C2'].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.subplots_adjust(right=0.7)

plt.tight_layout()
plt.show()
