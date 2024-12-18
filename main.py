"""Author : BRANDOIT Julien

This is the main file. This is the one that is really related to the paper computation. Other .py files should be 
considered as BLACK-BOXES.

Note : 
The main code was entierly written by me. All the real 'computations' and algorithms are implemented from scratch.
Nevertheless, ChatGPT and Github Copilot were used to help me with the writing of some part of the code.
I really insist on the fact that I wrote the code myself, and that I only used ChatGPT and Github Copilot to help me with
writing PLOTTING functions. I did not use them to write the main code, the algorithms, the computations, etc.
They were only used as quick ways of dealing with matplotlib and seaborn functions.
"""

import matplotlib.patches as mpatches
import concurrent.futures
from scipy.stats import gaussian_kde
from stg import generate_neuromodulated_population
import numpy as np
from scipy.stats import gamma
from utils import gamma_uniform_mean_std_matching
import matplotlib.pyplot as plt
import seaborn as sns
from stg import DICs, get_default_parameters
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
from stg import simulate_individual_t_eval, get_default_u0
import warnings
from utils import simulate_population_t_eval_multiprocessing
from tqdm import tqdm
from matplotlib.lines import Line2D

# == plotting setup ==
sns.set_context("notebook")
sns.set_style("white")
sns.set_palette("flare")
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "0.9"
plt.rcParams["grid.linestyle"] = "solid"
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# == Initial parameters ==

# maximal conductances distributions
g_bar_range_Na = [0, 6000.0]
g_bar_range_Kd = [0, 265]
g_bar_range_CaT = [0, 9.0]
g_bar_range_CaS = [0, 38]
g_bar_range_KCa = [0, 200]
g_bar_range_A = [0, 450.0]
g_bar_range_H = [0, 0.53]
g_bar_range_leak = [0.005, 0.015]

# fit the gamma distribution to the uniform distribution
g_Na_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_Na)
g_Kd_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_Kd)
g_CaT_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_CaT)
g_CaS_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_CaS)
g_KCa_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_KCa)
g_A_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_A)
g_H_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_H)
g_leak_gamma_params = gamma_uniform_mean_std_matching(*g_bar_range_leak)

# == Routines and wrappers ==

# Wrapper to simulate the population using multiprocessing
def simulate_population(population, U0=get_default_u0(), T_EVAL=np.arange(
        3000, 4000, 0.5), PARAMS=get_default_parameters(), MAX_WORKERS=16):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.asarray(simulate_population_t_eval_multiprocessing(
            simulate_individual_t_eval, population, U0, T_EVAL, PARAMS, MAX_WORKERS))[:, 1, :]

# Wrapper to generate a virtual population
def get_a_virtual_population(n_cells, SEED=np.random.randint(0, 100000)):
    np.random.seed(SEED)
    g_leak = gamma.rvs(
        a=g_leak_gamma_params[0],
        scale=g_leak_gamma_params[1],
        size=n_cells)
    g_Na = gamma.rvs(
        a=g_Na_gamma_params[0],
        scale=g_Na_gamma_params[1],
        size=n_cells)
    g_Kd = gamma.rvs(
        a=g_Kd_gamma_params[0],
        scale=g_Kd_gamma_params[1],
        size=n_cells)
    g_CaT = gamma.rvs(
        a=g_CaT_gamma_params[0],
        scale=g_CaT_gamma_params[1],
        size=n_cells)
    g_CaS = gamma.rvs(
        a=g_CaS_gamma_params[0],
        scale=g_CaS_gamma_params[1],
        size=n_cells)
    g_KCa = gamma.rvs(
        a=g_KCa_gamma_params[0],
        scale=g_KCa_gamma_params[1],
        size=n_cells)
    g_A = gamma.rvs(
        a=g_A_gamma_params[0],
        scale=g_A_gamma_params[1],
        size=n_cells)
    g_H = gamma.rvs(
        a=g_H_gamma_params[0],
        scale=g_H_gamma_params[1],
        size=n_cells)

    return np.array([g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak]).T

# Wrapper to simulate the model
def model(population):
    dics = DICs(-51.,
                population[:,
                           0],
                population[:,
                           1],
                population[:,
                           2],
                population[:,
                           3],
                population[:,
                           4],
                population[:,
                           5],
                population[:,
                           6],
                population[:,
                           7],
                **get_default_parameters())
    dics = np.array(dics).squeeze()[:3]
    return dics

# Wrapper to compute the sobol indices (first order and total)
def sobol_indices(n_cells, n_params=8, population=None):
    # Generate A and B populations
    if population is None:
        A = get_a_virtual_population(n_cells)
        B = get_a_virtual_population(n_cells)
    else:
        population_suffled = population.copy()

        A = population_suffled[:len(population_suffled) // 2]
        B = population_suffled[len(population_suffled) // 2:]

    A_i = [A.copy() for _ in range(n_params)]
    for i in range(n_params):
        A_i[i][:, i] = B[:, i].copy()

    Y_A = model(A)
    Y_B = model(B)
    Y_A_i = [model(A_i[i]) for i in range(n_params)]

    n_outputs = Y_A.shape[0]

    """
    S_indices contains the sobol indices;
    S[:, :, 0] contains the first order indices
    S[:, :, 1] contains the total indices
    """
    S_indices = np.zeros((n_outputs, n_params, 2))

    Var_Y = np.var(Y_A, axis=1)
    E_Y = np.mean(Y_A, axis=1)

    for o in range(n_outputs):
        for i in range(n_params):
            V_i = np.mean((Y_B[o] * (Y_A_i[i][o] - Y_A[o])))
            S_indices[o, i, 0] = V_i / Var_Y[o]
            V_i_T = 1 / 2 * np.mean(((Y_A[o] - Y_A_i[i][o])**2))
            S_indices[o, i, 1] = V_i_T / Var_Y[o]

    # we clip the values to avoid numerical errors, min is 0 and no max
    S_indices = np.clip(S_indices, 0, None)

    return S_indices

# Wrapper to build the marginals of the DICs based on the KDE using
# Scott's rule
def build_marginal(dics):
    return gaussian_kde(dics[0, :]), gaussian_kde(
        dics[1, :]), gaussian_kde(dics[2, :])

# Wrapper to compute all the estimators for the convergence analysis
# (mean, std, 95% mass interval, sobol indices)
def process_population(j, n_cells):
    S_indices = np.zeros((len(n_cells), 3, 8, 2))
    sum_S_indices = np.zeros((len(n_cells), 3, 2))
    mean_dics = np.zeros((len(n_cells), 3))
    std_dics = np.zeros((len(n_cells), 3))
    lower_bound_95_mass = np.zeros((len(n_cells), 3))
    upper_bound_95_mass = np.zeros((len(n_cells), 3))

    final_population = get_a_virtual_population(
        n_cells[-1], SEED=np.random.randint(0, 100000) + j)
    final_dics = model(final_population)

    for i, n in tqdm(enumerate(n_cells), total=len(n_cells),
                     desc=f"Convergence analysis for population {j + 1}"):
        population = final_population[:n]
        dics = final_dics[:, :n]
        S_indices[i] = sobol_indices(n, population=population)
        sum_S_indices[i, :, 0] = S_indices[i, :, :, 0].sum(axis=1)
        sum_S_indices[i, :, 1] = S_indices[i, :, :, 1].sum(axis=1)
        mean_dics[i] = np.mean(dics, axis=1)
        std_dics[i] = np.std(dics, axis=1)
        lower_bound_95_mass[i] = np.percentile(dics, 2.5, axis=1)
        upper_bound_95_mass[i] = np.percentile(dics, 97.5, axis=1)

    bounds_95_mass = np.stack(
        [lower_bound_95_mass, upper_bound_95_mass], axis=-1)

    return S_indices, sum_S_indices, mean_dics, std_dics, bounds_95_mass

# Wrapper to compute the convergence analysis
def do_convergence_analysis():
    number_of_population = 16
    N_min = 100
    N_max = 300000
    N_point = 150

    # n_cells is exponentially spaced between N_min and N_max
    n_cells = np.unique(
        np.logspace(
            np.log10(N_min),
            np.log10(N_max),
            N_point).astype(int))

    # force n_cells to be even (because A and B populations are generated by
    # splitting the population in two)
    for i, n in enumerate(n_cells):
        if n % 2:
            n_cells[i] = n + 1

    S_indices_population = []
    sum_S_indices_population = []
    mean_dics_population = []
    std_dics_population = []
    bounds_95_mass_population = []
    print(
        f"Processing convergence analysis for the {number_of_population} populations")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                process_population,
                range(
                    number_of_population),
                [n_cells] * number_of_population))

    for result in results:
        S_indices, sum_S_indices, mean_dics, std_dics, bounds_95_mass = result
        S_indices_population.append(S_indices)
        sum_S_indices_population.append(sum_S_indices)
        mean_dics_population.append(mean_dics)
        std_dics_population.append(std_dics)
        bounds_95_mass_population.append(bounds_95_mass)

    # We compute the mean of each estimators over the populations
    # We compute the std of each estimators over the populations
    # We build plot of running mean +- std for each estimators
    # A first (1x3) plot about mean_dics_population, std_dics_population, bounds_95_mass_population
    # A second (1x2) plot about sum_S_indices_population[:, :, 0],
    # sum_S_indices_population[:, :, 1]

    S_indices_population = np.array(S_indices_population)
    sum_S_indices_population = np.array(sum_S_indices_population)
    mean_dics_population = np.array(mean_dics_population)
    std_dics_population = np.array(std_dics_population)
    bounds_95_mass_population = np.array(bounds_95_mass_population)

    return n_cells, S_indices_population, sum_S_indices_population, mean_dics_population, std_dics_population, bounds_95_mass_population

# Wrapper to plot on an axis the mean +- std of the data
def plot_with_error(ax, x, y_mean, y_std, colors,
                    labels, xlabel, title, y_sci=False):
    for i, color in enumerate(colors):
        ax.plot(x, y_mean[:, i], color=color)
        ax.fill_between(x, y_mean[:, i] -
                        y_std[:, i], y_mean[:, i] +
                        y_std[:, i], alpha=0.5, color=color)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.add_patch(FancyArrowPatch((0, 0), (1, 0), mutation_scale=20,
                 color='black', lw=2, arrowstyle='->', transform=ax.transAxes))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1), mutation_scale=20,
                 color='black', lw=2, arrowstyle='->', transform=ax.transAxes))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if y_sci:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)
    ax.xaxis.offsetText.set_fontsize(18)

# == Code divided in functions ==
def generate_traces_graph():  # Figure 1
    PALETTE = "icefire"
    N = 2
    V = -51.
    g_u = 5.

    # == Spiking neurons ==
    g_s = 4.
    spiking_population = generate_neuromodulated_population(N, V, g_s, g_u)

    # == Bursting neurons ==
    g_s = -5.
    bursting_population = generate_neuromodulated_population(N, V, g_s, g_u)

    # == Simulate the two first neurons of each population ==

    population_to_simulate = np.vstack(
        [spiking_population[:2], bursting_population[:2]])
    T_EVAL = np.arange(3000, 4000, 0.1)
    simulated_population = simulate_population(
        population_to_simulate, T_EVAL=T_EVAL)

    # == Plot the results ==
    # First plot is a 2x2 grid with the first row being the spiking neurons
    # and the second row being the bursting neurons

    colors = sns.color_palette(PALETTE, 2)
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < 2:
            color = colors[0]  # Spiking neurons
            g_bar = spiking_population[i % 2]
        else:
            color = colors[1]  # Bursting neurons
            g_bar = bursting_population[i % 2]
        ax.plot(T_EVAL, simulated_population[i], color=color)
        title = "Spiking" if i < 2 else "Bursting"
        title += f" neuron {i % 2 + 1}\n"
        title += r"$\bar{\mathbf{g}}$ = [" + ", ".join(
            [
                f"{g_bar[0]:.0f}",
                f"{g_bar[1]:.0f}",
                f"{g_bar[2]:.2f}",
                f"{g_bar[3]:.2f}",
                f"{g_bar[4]:.1f}",
                f"{g_bar[5]:.0f}",
                f"{g_bar[6]:.3f}",
                f"{g_bar[7]:.3f}"]) + "]"
        ax.set_title(title)

        if i == 0 or i == 2:
            ax.set_ylabel("V (mV)")

        if i > 1:
            ax.set_xlabel("Time (ms)")

        ax.set_yticks([])

    text = r"$\bar{\mathbf{g}}$ = [" + r"$\bar{g}_{\text{Na}}$, $\bar{g}_{\text{Kd}}$, $\bar{g}_{\text{CaT}}$, $\bar{g}_{\text{CaS}}$, $\bar{g}_{\text{KCa}}$, $\bar{g}_{\text{A}}$, $\bar{g}_{\text{H}}$, $g_{\text{leak}}$]"
    plt.figtext(0.5, 0., text, ha="center", fontsize=18)
    plt.tight_layout()

    # save to pdf as "stg_behavior.pdf", need to expand the bottom margin to
    # fit the text
    plt.savefig("stg_behavior.pdf", bbox_inches='tight')
    plt.show()  # Figure 1


def print_table_3():  # Table 3
    print("Maximal conductances, Uniform range, (k, theta)")
    print("Na:  ", g_bar_range_Na, g_Na_gamma_params)
    print("Kd:  ", g_bar_range_Kd, g_Kd_gamma_params)
    print("CaT: ", g_bar_range_CaT, g_CaT_gamma_params)
    print("CaS: ", g_bar_range_CaS, g_CaS_gamma_params)
    print("KCa: ", g_bar_range_KCa, g_KCa_gamma_params)
    print("A:   ", g_bar_range_A, g_A_gamma_params)
    print("H:   ", g_bar_range_H, g_H_gamma_params)
    print("leak:", g_bar_range_leak, g_leak_gamma_params)


def plot_g_distributions():  # Figure 3
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i, (ax, g, label, g_param) in enumerate(zip(axs, population.T, [
        r"$\bar{g}_{\text{Na}}$",
        r"$\bar{g}_{\text{Kd}}$",
        r"$\bar{g}_{\text{CaT}}$",
        r"$\bar{g}_{\text{CaS}}$",
        r"$\bar{g}_{\text{KCa}}$",
        r"$\bar{g}_{\text{A}}$",
        r"$\bar{g}_{\text{H}}$",
        r"${g}_{\text{leak}}$"
    ], [
        g_Na_gamma_params,
        g_Kd_gamma_params,
        g_CaT_gamma_params,
        g_CaS_gamma_params,
        g_KCa_gamma_params,
        g_A_gamma_params,
        g_H_gamma_params,
        g_leak_gamma_params
    ])):

        x = np.linspace(0, 1.1 * np.max(g), 1000)
        y = gamma.pdf(x, a=g_param[0], scale=g_param[1])
        ax.plot(x, y, lw=2)
        ax.fill_between(x, y, alpha=0.5)

        if i == 0 or i == 4:
            ax.set_ylabel('Density', fontsize=20)
        else:
            ax.set_ylabel('')

        ax.set_xlabel('')  # Remove the default x-label

        # Add the label above the x-axis to the right
        ax.text(
            0.95,
            0.1,
            label,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=20)

        ax.set_xlim(0, 1.1 * np.max(g))
        ax.set_ylim(0, 1.1 * np.max(y))

        # Force scientific notation for y-axis and x-axis
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_major_formatter(ScalarFormatter())

        # Enforce scientific notation for both axes
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-100, 3.5))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # Set font size for tick labels including exponents
        # Increase y-axis ticks fontsize
        ax.tick_params(axis='y', labelsize=18)
        # Increase x-axis ticks fontsize
        ax.tick_params(axis='x', labelsize=18)
        ax.yaxis.offsetText.set_fontsize(18)
        ax.xaxis.offsetText.set_fontsize(18)

        ax.add_patch(
            FancyArrowPatch(
                (0,
                 0),
                (1,
                 0),
                mutation_scale=20,
                color='black',
                lw=2,
                arrowstyle='->',
                transform=ax.transAxes))
        ax.add_patch(
            FancyArrowPatch(
                (0,
                 0),
                (0,
                 1),
                mutation_scale=20,
                color='black',
                lw=2,
                arrowstyle='->',
                transform=ax.transAxes))

    plt.tight_layout()

    # Save as g_distribution.pdf
    plt.savefig("g_distributions.pdf")
    plt.show()


def plot_dics_distribution(dics):  # Figure 4
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    sns.histplot(dics[0,
                      :],
                 bins='auto',
                 kde=True,
                 ax=ax,
                 stat="density",
                 color="blue",
                 alpha=0.35,
                 linewidth=2,
                 label=r"$p(g_f)$",
                 edgecolor='none')
    sns.histplot(dics[1,
                      :],
                 bins='auto',
                 kde=True,
                 ax=ax,
                 stat="density",
                 color="green",
                 alpha=0.35,
                 linewidth=2,
                 label=r"$p(g_s)$",
                 edgecolor='none')
    sns.histplot(dics[2,
                      :],
                 bins='auto',
                 kde=True,
                 ax=ax,
                 stat="density",
                 color="red",
                 alpha=0.35,
                 linewidth=2,
                 label=r"$p(g_u)$",
                 edgecolor='none')

    ax.set_xlabel('', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    ax.legend(
        title=r"$p_{\text{KDE}}(\mathbf{g}_{\text{DICs}})$",
        loc="upper right",
        fontsize=15,
        title_fontsize=18)

    ax.set_xlim(-50, 50)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax.text(
        0.95,
        0.08,
        r"$g_f, g_s, g_u$",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=20)

    ax.add_patch(FancyArrowPatch((0, 0), (1, 0), mutation_scale=20,
                 color='black', lw=2, arrowstyle='->', transform=ax.transAxes))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1), mutation_scale=20,
                 color='black', lw=2, arrowstyle='->', transform=ax.transAxes))

    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)
    ax.xaxis.offsetText.set_fontsize(18)

    plt.tight_layout()
    plt.savefig("DICs_distribution.pdf")
    plt.show()


def print_table_1():  # Table 1
    print("DICs marginals")
    print(
        "f: ",
        f"{np.mean(dics[0, :]):.2f}",
        f"{np.std(dics[0, :]):.2f}",
        f"{np.percentile(dics[0, :], 2.5):.2f}",
        f"{np.percentile(dics[0, :], 97.5):.2f}")
    print(
        "s: ",
        f"{np.mean(dics[1, :]):.2f}",
        f"{np.std(dics[1, :]):.2f}",
        f"{np.percentile(dics[1, :], 2.5):.2f}",
        f"{np.percentile(dics[1, :], 97.5):.2f}")
    print(
        "u: ",
        f"{np.mean(dics[2, :]):.2f}",
        f"{np.std(dics[2, :]):.2f}",
        f"{np.percentile(dics[2, :], 2.5):.2f}",
        f"{np.percentile(dics[2, :], 97.5):.2f}")


# Not used in the paper; The 'plot_dics_distribution' function is used instead.
def plot_marginals(f_kde, s_kde, u_kde):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs = axs.flatten()
    colors = ["blue", "green", "red"]
    labels = [r"$g_f$", r"$g_s$", r"$g_u$"]
    kdes = [f_kde, s_kde, u_kde]
    for ax, color, label, kde in zip(axs, colors, labels, kdes):
        x = np.linspace(-50, 50, 1000)
        y = kde(x)
        ax.plot(x, y, lw=2, color=color)
        ax.fill_between(x, y, alpha=0.5, color=color)
        ax.set_xlabel('DICs', fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        ax.set_title(label, fontsize=20)
        ax.set_xlim(-50, 50)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.yaxis.offsetText.set_fontsize(18)
        ax.xaxis.offsetText.set_fontsize(18)
        ax.add_patch(FancyArrowPatch((0, 0), (1, 0), mutation_scale=20,
                     color='black', lw=2, arrowstyle='->', transform=ax.transAxes))
        ax.add_patch(FancyArrowPatch((0, 0), (0, 1), mutation_scale=20,
                     color='black', lw=2, arrowstyle='->', transform=ax.transAxes))

    plt.tight_layout()
    plt.savefig("DICs_marginals.pdf")
    plt.show()


def plot_table_2(S_indices):  # Table 2
    sum_first_order = S_indices[:, :, 0].sum(axis=1)
    sum_total = S_indices[:, :, 1].sum(axis=1)
    # make a plot of the sobol indices : two heat maps, first order and total
    # indices
    fig, ax = plt.subplots(
        1, 4, figsize=(
            15, 3), gridspec_kw={
                'width_ratios': [
                    0.45, 0.05, 0.45, 0.05]}, sharey=True)
    y_ticks = [r"$g_f$", r"$g_s$", r"$g_u$"]
    x_ticks = [
        r"$\bar{g}_{\text{Na}}$",
        r"$\bar{g}_{\text{Kd}}$",
        r"$\bar{g}_{\text{CaT}}$",
        r"$\bar{g}_{\text{CaS}}$",
        r"$\bar{g}_{\text{KCa}}$",
        r"$\bar{g}_{\text{A}}$",
        r"$\bar{g}_{\text{H}}$",
        r"${g}_{\text{leak}}$"]
    ax = ax.flatten()

    # First order indices heatmap
    sns.heatmap(S_indices[:,
                          :,
                          0],
                annot=True,
                fmt=".2f",
                ax=ax[0],
                cmap="magma",
                cbar=False,
                vmin=0,
                vmax=1,
                annot_kws={"size": 14})
    ax[0].set_title(r'First order indices $S^i_m$', fontsize=16)
    ax[0].set_xlabel('', fontsize=16)
    ax[0].set_ylabel('', fontsize=16)
    ax[0].set_xticklabels(x_ticks, fontsize=16)
    ax[0].set_yticklabels(y_ticks, fontsize=16)

    # Sum of first order indices as column heat map
    sns.heatmap(sum_first_order[:,
                                None],
                annot=True,
                fmt=".2f",
                ax=ax[1],
                cmap="magma",
                cbar=False,
                vmin=0,
                vmax=1,
                annot_kws={"size": 14})
    ax[1].set_xticklabels([r'$\sum_m S^i_m$'], fontsize=16)
    ax[1].set_ylabel('', fontsize=12)

    # Total indices heatmap
    sns.heatmap(S_indices[:,
                          :,
                          1],
                annot=True,
                fmt=".2f",
                ax=ax[2],
                cmap="magma",
                cbar=False,
                vmin=0,
                vmax=1,
                annot_kws={"size": 14})
    ax[2].set_title(r'Total indices $S^i_{T_m}$', fontsize=16)
    ax[2].set_xlabel('', fontsize=12)
    ax[2].set_ylabel('', fontsize=12)
    ax[2].set_xticklabels(x_ticks, fontsize=16)
    ax[2].set_yticklabels(y_ticks, fontsize=16)

    # Sum of total indices as column heat map
    sns.heatmap(sum_total[:,
                          None],
                annot=True,
                fmt=".2f",
                ax=ax[3],
                cmap="magma",
                cbar=False,
                vmin=0,
                vmax=1,
                annot_kws={"size": 14})
    ax[3].set_xticklabels([r'$\sum_m S^i_{T_m}$'], fontsize=16)
    ax[3].set_ylabel('', fontsize=18)
    ax[3].set_yticklabels(y_ticks, fontsize=18)

    plt.tight_layout()
    # save as sobol_indices.pdf
    plt.savefig("sobol_indices.pdf")
    plt.show()  # Table 2


def plot_figure_5():  # Figure 5
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs = axs.flatten()

    # Plot for Mean of DICs
    plot_with_error(axs[0], n_cells, mean_mean_dics_population, std_mean_dics_population, ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{g_{\text{DICs}, i}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{g_{\text{DICs}, i}}\right](N)$")

    # Plot for Standard Deviation of DICs
    plot_with_error(axs[1], n_cells, mean_std_dics_population, std_std_dics_population, ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\sigma_{g_{\text{DICs}, i}}(N) \pm \sigma_{\text{pop.}}\left[\sigma_{g_{\text{DICs}, i}}\right](N)$")

    # Plot for 95% Mass Interval of DICs
    for i, color in enumerate(["blue", "green", "red"]):
        axs[2].plot(
            n_cells, mean_bounds_95_mass_population[:, i, 0], color=color)
        axs[2].fill_between(n_cells, mean_bounds_95_mass_population[:, i, 0] - std_bounds_95_mass_population[:, i, 0],
                            mean_bounds_95_mass_population[:, i, 0] + std_bounds_95_mass_population[:, i, 0], alpha=0.5, color=color)
        axs[2].plot(n_cells, mean_bounds_95_mass_population[:, i, 1],
                    color=color, linestyle="--")
        axs[2].fill_between(n_cells, mean_bounds_95_mass_population[:, i, 1] - std_bounds_95_mass_population[:, i, 1],
                            mean_bounds_95_mass_population[:, i, 1] + std_bounds_95_mass_population[:, i, 1], alpha=0.5, color=color)
    axs[2].set_xlabel(r"Population size $N$", fontsize=18)
    axs[2].set_title(
        r"$L^{95\%}_{\text{DICs}, i}(N)$ and $U^{95\%}_{\text{DICs}, i}(N) \pm \sigma_{\text{pop.}}$",
        fontsize=18)
    axs[2].add_patch(
        FancyArrowPatch(
            (0,
             0),
            (1,
             0),
            mutation_scale=20,
            color='black',
            lw=2,
            arrowstyle='->',
            transform=axs[2].transAxes))
    axs[2].add_patch(
        FancyArrowPatch(
            (0,
             0),
            (0,
             1),
            mutation_scale=20,
            color='black',
            lw=2,
            arrowstyle='->',
            transform=axs[2].transAxes))
    axs[2].xaxis.set_major_formatter(ScalarFormatter())
    axs[2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    axs[2].tick_params(axis='y', labelsize=18)
    axs[2].tick_params(axis='x', labelsize=18)
    axs[2].yaxis.offsetText.set_fontsize(18)
    axs[2].xaxis.offsetText.set_fontsize(18)

    # Adjust layout to make space for the shared legend
    plt.subplots_adjust(bottom=0.25)  # Add space at the bottom

    # Shared Legend with Filled Rectangles
    patches = [mpatches.Patch(color="blue", label=r"$g_f$"),
               mpatches.Patch(color="green", label=r"$g_s$"),
               mpatches.Patch(color="red", label=r"$g_u$")]
    fig.legend(
        handles=patches,
        loc="lower center",
        fontsize=20,
        ncol=3,
        frameon=False)

    plt.savefig("convergence_analysis_table1.pdf", bbox_inches="tight")
    plt.show()


def plot_figure_6():  # Figure 6

    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    axs = axs.flatten()

    # Plot for Sum of First Order Indices
    plot_with_error(axs[0], n_cells, mean_sum_S_indices_population[:, :, 0], std_sum_S_indices_population[:, :, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{\Sigma_m S^i_m}(N) \pm \sigma_{\text{pop.}}\left[\mu_{\Sigma_m S^i_m}\right](N)$")

    # Plot for Sum of Total Indices
    plot_with_error(axs[1], n_cells, mean_sum_S_indices_population[:, :, 1], std_sum_S_indices_population[:, :, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{\Sigma_m S^i_{T_m}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{\Sigma_m S^i_{T_m}}\right](N)$")

    # Adjust layout to make space for the shared legend
    plt.subplots_adjust(bottom=0.25)  # Add space at the bottom

    # Shared Legend with Filled Rectangles
    patches = [mpatches.Patch(color="blue", label=r"$g_f$"),
               mpatches.Patch(color="green", label=r"$g_s$"),
               mpatches.Patch(color="red", label=r"$g_u$")]
    fig.legend(
        handles=patches,
        loc="lower center",
        fontsize=20,
        ncol=3,
        frameon=False)

    plt.savefig("convergence_analysis_table2.pdf", bbox_inches="tight")
    plt.show()


def plot_figure_7():  # Figure 7

    # We make a (8x2) plot of the convergence of all the SOBOL indices using the plot_with_error function
    # First col plot is the first order indices
    # Second col plot is the total indices
    # First file (4x2) plot
    fig, axs = plt.subplots(4, 2, figsize=(20, 20), sharex=True)
    axs = axs.flatten()
    plot_with_error(axs[0], n_cells, mean_S_indices_population[:, :, 0, 0], std_S_indices_population[:, :, 0, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{Na}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{Na}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[1], n_cells, mean_S_indices_population[:, :, 0, 1], std_S_indices_population[:, :, 0, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{Na}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{Na}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[2], n_cells, mean_S_indices_population[:, :, 1, 0], std_S_indices_population[:, :, 1, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{Kd}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{Kd}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[3], n_cells, mean_S_indices_population[:, :, 1, 1], std_S_indices_population[:, :, 1, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{Kd}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{Kd}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[4], n_cells, mean_S_indices_population[:, :, 2, 0], std_S_indices_population[:, :, 2, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{CaT}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{CaT}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[5], n_cells, mean_S_indices_population[:, :, 2, 1], std_S_indices_population[:, :, 2, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{CaT}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{CaT}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[6], n_cells, mean_S_indices_population[:, :, 3, 0], std_S_indices_population[:, :, 3, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{CaS}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{CaS}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[7], n_cells, mean_S_indices_population[:, :, 3, 1], std_S_indices_population[:, :, 3, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{CaS}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{CaS}}}}}\right](N)$", y_sci=True)

    # Adjust layout to make space for the shared legend
    plt.subplots_adjust(bottom=0.25 / 4)  # Add space at the bottom

    # Shared Legend with Filled Rectangles
    patches = [mpatches.Patch(color="blue", label=r"$g_f$"),
               mpatches.Patch(color="green", label=r"$g_s$"),
               mpatches.Patch(color="red", label=r"$g_u$")]
    fig.legend(
        handles=patches,
        loc="lower center",
        fontsize=20,
        ncol=3,
        frameon=False)

    plt.savefig("convergence_analysis_appendix_part1.pdf", bbox_inches="tight")
    plt.show()

    # Second file (4x2) plot
    fig, axs = plt.subplots(4, 2, figsize=(20, 20), sharex=True)
    axs = axs.flatten()
    plot_with_error(axs[0], n_cells, mean_S_indices_population[:, :, 4, 0], std_S_indices_population[:, :, 4, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{KCa}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{KCa}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[1], n_cells, mean_S_indices_population[:, :, 4, 1], std_S_indices_population[:, :, 4, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{KCa}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{KCa}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[2], n_cells, mean_S_indices_population[:, :, 5, 0], std_S_indices_population[:, :, 5, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{A}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{A}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[3], n_cells, mean_S_indices_population[:, :, 5, 1], std_S_indices_population[:, :, 5, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{A}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{A}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[4], n_cells, mean_S_indices_population[:, :, 6, 0], std_S_indices_population[:, :, 6, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{\bar{g}_{\text{H}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{\bar{g}_{\text{H}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[5], n_cells, mean_S_indices_population[:, :, 6, 1], std_S_indices_population[:, :, 6, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{\bar{g}_{\text{H}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{\bar{g}_{\text{H}}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[6], n_cells, mean_S_indices_population[:, :, 7, 0], std_S_indices_population[:, :, 7, 0], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{g_{\text{leak}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{g_{\text{leak}}}}\right](N)$", y_sci=True)

    plot_with_error(axs[7], n_cells, mean_S_indices_population[:, :, 7, 1], std_S_indices_population[:, :, 7, 1], ["blue", "green", "red"],
                    [r"$g_f$", r"$g_s$", r"$g_u$"], r"Population size $N$",
                    r"$\mu_{S^i_{T_{g_{\text{leak}}}}}(N) \pm \sigma_{\text{pop.}}\left[\mu_{S^i_{T_{g_{\text{leak}}}}}\right](N)$", y_sci=True)

    # Adjust layout to make space for the shared legend
    plt.subplots_adjust(bottom=0.25 / 4)  # Add space at the bottom

    # Shared Legend with Filled Rectangles
    patches = [mpatches.Patch(color="blue", label=r"$g_f$"),
               mpatches.Patch(color="green", label=r"$g_s$"),
               mpatches.Patch(color="red", label=r"$g_u$")]
    fig.legend(
        handles=patches,
        loc="lower center",
        fontsize=20,
        ncol=3,
        frameon=False)

    plt.savefig("convergence_analysis_appendix_part2.pdf", bbox_inches="tight")
    plt.show()


def print_table_4():  # Table 4
    # print all the final std of the estimators
    print("Final std of the estimators")
    print("Mean of DICs: ", std_mean_dics_population[-1])
    print("Std of DICs: ", std_std_dics_population[-1])
    print("95% Mass Interval of DICs: ", std_bounds_95_mass_population[-1])
    print("Sum of First Order Indices: ",
          std_sum_S_indices_population[-1, :, 0])
    print("Sum of Total Indices: ", std_sum_S_indices_population[-1, :, 1])
    print("First order indices: ", std_S_indices_population[-1, :, :, 0].T)
    print("Total indices: ", std_S_indices_population[-1, :, :, 1].T)


if __name__ == "__main__":
    print("Running the main function")

    print("Generating traces graph")
    generate_traces_graph()  # Figure 1

    n_cells = 300000
    print(f"Generating virtual population of {n_cells} cells")
    population = get_a_virtual_population(n_cells)
    print("Computing DICs")
    dics = model(population)

    print("Plotting Figure 3")
    plot_g_distributions()  # Figure 3

    print("Plotting Figure 4")
    plot_dics_distribution(dics)  # Figure 4

    print("Building marginals")
    f_kde, s_kde, u_kde = build_marginal(dics)

    print("Plotting marginals")
    # Not used in the paper; The 'plot_dics_distribution' function is used
    # instead.
    plot_marginals(f_kde, s_kde, u_kde)

    print("Printing Table 1")
    print_table_1()  # Table 1

    print("Printing Table 3")
    print_table_3()  # Table 3

    print("Computing Sobol indices")
    S_indices = sobol_indices(n_cells, population=population)

    print("Plotting Table 2")
    plot_table_2(S_indices)  # Table 2

    print("Convergence analysis... (can take a while depending on the number of cells; ~ 2min for 300000 cells)")
    n_cells, S_indices_population, sum_S_indices_population, mean_dics_population, std_dics_population, bounds_95_mass_population = do_convergence_analysis()
    mean_S_indices_population = S_indices_population.mean(axis=0)
    std_S_indices_population = S_indices_population.std(axis=0)
    mean_sum_S_indices_population = sum_S_indices_population.mean(axis=0)
    std_sum_S_indices_population = sum_S_indices_population.std(axis=0)
    mean_mean_dics_population = mean_dics_population.mean(axis=0)
    std_mean_dics_population = mean_dics_population.std(axis=0)
    mean_std_dics_population = std_dics_population.mean(axis=0)
    std_std_dics_population = std_dics_population.std(axis=0)
    mean_bounds_95_mass_population = bounds_95_mass_population.mean(axis=0)
    std_bounds_95_mass_population = bounds_95_mass_population.std(axis=0)
    print("Done !")

    print("Plotting Figure 5")
    plot_figure_5()  # Figure 5

    print("Plotting Figure 6")
    plot_figure_6()  # Figure 6

    # == appendix figure and table ==
    print("Plotting Figure 7")
    plot_figure_7()  # Figure 7

    print("Printing Table 4")
    print_table_4()  # Table 4
