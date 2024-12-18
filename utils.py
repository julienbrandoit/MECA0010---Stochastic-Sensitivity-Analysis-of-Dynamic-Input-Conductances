"""
Author : BRANDOIT Julien

This file should be considered as a BLACK BOX. It contains utility functions that are used in the main scripts.

Some of the functions may be not used in the main scripts. This is because this file is part of a larger project and some functions are used in other scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# == General utils functions ==

def gsigmoid(V, A, B, C, D):
    return A + B / (1 + np.exp((V + D) / C))


  
def d_gsigmoid(V, A, B, C, D):
    return -B * np.exp((V + D) / C) / (C * (1 + np.exp((V + D) / C)) ** 2)

def gamma_uniform_mean_std_matching(uniform_a, uniform_b):
    # shape and scale ?
    # in numpy k is the shape and theta is the scale
    # p(x) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k))
    # such that the mean is k*theta and the variance is k*theta^2

    # the mean of the uniform distribution is (a+b)/2
    # the variance of the uniform distribution is (b-a)^2/12

    # solving for k and theta
    # k = 3(a+b)^2 / (b-a)^2 and theta = (b-a)^2/(6 * (a+b))

    a = uniform_a
    b = uniform_b
    
    p = a+b
    q_sq = (b-a)**2
    k = 3*p**2/q_sq
    theta = q_sq/(6*p)
    return k, theta

# == simulation utils functions ==

def simulate_population_multiprocessing(simulation_function, population, u0, T_final, dt, params, max_workers=8):
    traces = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(u0, individual, T_final, dt, params) for individual in population]
        results = list(tqdm(executor.map(simulation_function, tasks), total=len(population), desc='Simulating population (multiprocessing)'))
    
    for result in results:
        traces.append(result)
    
    return traces

def simulate_population_t_eval_multiprocessing(simulation_function, population, u0, t_eval, params, max_workers=8):
    traces = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(u0, individual, t_eval, params) for individual in population]
        results = list(tqdm(executor.map(simulation_function, tasks), total=len(population), desc='Simulating population (multiprocessing)'))
    
    for result in results:
        traces.append(result)
    
    return traces

def simulate_population(simulation_function, population, u0, T_final, dt, params):
    traces = []
    for i in tqdm(range(len(population)), desc='Simulating population'):
        individual = population[i]
        trace = simulation_function([u0, individual, T_final, dt, params])
        traces.append(trace)
    return traces

# == analysis utils functions ==

def analyze_individual(individual_params):
    individual, params, DICs_find_critical_functions = individual_params
    g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak = individual
    V_init = np.arange(-100, 0, 0.5)
    V_th, g_th = DICs_find_critical_functions(V_init, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca'],
                                        max_iter=1000, x_tol=1e-6, y_tol=1e-6, verbose=True)
    
    return g_th, V_th

def analyze_population(population, params, DICs_find_critical_functions):
    DICs = []
    V_ths = []
    for i in tqdm(range(len(population)), desc='Analyzing population'):
        individual = population[i]
        g_th, V_th = analyze_individual((individual, params, DICs_find_critical_functions))
        DICs.append(g_th)
        V_ths.append(V_th)

    DICs = np.array(DICs)
    V_ths = np.array(V_ths)

    return DICs, V_ths

def analyze_population_multiprocessing(population, params, DICs_find_critical_functions, max_workers=8):
    DICs = []
    V_ths = []
    
    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        individual_params = [(individual, params, DICs_find_critical_functions) for individual in population]
        
        # Map the process_individual function to the executor
        results = list(tqdm(executor.map(analyze_individual, individual_params), total=len(population), desc='Analyzing population (multiprocessing)'))
    
    # Collect results
    for g_th, V_th in results:
        DICs.append(g_th)
        V_ths.append(V_th)
    
    DICs = np.array(DICs)
    V_ths = np.array(V_ths)

    return DICs, V_ths

# == plot utils functions ==

def plot_ss_trace(t, V, t_min=3000):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    start_t = np.where(t > t_min)[0][0]
    tt = t[start_t:]
    VV = V[start_t:]
    ax.plot(tt, VV)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('Voltage trace')
    plt.show()


def plot_n_ss_traces(t, V, n_traces=5, t_min=3000, max_per_row=3):
    n_rows = int(np.ceil(n_traces / max_per_row))

    n_cols = min(n_traces, max_per_row)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*6.4, n_rows*4))
    for i in range(n_traces):
        tt = t[i]
        VV = V[i]
        start_t = np.where(tt > t_min)[0][0]
        tt = tt[start_t:]
        VV = VV[start_t:]
        row = i // max_per_row
        col = i % max_per_row
        if n_rows == 1:
            ax[col].plot(tt, VV)
            ax[col].set_xlabel('Time (ms)')
            ax[col].set_ylabel('Voltage (mV)')
            ax[col].set_title('Voltage trace {}'.format(i))
        else:    
            ax[row, col].plot(tt, VV)
            ax[row, col].set_xlabel('Time (ms)')
            ax[row, col].set_ylabel('Voltage (mV)')
            ax[row, col].set_title('Voltage trace {}'.format(i))
    plt.tight_layout()
    plt.show()

# == DICs computation related utils functions ==

def w_factor(V, tau_x, tau_1, tau_2, default=1):
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x(V) > tau_1(V)) & (tau_x(V) <= tau_2(V))
    mask_2 = tau_x(V) > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1]))-np.log(tau_x(V[mask_1]))) / (np.log(tau_2(V[mask_1]))-np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result

def w_factor_constant_tau(V, tau_x, tau_1, tau_2, default=1):
    V = np.asarray(V)
    result = np.ones_like(V)*default
    mask_1 = (tau_x > tau_1(V)) & (tau_x <= tau_2(V))
    mask_2 = tau_x > tau_2(V)
    result[mask_1] =  (np.log(tau_2(V[mask_1]))-np.log(tau_x)) / (np.log(tau_2(V[mask_1]))-np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result

def get_w_factors(V, tau_x, tau_f, tau_s, tau_u):
    return w_factor(V, tau_x, tau_f, tau_s, default=1), w_factor(V, tau_x, tau_s, tau_u, default=1)

def get_w_factors_constant_tau(V, tau_x, tau_f, tau_s, tau_u):
    return w_factor_constant_tau(V, tau_x, tau_f, tau_s), w_factor_constant_tau(V, tau_x, tau_s, tau_u)

# == Analytical and numerical utils functions ==

def find_first_decreasing_zero(x, y, get_index = False):
    # find the first index where y is decreasing and y is zero
    
    for i in range(len(y)-1):
        if y[i] >= 0 and y[i+1] < 0:
            if get_index:
                return i, x[i]
            else:
                return x[i]

    if get_index:
        return None, None        
    return None

def bisection(f, a, b, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):

    f_a = f(a)
    f_b = f(b)

    if abs(f_a) <= y_tol:
        return a
    if abs(f_b) <= y_tol:
        return b

    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) <= y_tol or (b - a) / 2 < x_tol:
            return c
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    if verbose:
        print("Bisection method did not converge after {} iterations".format(max_iter))

    return c

def find_first_decreasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    # find the first index where y is decreasing and y is zero
    x = x_init
    y = f(x)

    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0:
            return bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
                
    return np.nan

def find_first_increasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    # find the first index where y is decreasing and y is zero
    x = x_init
    y = f(x)

    for i in range(len(y)-1):
        if y[i] < 0 and y[i+1] > 0:
            return bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
                
    return np.nan

def find_first_decreasing_and_first_increasing_zero_bisection(x_init, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=False):
    # find the first index where y is decreasing and y is zero
    x = x_init
    y = f(x)

    first_decreasing_zero = np.nan
    first_increasing_zero = np.nan

    for i in range(len(y)-1):
        if y[i] > 0 and y[i+1] < 0 and np.isnan(first_decreasing_zero):
            first_decreasing_zero = bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
        if y[i] < 0 and y[i+1] > 0 and np.isnan(first_increasing_zero):
            first_increasing_zero = bisection(f, x[i], x[i+1], y_tol, x_tol, max_iter, verbose)
                
    return first_decreasing_zero, first_increasing_zero

def extract_ss_trace(sols, t_min=3000):
    m = sols.ndim
    if m == 2:
        sols = sols[np.newaxis, :, :]
    t0 = sols[0, 0, :]


    sols = sols[:, :, t0 >= t_min]
    if m == 2:
        return sols[0, :, :]
    return sols

def get_spiking_times(t, V, spike_high_threshold=10, spike_low_threshold=0):
    # this function does NOT handle batch processing

    above_threshold = V > spike_high_threshold
    below_threshold = V < spike_low_threshold

    spike_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    spike_ends = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1
    
    # Only consider starts that have a corresponding end after them
    if len(spike_starts) == 0 or len(spike_ends) == 0:
        return np.array([]), np.array([])
    
    valid_starts = spike_starts[spike_starts < spike_ends[-1]]

    spike_times = t[valid_starts]
    
    return valid_starts, spike_times