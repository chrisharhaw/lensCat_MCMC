# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 00:13:15 2026

@author: jakub
"""

import time
import multiprocessing as mp
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, gaussian_filter1d

c = 299792.458
arcsec_per_radian = 206265.0
seed = 67
model = "P2" # P1 or P2
catalog_file = "Chen2019_130_strong_lenses_with_delta.csv"

eta = -0.066
eta_error = 0.035
sigma_sys_frac = 0.03

omega_min = 0.0
omega_max = 1.0
gamma_0_min = 0.5
gamma_0_max = 2.5
gamma_z_min = -1.5
gamma_z_max = 1.5
sigma_gamma_prior_scale = 0.5

beta_mean = 0.18
beta_error = 0.13
beta_min = beta_mean - 2.0 * beta_error
beta_max = beta_mean + 2.0 * beta_error

n_walkers = 32
burn_in = 750
production_steps = 5000
convergence_multiple = 50.0

n_dist = 24
n_beta = 16
n_gamma_parts = 8
n_gamma_part = 8
n_processes = 6

show_progress = True
make_plots = True
plot_bins = 70
plot_smoothing = 1.0

dist_x, dist_w = leggauss(n_dist)
beta_x, beta_w = leggauss(n_beta)
beta_mid = 0.5 * (beta_min + beta_max)
beta_half = 0.5 * (beta_max - beta_min)
beta_nodes = beta_mid + beta_half * beta_x
beta_norm = norm.cdf(beta_max, loc=beta_mean, scale=beta_error) - norm.cdf(beta_min, loc=beta_mean, scale=beta_error)
log_beta_w = np.log(beta_half) + np.log(beta_w) + norm.logpdf(beta_nodes, loc=beta_mean, scale=beta_error) - np.log(beta_norm)

worker_data = None
worker_gamma = None
worker_kernel = None

def log_diff(log_a, log_b): return log_a + np.log1p(-np.exp(np.minimum(log_b - log_a, -1.0e-15)))
def comoving_int(z, omega_m):
    z = np.asarray(z, dtype=float)
    z_nodes = 0.5 * z[:, None] * (dist_x[None, :] + 1.0)
    E = np.sqrt(omega_m * (1.0 + z_nodes) ** 3 + 1.0 - omega_m)
    return 0.5 * z * np.sum(dist_w[None, :] / E, axis=1)
def dist_ratio(z_l, z_s, omega_m):
    chi_l = comoving_int(z_l, omega_m)
    chi_s = comoving_int(z_s, omega_m)
    return (chi_s - chi_l) / chi_s
def load_catalog():
    catalog = pd.read_csv(catalog_file)
    required = ["z_l", "z_s", "theta_E_arcsec", "theta_eff_arcsec", "theta_ap_arcsec", "sigma_ap_km_s", "sigma_ap_err_km_s", "delta"]
    for column in required: catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    return catalog.dropna(subset=required).reset_index(drop=True)
def prepare_data(catalog):
    theta_eff = catalog["theta_eff_arcsec"].to_numpy(float)
    theta_ap = catalog["theta_ap_arcsec"].to_numpy(float)
    sigma_ap = catalog["sigma_ap_km_s"].to_numpy(float)
    sigma_ap_error = catalog["sigma_ap_err_km_s"].to_numpy(float)
    aperture_ratio = theta_eff / (2.0 * theta_ap)
    correction = aperture_ratio ** eta
    sigma_obs = sigma_ap * correction
    sigma_stat = sigma_ap_error * correction
    sigma_ap_corr = sigma_obs * np.abs(np.log(aperture_ratio)) * eta_error
    sigma_sys = sigma_sys_frac * sigma_obs
    sigma_error = np.sqrt(sigma_stat ** 2 + sigma_ap_corr ** 2 + sigma_sys ** 2)
    return {"z_l": catalog["z_l"].to_numpy(float), "z_s": catalog["z_s"].to_numpy(float), "theta_E": catalog["theta_E_arcsec"].to_numpy(float), "theta_eff": theta_eff, "sigma_obs": sigma_obs, "sigma_error": sigma_error, "delta": catalog["delta"].to_numpy(float), "n_lenses": len(catalog)}
def legendre_grid(n_parts, n_each):
    x, w = leggauss(n_each)
    nodes = []
    weights = []
    for part in range(n_parts):
        left = part / n_parts
        right = (part + 1.0) / n_parts
        nodes.append(0.5 * (left + right) + 0.5 * (right - left) * x)
        weights.append(0.5 * (right - left) * w)
    return np.concatenate(nodes), np.concatenate(weights)
def gamma_grid(data):
    unit_x, unit_w = legendre_grid(n_gamma_parts, n_gamma_part)
    lower = np.empty((data["n_lenses"], n_beta))
    upper = np.empty_like(lower)
    nodes = np.empty((data["n_lenses"], n_beta, len(unit_x)))
    log_weights = np.empty_like(nodes)
    for j, beta in enumerate(beta_nodes):
        lower_j = np.maximum(1.0 + 1.0e-9, 3.0 - data["delta"] + 1.0e-9)
        lower_j = np.maximum(lower_j, 2.0 + 2.0 * beta - data["delta"] + 1.0e-9)
        upper_j = 5.0 - data["delta"] - 1.0e-9
        width = upper_j - lower_j
        nodes[:, j, :] = lower_j[:, None] + width[:, None] * unit_x[None, :]
        log_weights[:, j, :] = np.log(width[:, None]) + np.log(unit_w[None, :])
        lower[:, j] = lower_j
        upper[:, j] = upper_j
    return {"nodes": nodes, "log_weights": log_weights, "lower": lower, "upper": upper}
def log_trunc_norm(lower, upper, mean, sigma):
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    a, b, mean = np.broadcast_arrays(a, b, mean)
    result = np.empty(a.shape)
    positive = a > 0.0
    if np.any(~positive): result[~positive] = log_diff(log_ndtr(b[~positive]), log_ndtr(a[~positive]))
    if np.any(positive): result[positive] = log_diff(log_ndtr(-a[positive]), log_ndtr(-b[positive]))
    return result
def log_gamma_pdf(gamma_nodes, lower, upper, mean, sigma_gamma):
    z = (gamma_nodes - mean[:, None, None]) / sigma_gamma
    log_norm = log_trunc_norm(lower, upper, mean[:, None], sigma_gamma)
    return -0.5 * z ** 2 - np.log(sigma_gamma) - 0.5 * np.log(2.0 * np.pi) - log_norm[:, :, None]
def gamma_mean(pars, z_l):
    if model == "P1": return np.full(len(z_l), pars[1])
    return pars[1] + pars[2] * z_l
def epl_kernel(gamma_quad, data):
    gamma_nodes = gamma_quad["nodes"]
    delta = data["delta"][:, None, None]
    theta_E = data["theta_E"][:, None, None]
    theta_eff = data["theta_eff"][:, None, None]
    beta = beta_nodes[None, :, None]
    xi = gamma_nodes + delta - 2.0
    term1 = np.exp(gammaln((xi - 1.0) / 2.0) - gammaln(xi / 2.0))
    term2 = np.exp(gammaln((xi + 1.0) / 2.0) - gammaln((xi + 2.0) / 2.0))
    bracket = term1 - beta * term2
    log_profile = gammaln(gamma_nodes / 2.0) + gammaln(delta / 2.0) - gammaln((gamma_nodes - 1.0) / 2.0) - gammaln((delta - 1.0) / 2.0)
    F = bracket * np.exp(log_profile)
    kernel = c ** 2 / (2.0 * np.sqrt(np.pi)) * (theta_E / arcsec_per_radian) * (3.0 - delta) / ((xi - 2.0 * beta) * (3.0 - xi)) * F * (theta_eff / (2.0 * theta_E)) ** (2.0 - gamma_nodes)
    valid = (gamma_nodes > 1.0) & (xi > 1.0) & (xi < 3.0) & (xi - 2.0 * beta > 0.0) & (bracket > 0.0) & np.isfinite(kernel) & (kernel > 0.0)
    return np.where(valid, kernel, np.nan), valid
def epl_sigma(omega_m, data, kernel):
    ratio = dist_ratio(data["z_l"], data["z_s"], omega_m)
    sigma2 = kernel["value"] / ratio[:, None, None]
    return np.where(kernel["valid"], np.sqrt(np.maximum(sigma2, 0.0)), np.nan)
def log_prior(pars):
    omega_m = pars[0]
    gamma_0 = pars[1]
    sigma_gamma = pars[-1]
    if not omega_min < omega_m < omega_max: return -np.inf
    if not gamma_0_min < gamma_0 < gamma_0_max: return -np.inf
    if model == "P2" and not gamma_z_min < pars[2] < gamma_z_max: return -np.inf
    if sigma_gamma <= 0.0: return -np.inf
    return -0.5 * (sigma_gamma / sigma_gamma_prior_scale) ** 2
def log_likelihood(pars):
    omega_m = pars[0]
    sigma_gamma = pars[-1]
    mean = gamma_mean(pars, worker_data["z_l"])
    sigma_model = epl_sigma(omega_m, worker_data, worker_kernel)
    log_p_gamma = log_gamma_pdf(worker_gamma["nodes"], worker_gamma["lower"], worker_gamma["upper"], mean, sigma_gamma)
    variance = worker_data["sigma_error"][:, None, None] ** 2
    residual = worker_data["sigma_obs"][:, None, None] - sigma_model
    log_sigma = -0.5 * (residual ** 2 / variance + np.log(2.0 * np.pi * variance))
    log_gamma = logsumexp(worker_gamma["log_weights"] + log_p_gamma + log_sigma, axis=2)
    log_each_beta = np.sum(log_gamma, axis=0)
    total = logsumexp(log_each_beta + log_beta_w)
    return total if np.isfinite(total) else -np.inf
def log_probability(pars):
    prior = log_prior(pars)
    return -np.inf if not np.isfinite(prior) else prior + log_likelihood(pars)
def init_worker(data, gamma_quad, kernel):
    global worker_data, worker_gamma, worker_kernel
    worker_data = data
    worker_gamma = gamma_quad
    worker_kernel = kernel
def make_starting_positions(run_seed):
    rng = np.random.default_rng(run_seed)
    n_dim = 3 if model == "P1" else 4
    center = np.array([0.30, 2.03, 0.18]) if model == "P1" else np.array([0.30, 2.08, -0.23, 0.18])
    scale = np.array([0.05, 0.04, 0.03]) if model == "P1" else np.array([0.05, 0.04, 0.04, 0.03])
    start = np.zeros((n_walkers, n_dim))
    for i in range(n_walkers):
        position = center + scale * rng.normal(size=n_dim)
        while not np.isfinite(log_prior(position)): position = center + scale * rng.normal(size=n_dim)
        start[i] = position
    return start
def run_mcmc(data, gamma_quad, kernel):
    np.random.seed(seed)
    start = make_starting_positions(seed)
    n_dim = start.shape[1]
    init_worker(data, gamma_quad, kernel)
    with mp.Pool(processes=n_processes, initializer=init_worker, initargs=(data, gamma_quad, kernel)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        state = sampler.run_mcmc(start, burn_in, progress=show_progress)
        sampler.reset()
        sampler.run_mcmc(state, production_steps, progress=show_progress)
    samples = sampler.get_chain(flat=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        converged = bool(np.all(production_steps > convergence_multiple * tau))
    except (emcee.autocorr.AutocorrError, ValueError):
        tau = np.full(n_dim, np.nan)
        converged = False
    return samples, converged, tau, float(np.mean(sampler.acceptance_fraction))
def parameter_summary(values):
    p16, median, p84 = np.percentile(values, [16, 50, 84])
    return {"p16": p16, "median": median, "p84": p84, "error_minus": median - p16, "error_plus": p84 - median}
def normalized_pdf(values, value_range):
    density, edges = np.histogram(values, bins=plot_bins, range=value_range, density=True)
    density = gaussian_filter1d(density.astype(float), plot_smoothing)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.max(density) > 0.0: density = density / np.max(density)
    return centers, density
def density_grid(x, y, x_range, y_range):
    density, x_edges, y_edges = np.histogram2d(x, y, bins=plot_bins, range=[x_range, y_range])
    density = gaussian_filter(density.T.astype(float), plot_smoothing)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return np.meshgrid(x_centers, y_centers), density
def contour_levels(density):
    sorted_density = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative = cumulative / cumulative[-1]
    level_68 = sorted_density[np.searchsorted(cumulative, 0.68)]
    level_95 = sorted_density[np.searchsorted(cumulative, 0.95)]
    if level_68 <= level_95: level_68 = np.nextafter(level_95, np.inf)
    return level_95, level_68
def plot_contours(ax, x, y, x_range, y_range):
    (x_grid, y_grid), density = density_grid(x, y, x_range, y_range)
    level_95, level_68 = contour_levels(density)
    maximum = np.max(density)
    if maximum <= level_68: maximum = np.nextafter(level_68, np.inf)
    ax.contourf(x_grid, y_grid, density, levels=[level_95, level_68, maximum], colors=[to_rgba("blue", 0.15), to_rgba("blue", 0.32)])
    ax.contour(x_grid, y_grid, density, levels=[level_95, level_68], colors=["blue"], linestyles=["--", "-"], linewidths=1.2)
def corner_plot(samples, labels, ranges):
    n_parameters = len(labels)
    fig, axes = plt.subplots(n_parameters, n_parameters, figsize=(2.7 * n_parameters, 2.7 * n_parameters))
    for row in range(n_parameters):
        for column in range(n_parameters):
            ax = axes[row, column]
            if row < column:
                ax.axis("off")
            elif row == column:
                x_pdf, pdf = normalized_pdf(samples[:, column], ranges[column])
                ax.plot(x_pdf, pdf, linewidth=1.5)
                ax.set_xlim(ranges[column])
                ax.set_ylim(0.0, 1.08)
                if column == 0: ax.set_ylabel("PDF")
            else:
                plot_contours(ax, samples[:, column], samples[:, row], ranges[column], ranges[row])
                ax.set_xlim(ranges[column])
                ax.set_ylim(ranges[row])
            if row == n_parameters - 1 and row >= column: ax.set_xlabel(labels[column])
            elif row != column: ax.set_xticklabels([])
            if column == 0 and row > 0: ax.set_ylabel(labels[row])
            elif column > 0: ax.set_yticklabels([])
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.97, wspace=0.08, hspace=0.08)
    plt.show()
def main():
    start_time = time.perf_counter()
    catalog = load_catalog()
    data = prepare_data(catalog)
    gamma_quad = gamma_grid(data)
    kernel_value, kernel_valid = epl_kernel(gamma_quad, data)
    kernel = {"value": kernel_value, "valid": kernel_valid}
    names = ["Omega_m", "gamma_0", "sigma_gamma"] if model == "P1" else ["Omega_m", "gamma_0", "gamma_z", "sigma_gamma"]
    labels = [r"$\Omega_m$", r"$\gamma_0$", r"$\sigma_\gamma$"] if model == "P1" else [r"$\Omega_m$", r"$\gamma_0$", r"$\gamma_z$", r"$\sigma_\gamma$"]
    ranges = [(0.0, 1.0), (0.5, 2.5), (0.0, 0.8)] if model == "P1" else [(0.0, 1.0), (0.5, 2.5), (-1.5, 1.5), (0.0, 0.8)]
    print("HIERARCHICAL", model, "MCMC")
    print("Processes =", n_processes)
    print()
    samples, converged, tau, acceptance = run_mcmc(data, gamma_quad, kernel)
    print()
    print("RESULTS")
    for j, name in enumerate(names):
        result = parameter_summary(samples[:, j])
        print(name, "=", round(result["median"], 4), "+", round(result["error_plus"], 4), "-", round(result["error_minus"], 4))
    print("Converged =", converged)
    print("Tau =", np.round(tau, 2))
    print("Acceptance fraction =", round(acceptance, 3))
    print("Runtime minutes =", round((time.perf_counter() - start_time) / 60.0, 2))
    if make_plots: corner_plot(samples, labels, ranges)

if __name__ == "__main__":
    mp.freeze_support()
    main()