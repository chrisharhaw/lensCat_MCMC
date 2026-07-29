# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 02:01:55 2026

@author: jakub
"""
import os
import glob
import emcee
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import gamma, logsumexp
from scipy.stats import norm
from slsim.Util.ParamDistributions.kext_gext_distributions import LineOfSightDistribution

c = 299792.458
arcsec_per_radian = 206265.0
seed = 67

#catalog_folder = "."
#catalog_pattern = "Chen2019_161_strong_lenses.csv"
catalog_folder = "mock_catalogs"
catalog_pattern = "Chen_mock_*.csv"
max_catalogs = 50

delta_file = "delta_values.csv"
kappa_distribution_file = "no_nonlinear_distributions.h5"

eta = -0.066
eta_error = 0.035
sigma_systematic_relative_error = 0.03

omega_min = 0.0
omega_max = 1.0

gamma_0_min = 0.5
gamma_0_max = 2.5

gamma_z_min = -1.5
gamma_z_max = 1.5

gamma_s_min = -1.5
gamma_s_max = 1.5

beta_mean = 0.18
beta_error = 0.13
beta_min = beta_mean - 2.0 * beta_error
beta_max = beta_mean + 2.0 * beta_error

n_walkers = 32
burn_in = 750
production_steps = 5000
convergence_multiple = 50.0

n_distance_quad = 24
n_kappa_quad = 24
n_kappa_draws = 5000
n_beta_quad = 24

show_progress = False

distance_x, distance_w = leggauss(n_distance_quad)

beta_x, beta_w = leggauss(n_beta_quad)
beta_middle = 0.5 * (beta_min + beta_max)
beta_half_width = 0.5 * (beta_max - beta_min)
beta_nodes = beta_middle + beta_half_width * beta_x
beta_normalization = norm.cdf(beta_max, loc=beta_mean, scale=beta_error) - norm.cdf(beta_min, loc=beta_mean, scale=beta_error)
log_beta_weights = np.log(beta_half_width) + np.log(beta_w) + norm.logpdf(beta_nodes, loc=beta_mean, scale=beta_error) - np.log(beta_normalization)

kappa_prior_cache = {}

def normalize_lens_name(values):
    return values.astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

def load_delta_table():
    delta_table = pd.read_csv(delta_file)
    normalized_columns = {column: str(column).strip().lower().replace("δ", "delta") for column in delta_table.columns}
    name_column = next((column for column, name in normalized_columns.items() if "lens" in name or "name" in name or "system" in name), delta_table.columns[0])
    delta_column = next((column for column, name in normalized_columns.items() if "delta" in name), None)
    delta_table = delta_table[[name_column, delta_column]].copy()
    delta_table.columns = ["lens_name_delta", "delta"]
    delta_table["lens_key"] = normalize_lens_name(delta_table["lens_name_delta"])
    delta_table["delta"] = pd.to_numeric(delta_table["delta"], errors="coerce")
    delta_table = delta_table.dropna(subset=["delta"])
    delta_table = delta_table.drop_duplicates(subset=["lens_key"])
    return delta_table[["lens_key", "delta"]]

def create_los_distribution():
    return LineOfSightDistribution(nonlinear_correction_path=None, no_correction_path=kappa_distribution_file)

def extract_kappa(result):
    if isinstance(result, dict):
        key = next((name for name in result if "kappa" in str(name).lower()), None)
        return np.asarray(result[key], dtype=float).reshape(-1)
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=float).reshape(-1)
    result_array = np.asarray(result, dtype=float)
    if result_array.ndim >= 2 and result_array.shape[-1] >= 2:
        return result_array[..., 0].reshape(-1)
    return result_array.reshape(-1)

def draw_kappa_samples(los_distribution, z_lens, z_source, number_of_samples):
    samples = np.empty(number_of_samples, dtype=float)
    for j in range(number_of_samples):
        result = los_distribution.get_kappa_gamma(z_source=float(z_source), z_lens=float(z_lens), use_nonlinear_correction=False)
        samples[j] = extract_kappa(result)[0]
    return samples

def prepare_kappa_priors(z_l, z_s, los_distribution):
    kappa_nodes_lens = np.zeros((len(z_l), n_kappa_quad))
    log_kappa_weights_lens = np.full((len(z_l), n_kappa_quad), -np.log(n_kappa_quad))
    quantiles = (np.arange(n_kappa_quad) + 0.5) / n_kappa_quad
    for i in range(len(z_l)):
        key = (round(float(z_l[i]), 6), round(float(z_s[i]), 6))
        if key not in kappa_prior_cache:
            samples = draw_kappa_samples(los_distribution, z_l[i], z_s[i], n_kappa_draws)
            samples = samples[np.isfinite(samples) & (samples < 1.0)]
            nodes = np.quantile(samples, quantiles)
            kappa_prior_cache[key] = nodes
        kappa_nodes_lens[i] = kappa_prior_cache[key]
    return kappa_nodes_lens, log_kappa_weights_lens

def comoving_integral(z, omega_m):
    z = np.asarray(z, dtype=float)
    z_nodes = 0.5 * z[:, None] * (distance_x[None, :] + 1.0)
    E = np.sqrt(omega_m * (1.0 + z_nodes) ** 3 + 1.0 - omega_m)
    return 0.5 * z * np.sum(distance_w[None, :] / E, axis=1)

def distance_ratio(z_l, z_s, omega_m):
    chi_l = comoving_integral(z_l, omega_m)
    chi_s = comoving_integral(z_s, omega_m)
    return (chi_s - chi_l) / chi_s

def surface_density_tilde(z_l, theta_eff, sigma_obs, omega_m):
    chi_l = comoving_integral(z_l, omega_m)
    distance_l_hinv_mpc = (c / 100.0) * chi_l / (1.0 + z_l)
    radius_eff_hinv_kpc = (theta_eff / arcsec_per_radian) * distance_l_hinv_mpc * 1000.0
    return (sigma_obs / 100.0) ** 2 / (radius_eff_hinv_kpc / 10.0)

def gamma_profile(parameters, z_l, theta_eff, sigma_obs):
    omega_m, gamma_0, gamma_z, gamma_s = parameters
    sigma_tilde = surface_density_tilde(z_l, theta_eff, sigma_obs, omega_m)
    return gamma_0 + gamma_z * z_l + gamma_s * np.log10(sigma_tilde)

def log_prior(parameters, z_l, theta_eff, sigma_obs, delta):
    omega_m, gamma_0, gamma_z, gamma_s = parameters
    if not omega_min < omega_m < omega_max:
        return -np.inf
    if not gamma_0_min < gamma_0 < gamma_0_max:
        return -np.inf
    if not gamma_z_min < gamma_z < gamma_z_max:
        return -np.inf
    if not gamma_s_min < gamma_s < gamma_s_max:
        return -np.inf
    gamma_lens = gamma_profile(parameters, z_l, theta_eff, sigma_obs)
    xi = gamma_lens + delta - 2.0
    if np.any(gamma_lens <= 1.0):
        return -np.inf
    if np.any(delta <= 1.0):
        return -np.inf
    if np.any(xi <= 1.0):
        return -np.inf
    if np.any(xi >= 3.0):
        return -np.inf
    if np.any(xi[:, None] - 2.0 * beta_nodes[None, :] <= 0.0):
        return -np.inf
    return 0.0

def sigma_model_epl_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, delta):
    omega_m, gamma_0, gamma_z, gamma_s = parameters
    ratio = distance_ratio(z_l, z_s, omega_m)
    gamma_lens = gamma_profile(parameters, z_l, theta_eff, sigma_obs)
    xi = gamma_lens + delta - 2.0
    theta_E_radian = theta_E / arcsec_per_radian
    xi_grid = xi[:, None]
    beta_grid = beta_nodes[None, :]
    first_gamma_term = gamma((xi_grid - 1.0) / 2.0) / gamma(xi_grid / 2.0)
    second_gamma_term = beta_grid * gamma((xi_grid + 1.0) / 2.0) / gamma((xi_grid + 2.0) / 2.0)
    F = (first_gamma_term - second_gamma_term) * gamma(gamma_lens[:, None] / 2.0) * gamma(delta[:, None] / 2.0) / (gamma((gamma_lens[:, None] - 1.0) / 2.0) * gamma((delta[:, None] - 1.0) / 2.0))
    sigma_squared = c ** 2 / (2.0 * np.sqrt(np.pi)) * theta_E_radian[:, None] / ratio[:, None] * (3.0 - delta[:, None]) / ((xi_grid - 2.0 * beta_grid) * (3.0 - xi_grid)) * F * (theta_eff[:, None] / (2.0 * theta_E[:, None])) ** (2.0 - gamma_lens[:, None])
    if np.any(sigma_squared <= 0.0) or not np.all(np.isfinite(sigma_squared)):
        return None
    return np.sqrt(sigma_squared)

def sigma_model_epl_with_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, delta, kappa_nodes_lens):
    sigma_zero = sigma_model_epl_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, delta)
    if sigma_zero is None:
        return None
    return sigma_zero[:, :, None] * np.sqrt(1.0 - kappa_nodes_lens[:, None, :])

def log_likelihood_with_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta, kappa_nodes_lens, log_kappa_weights_lens):
    sigma_model = sigma_model_epl_with_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, delta, kappa_nodes_lens)
    if sigma_model is None or not np.all(np.isfinite(sigma_model)):
        return -np.inf
    variance = sigma_error[:, None, None] ** 2
    residual = sigma_obs[:, None, None] - sigma_model
    log_sigma_likelihood = -0.5 * (residual ** 2 / variance + np.log(2.0 * np.pi * variance))
    loglike_each_lens_beta = logsumexp(log_sigma_likelihood + log_kappa_weights_lens[:, None, :], axis=2)
    loglike_each_beta = np.sum(loglike_each_lens_beta, axis=0)
    total_loglike = logsumexp(loglike_each_beta + log_beta_weights)
    return total_loglike if np.isfinite(total_loglike) else -np.inf

def log_likelihood_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta):
    sigma_model = sigma_model_epl_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, delta)
    if sigma_model is None or not np.all(np.isfinite(sigma_model)):
        return -np.inf
    variance = sigma_error[:, None] ** 2
    residual = sigma_obs[:, None] - sigma_model
    log_sigma_likelihood = -0.5 * (residual ** 2 / variance + np.log(2.0 * np.pi * variance))
    loglike_each_beta = np.sum(log_sigma_likelihood, axis=0)
    total_loglike = logsumexp(loglike_each_beta + log_beta_weights)
    return total_loglike if np.isfinite(total_loglike) else -np.inf

def log_probability_with_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta, kappa_nodes_lens, log_kappa_weights_lens):
    prior = log_prior(parameters, z_l, theta_eff, sigma_obs, delta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood_with_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta, kappa_nodes_lens, log_kappa_weights_lens)

def log_probability_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta):
    prior = log_prior(parameters, z_l, theta_eff, sigma_obs, delta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood_without_kappa(parameters, z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta)

def prepare_observations(catalog):
    theta_eff = catalog["theta_eff_arcsec"].to_numpy(dtype=float)
    theta_ap = catalog["theta_ap_arcsec"].to_numpy(dtype=float)
    sigma_ap = catalog["sigma_ap_km_s"].to_numpy(dtype=float)
    sigma_ap_error = catalog["sigma_ap_err_km_s"].to_numpy(dtype=float)
    aperture_ratio = theta_eff / (2.0 * theta_ap)
    correction = aperture_ratio ** eta
    sigma_obs = sigma_ap * correction
    sigma_statistical = sigma_ap_error * correction
    sigma_aperture = sigma_obs * np.abs(np.log(aperture_ratio)) * eta_error
    sigma_error_kappa = np.sqrt(sigma_statistical ** 2 + sigma_aperture ** 2)
    sigma_systematic = sigma_systematic_relative_error * sigma_obs
    sigma_error_zero = np.sqrt(sigma_statistical ** 2 + sigma_aperture ** 2 + sigma_systematic ** 2)
    return sigma_obs, sigma_error_kappa, sigma_error_zero

def load_catalog(filename, delta_table):
    catalog = pd.read_csv(filename)
    catalog["lens_key"] = normalize_lens_name(catalog["lens_name"])
    catalog = catalog.merge(delta_table, on="lens_key", how="inner")
    required_columns = ["z_l", "z_s", "theta_E_arcsec", "theta_eff_arcsec", "theta_ap_arcsec", "sigma_ap_km_s", "sigma_ap_err_km_s", "delta"]
    for column in required_columns:
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    catalog = catalog.dropna(subset=required_columns).copy()
    return catalog

def make_starting_positions(z_l, theta_eff, sigma_obs, delta, run_seed):
    rng = np.random.default_rng(run_seed)
    start = np.zeros((n_walkers, 4))
    center = np.array([0.30, 1.21, -0.22, 0.66])
    scale = np.array([0.05, 0.05, 0.05, 0.05])
    for i in range(n_walkers):
        position = center + scale * rng.normal(size=4)
        while not np.isfinite(log_prior(position, z_l, theta_eff, sigma_obs, delta)):
            position = center + scale * rng.normal(size=4)
        start[i] = position
    return start

def run_mcmc(z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta, run_seed, use_kappa, kappa_nodes_lens=None, log_kappa_weights_lens=None):
    np.random.seed(run_seed)
    start = make_starting_positions(z_l, theta_eff, sigma_obs, delta, run_seed)
    probability_function = log_probability_with_kappa if use_kappa else log_probability_without_kappa
    probability_arguments = (z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta, kappa_nodes_lens, log_kappa_weights_lens) if use_kappa else (z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error, delta)
    sampler = emcee.EnsembleSampler(n_walkers, 4, probability_function, args=probability_arguments)
    state = sampler.run_mcmc(start, burn_in, progress=show_progress)
    sampler.reset()
    sampler.run_mcmc(state, production_steps, progress=show_progress)
    samples = sampler.get_chain(flat=True)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        converged = bool(np.all(production_steps > convergence_multiple * tau))
    except (emcee.autocorr.AutocorrError, ValueError):
        converged = False
    return samples, converged

def parameter_summary(values):
    p16, median, p84 = np.percentile(values, [16, 50, 84])
    return {"p16": p16, "median": median, "p84": p84, "error_minus": median - p16, "error_plus": p84 - median}

def print_parameter_result(name, result):
    print(name, "=", round(result["median"], 4), "+", round(result["error_plus"], 4), "-", round(result["error_minus"], 4))

delta_table = load_delta_table()
los_distribution = create_los_distribution()
np.random.seed(seed + 100000)

mock_files = sorted(glob.glob(os.path.join(catalog_folder, catalog_pattern)))

if max_catalogs is not None:
    mock_files = mock_files[:max_catalogs]

print("Number of catalogs =", len(mock_files))
print("Number of lenses with delta =", len(delta_table))
print("Kappa distribution file =", kappa_distribution_file)
print("Comparison: EPL P3 with kappa marginalization versus EPL P3 with kappa_ext = 0 and 3 percent error")
print()

parameter_names = ["Omega_m", "gamma_0", "gamma_z", "gamma_s"]
results = []

for i, mock_file in enumerate(mock_files):
    print()
    print("MOCK", i + 1, "/", len(mock_files), "-", os.path.basename(mock_file))
    catalog = load_catalog(mock_file, delta_table)
    z_l = catalog["z_l"].to_numpy(dtype=float)
    z_s = catalog["z_s"].to_numpy(dtype=float)
    theta_E = catalog["theta_E_arcsec"].to_numpy(dtype=float)
    theta_eff = catalog["theta_eff_arcsec"].to_numpy(dtype=float)
    delta = catalog["delta"].to_numpy(dtype=float)
    sigma_obs, sigma_error_kappa, sigma_error_zero = prepare_observations(catalog)
    kappa_nodes_lens, log_kappa_weights_lens = prepare_kappa_priors(z_l, z_s, los_distribution)
    samples_kappa, converged_kappa = run_mcmc(z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error_kappa, delta, seed + i, True, kappa_nodes_lens, log_kappa_weights_lens)
    samples_zero, converged_zero = run_mcmc(z_l, z_s, theta_E, theta_eff, sigma_obs, sigma_error_zero, delta, seed + i, False)
    summaries_kappa = [parameter_summary(samples_kappa[:, j]) for j in range(4)]
    summaries_zero = [parameter_summary(samples_zero[:, j]) for j in range(4)]
    print("KAPPA MARGINALIZED")
    for name, summary in zip(parameter_names, summaries_kappa):
        print_parameter_result(name, summary)
    print("Convergence =", converged_kappa)
    print()
    print("KAPPA_EXT = 0 WITH 3 PERCENT ERROR")
    for name, summary in zip(parameter_names, summaries_zero):
        print_parameter_result(name, summary)
    print("Convergence =", converged_zero)
    result = {
        "catalog": os.path.basename(mock_file),
        "n_lenses": len(catalog),
        "converged_kappa": converged_kappa,
        "converged_zero": converged_zero
    }
    for j, name in enumerate(parameter_names):
        key = name.lower()
        result[key + "_kappa_median"] = summaries_kappa[j]["median"]
        result[key + "_kappa_error_minus"] = summaries_kappa[j]["error_minus"]
        result[key + "_kappa_error_plus"] = summaries_kappa[j]["error_plus"]
        result[key + "_zero_median"] = summaries_zero[j]["median"]
        result[key + "_zero_error_minus"] = summaries_zero[j]["error_minus"]
        result[key + "_zero_error_plus"] = summaries_zero[j]["error_plus"]
        result["delta_" + key] = summaries_kappa[j]["median"] - summaries_zero[j]["median"]
    results.append(result)

results_table = pd.DataFrame(results)
results_table.to_csv("EPL_P3_results.csv", index=False)

both_converged = results_table.loc[results_table["converged_kappa"] & results_table["converged_zero"]]

print()
print("SUMMARY")
print("Converged with kappa =", int(results_table["converged_kappa"].sum()), "/", len(results_table))
print("Converged with kappa_ext = 0 =", int(results_table["converged_zero"].sum()), "/", len(results_table))
print("Converged in both fits =", len(both_converged), "/", len(results_table))

if len(both_converged) > 0:
    for name in parameter_names:
        key = name.lower()

        kappa_medians = both_converged[key + "_kappa_median"].to_numpy(dtype=float)
        zero_kappa_medians = both_converged[key + "_zero_median"].to_numpy(dtype=float)

        kappa_errors_minus = both_converged[key + "_kappa_error_minus"].to_numpy(dtype=float)
        kappa_errors_plus = both_converged[key + "_kappa_error_plus"].to_numpy(dtype=float)
        chen_errors_minus = both_converged[key + "_zero_error_minus"].to_numpy(dtype=float)
        chen_errors_plus = both_converged[key + "_zero_error_plus"].to_numpy(dtype=float)

        kappa_widths = kappa_errors_minus + kappa_errors_plus
        zero_kappa_widths = chen_errors_minus + chen_errors_plus
        shifts = kappa_medians - zero_kappa_medians

        shift_p16, shift_median, shift_p84 = np.percentile(shifts, [16, 50, 84])

        mean_kappa_median = np.mean(kappa_medians)
        mean_zero_kappa_medians = np.mean(zero_kappa_medians)

        kappa_catalog_sd = np.std(kappa_medians, ddof=1) if len(kappa_medians) > 1 else 0.0
        chen_catalog_sd = np.std(zero_kappa_medians, ddof=1) if len(zero_kappa_medians) > 1 else 0.0
        shift_sd = np.std(shifts, ddof=1) if len(shifts) > 1 else 0.0

        mean_kappa_error_minus = np.mean(kappa_errors_minus)
        mean_kappa_error_plus = np.mean(kappa_errors_plus)
        mean_chen_error_minus = np.mean(chen_errors_minus)
        mean_chen_error_plus = np.mean(chen_errors_plus)

        mean_kappa_width = np.mean(kappa_widths)
        mean_zero_kappa_widths = np.mean(zero_kappa_widths)

        mean_shift = np.mean(shifts)
        positive_shift_fraction = np.mean(shifts > 0.0)

        width_ratio = mean_kappa_width / mean_zero_kappa_widths if mean_zero_kappa_widths > 0.0 else np.nan
        lower_error_ratio = mean_kappa_error_minus / mean_chen_error_minus if mean_chen_error_minus > 0.0 else np.nan
        upper_error_ratio = mean_kappa_error_plus / mean_chen_error_plus if mean_chen_error_plus > 0.0 else np.nan

        print()
        print(name)
        print()
        print(" KAPPA MARGINALIZED")
        print(f"Mean posterior median = {mean_kappa_median:.4f}")
        print(f"Standard deviation between catalogs = {kappa_catalog_sd:.4f}")
        print(f"Mean posterior errors = -{mean_kappa_error_minus:.4f} +{mean_kappa_error_plus:.4f}")
        print(f"Mean 68 percent interval width = {mean_kappa_width:.4f}")
        print()
        print("KAPPA_EXT = 0 WITH 3 PERCENT ERROR")
        print(f"Mean posterior median = {mean_zero_kappa_medians:.4f}")
        print(f"Standard deviation between catalogs = {chen_catalog_sd:.4f}")
        print(f"Mean posterior errors = -{mean_chen_error_minus:.4f} +{mean_chen_error_plus:.4f}")
        print(f"Mean 68 percent interval width = {mean_zero_kappa_widths:.4f}")
        print()
        print("DIFFERENCE BETWEEN MODELS")
        print(f"Mean shift = {mean_shift:.4f}")
        print(f"Median shift = {shift_median:.4f}")
        print(f"16th to 84th percentile of shifts = [{shift_p16:.4f}, {shift_p84:.4f}]")
        print(f"Standard deviation of shifts = {shift_sd:.4f}")
        print(f"Catalogs with positive shift = {100.0 * positive_shift_fraction:.1f} percent")
        print(f"Mean interval width ratio = {width_ratio:.4f}")
        print(f"Mean lower error ratio = {lower_error_ratio:.4f}")
        print(f"Mean upper error ratio = {upper_error_ratio:.4f}")

else:
    print("No catalogs converged in both fits.")