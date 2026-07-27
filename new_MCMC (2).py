# -*- coding: utf-8 -*-
import os
import glob
import emcee
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.stats import skewnorm

c = 299792.458
arcsec_per_radian = 206265.0
seed = 67
catalog_folder = "mock_catalogs"
catalog_pattern = "Chen_mock_*.csv"
max_catalogs = 150

eta = -0.066
eta_error = 0.035
sigma_systematic_relative_error = 0.03

kappa_mean = 0.00649
kappa_shape = 2.0
kappa_scale = 0.07
kappa_min = -0.20
kappa_max = 0.30

omega_min = 0.0
omega_max = 1.0
n_walkers = 24
burn_in = 500
production_steps = 2000
convergence_multiple = 50.0

n_distance_quad = 24
n_kappa_quad = 24
show_progress = False

distance_x, distance_w = leggauss(n_distance_quad)
kappa_delta = kappa_shape / np.sqrt(1.0 + kappa_shape ** 2)
kappa_location = kappa_mean - kappa_scale * kappa_delta * np.sqrt(2.0 / np.pi)
kappa_x, kappa_w = leggauss(n_kappa_quad)
kappa_middle = 0.5 * (kappa_min + kappa_max)
kappa_half_width = 0.5 * (kappa_max - kappa_min)
kappa_nodes = kappa_middle + kappa_half_width * kappa_x
kappa_normalization = skewnorm.cdf(kappa_max, kappa_shape, loc=kappa_location, scale=kappa_scale) - skewnorm.cdf(kappa_min, kappa_shape, loc=kappa_location, scale=kappa_scale)
log_kappa_weights = np.log(kappa_half_width) + np.log(kappa_w) + skewnorm.logpdf(kappa_nodes, kappa_shape, loc=kappa_location, scale=kappa_scale) - np.log(kappa_normalization)


def comoving_integral(z, omega_m):
    z = np.asarray(z, dtype=float)
    z_nodes = 0.5 * z[:, None] * (distance_x[None, :] + 1.0)
    E = np.sqrt(omega_m * (1.0 + z_nodes) ** 3 + 1.0 - omega_m)
    return 0.5 * z * np.sum(distance_w[None, :] / E, axis=1)


def distance_ratio(z_l, z_s, omega_m):
    chi_l = comoving_integral(z_l, omega_m)
    chi_s = comoving_integral(z_s, omega_m)
    return (chi_s - chi_l) / chi_s


def log_prior(parameters):
    omega_m = parameters[0]
    if not omega_min < omega_m < omega_max:
        return -np.inf
    return 0.0


def sigma_model_with_kappa(parameters, z_l, z_s, theta_E):
    omega_m = parameters[0]
    ratio = distance_ratio(z_l, z_s, omega_m)
    if np.any(ratio <= 0.0) or not np.all(np.isfinite(ratio)):
        return None
    theta_radian = theta_E / arcsec_per_radian
    return c * np.sqrt(theta_radian[:, None] * (1.0 - kappa_nodes[None, :]) / (4.0 * np.pi * ratio[:, None]))


def sigma_model_without_kappa(parameters, z_l, z_s, theta_E):
    omega_m = parameters[0]
    ratio = distance_ratio(z_l, z_s, omega_m)
    if np.any(ratio <= 0.0) or not np.all(np.isfinite(ratio)):
        return None
    theta_radian = theta_E / arcsec_per_radian
    return c * np.sqrt(theta_radian / (4.0 * np.pi * ratio))


def log_likelihood_with_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error):
    sigma_model = sigma_model_with_kappa(parameters, z_l, z_s, theta_E)
    if sigma_model is None or not np.all(np.isfinite(sigma_model)):
        return -np.inf
    variance = sigma_error[:, None] ** 2
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        return -np.inf
    residual = sigma_obs[:, None] - sigma_model
    log_sigma_likelihood = -0.5 * (residual ** 2 / variance + np.log(2.0 * np.pi * variance))
    loglike_each_lens = logsumexp(log_sigma_likelihood + log_kappa_weights[None, :], axis=1)
    total_loglike = np.sum(loglike_each_lens)
    if not np.isfinite(total_loglike):
        return -np.inf
    return total_loglike


def log_likelihood_without_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error):
    sigma_model = sigma_model_without_kappa(parameters, z_l, z_s, theta_E)
    if sigma_model is None or not np.all(np.isfinite(sigma_model)):
        return -np.inf
    variance = sigma_error ** 2
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        return -np.inf
    residual = sigma_obs - sigma_model
    total_loglike = -0.5 * np.sum(residual ** 2 / variance + np.log(2.0 * np.pi * variance))
    if not np.isfinite(total_loglike):
        return -np.inf
    return total_loglike


def log_probability_with_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error):
    prior = log_prior(parameters)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood_with_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error)


def log_probability_without_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error):
    prior = log_prior(parameters)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood_without_kappa(parameters, z_l, z_s, theta_E, sigma_obs, sigma_error)


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


def load_catalog(filename):
    return pd.read_csv(filename)


def run_mcmc(z_l, z_s, theta_E, sigma_obs, sigma_error, run_seed, use_kappa):
    rng = np.random.default_rng(run_seed)
    np.random.seed(run_seed)
    start = rng.uniform(0.10, 0.90, size=(n_walkers, 1))
    probability_function = log_probability_with_kappa if use_kappa else log_probability_without_kappa
    sampler = emcee.EnsembleSampler(n_walkers, 1, probability_function, args=(z_l, z_s, theta_E, sigma_obs, sigma_error))
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


mock_files = sorted(glob.glob(os.path.join(catalog_folder, catalog_pattern)))

if max_catalogs is not None:
    mock_files = mock_files[:max_catalogs]

print("Number of catalogs =", len(mock_files))
print("Convergence criterion: production steps >", convergence_multiple, "* tau for Omega_m")
print("Comparison: kappa marginalization versus kappa_ext = 0")
print()

results = []

for i, mock_file in enumerate(mock_files):
    print()
    print("MOCK", i + 1, "/", len(mock_files), "-", os.path.basename(mock_file))

    catalog = load_catalog(mock_file)
    z_l = catalog["z_l"].to_numpy(dtype=float)
    z_s = catalog["z_s"].to_numpy(dtype=float)
    theta_E = catalog["theta_E_arcsec"].to_numpy(dtype=float)
    sigma_obs, sigma_error_kappa, sigma_error_zero = prepare_observations(catalog)

    samples_kappa, converged_kappa = run_mcmc(z_l, z_s, theta_E, sigma_obs, sigma_error_kappa, seed + i, True)
    samples_zero, converged_zero = run_mcmc(z_l, z_s, theta_E, sigma_obs, sigma_error_zero, seed + i, False)

    omega_kappa = parameter_summary(samples_kappa[:, 0])
    omega_zero = parameter_summary(samples_zero[:, 0])
    delta_omega = omega_kappa["median"] - omega_zero["median"]

    print("KAPPA MARGINALIZED")
    print_parameter_result("Omega_m", omega_kappa)
    print("Convergence =", converged_kappa)

    print()
    print("KAPPA_EXT = 0")
    print_parameter_result("Omega_m", omega_zero)
    print("Convergence =", converged_zero)

    results.append({
        "catalog": os.path.basename(mock_file),
        "n_lenses": len(catalog),
        "omega_kappa_p16": omega_kappa["p16"],
        "omega_kappa_median": omega_kappa["median"],
        "omega_kappa_p84": omega_kappa["p84"],
        "omega_kappa_error_minus": omega_kappa["error_minus"],
        "omega_kappa_error_plus": omega_kappa["error_plus"],
        "omega_zero_p16": omega_zero["p16"],
        "omega_zero_median": omega_zero["median"],
        "omega_zero_p84": omega_zero["p84"],
        "omega_zero_error_minus": omega_zero["error_minus"],
        "omega_zero_error_plus": omega_zero["error_plus"],
        "delta_omega": delta_omega,
        "converged_kappa": converged_kappa,
        "converged_zero": converged_zero,
    })

results_table = pd.DataFrame(results)

print()
print()
print("SUMMARY")

kappa_results = results_table.loc[results_table["converged_kappa"]]
zero_results = results_table.loc[results_table["converged_zero"]]
both_converged = results_table.loc[results_table["converged_kappa"] & results_table["converged_zero"]]

n_catalogs = len(results_table)
n_kappa = len(kappa_results)
n_zero = len(zero_results)
n_both = len(both_converged)

print()
print("CONVERGENCE")
print("With kappa marginalization =", n_kappa, "/", n_catalogs)
print("With kappa_ext = 0 =", n_zero, "/", n_catalogs)
print("Converged in both fits =", n_both, "/", n_catalogs)

print()
print("KAPPA MARGINALIZATION")

if n_kappa > 0:
    omega_mean = kappa_results["omega_kappa_median"].mean()
    omega_minus = kappa_results["omega_kappa_error_minus"].mean()
    omega_plus = kappa_results["omega_kappa_error_plus"].mean()
    print(f"Mean Omega_m = {omega_mean:.4f} +{omega_plus:.4f} -{omega_minus:.4f}")
else:
    print("No converged fits.")

print()
print("KAPPA_EXT = 0")

if n_zero > 0:
    omega_mean = zero_results["omega_zero_median"].mean()
    omega_minus = zero_results["omega_zero_error_minus"].mean()
    omega_plus = zero_results["omega_zero_error_plus"].mean()
    print(f"Mean Omega_m = {omega_mean:.4f} +{omega_plus:.4f} -{omega_minus:.4f}")
else:
    print("No converged fits.")

print()
print("EFFECT OF IGNORING KAPPA")

if n_both > 0:
    omega_shift = both_converged["delta_omega"].to_numpy(dtype=float)
    omega_kappa_paired_mean = both_converged["omega_kappa_median"].mean()
    omega_zero_paired_mean = both_converged["omega_zero_median"].mean()
    omega_shift_mean = np.mean(omega_shift)
    omega_shift_median = np.median(omega_shift)
    omega_shift_p16, omega_shift_p84 = np.percentile(omega_shift, [16, 84])
    omega_shift_std = np.std(omega_shift, ddof=1) if n_both > 1 else 0.0
    omega_shift_sem = omega_shift_std / np.sqrt(n_both)
    print(f"Paired mean Omega_m with kappa marginalization = {omega_kappa_paired_mean:.4f}")
    print(f"Paired mean Omega_m with kappa_ext = 0 = {omega_zero_paired_mean:.4f}")
    print(f"Mean change in Omega_m = {omega_shift_mean:.4f}")
    print(f"Median change in Omega_m = {omega_shift_median:.4f}")
    print(f"16th-84th percentile of change = [{omega_shift_p16:.4f}, {omega_shift_p84:.4f}]")
    print(f"Standard deviation of change = {omega_shift_std:.4f}")
    print(f"Standard error of mean change = {omega_shift_sem:.4f}")
else:
    print("No catalogs converged in both fits.")