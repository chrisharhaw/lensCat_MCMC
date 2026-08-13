import time
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import norm
from tqdm import tqdm

c = 299792.458
arcsec_per_radian = 206265.0
catalog_file = "Chen2019_130_strong_lenses_with_delta.csv"

eta = -0.066
eta_error = 0.035
sigma_sys_frac = 0.03

beta_mean = 0.18
beta_error = 0.13
beta_min = beta_mean - 2.0 * beta_error
beta_max = beta_mean + 2.0 * beta_error

s_min = 50.0
s_max = 600.0

sigma_gamma_min = 0.03
sigma_gamma_max = 0.35
sigma_gamma_prior_scale = 0.20

omega_grid = np.array([0.000, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.38, 0.50, 0.70, 0.90])

n_dist = 32
n_beta = 16
n_s_obs = 12
n_gamma_parts = 8
n_gamma_part = 8
n_s_den = 40

opt_options = {"maxiter": 200, "maxfev": 700, "xtol": 1.0e-4, "ftol": 1.0e-4}
bounds = [(0.50, 2.50), (-1.50, 1.50), (-1.50, 1.50), (np.log(sigma_gamma_min), np.log(sigma_gamma_max))]
starts = [np.array([1.21, -0.24, 0.66, np.log(0.10)]), np.array([1.40, -0.10, 0.53, np.log(0.14)]), np.array([1.85, -0.09, 0.09, np.log(0.14)])]
n_processes = 6

dist_x, dist_w = leggauss(n_dist)
beta_x, beta_w = leggauss(n_beta)
beta_mid = 0.5 * (beta_min + beta_max)
beta_half = 0.5 * (beta_max - beta_min)
beta_nodes = beta_mid + beta_half * beta_x
beta_norm = norm.cdf(beta_max, loc=beta_mean, scale=beta_error) - norm.cdf(beta_min, loc=beta_mean, scale=beta_error)
log_beta_w = np.log(beta_half) + np.log(beta_w) + norm.logpdf(beta_nodes, loc=beta_mean, scale=beta_error) - np.log(beta_norm)
s_x, s_w = hermgauss(n_s_obs)
log_s_w = np.log(s_w) - 0.5 * np.log(np.pi)

worker_data = None
worker_gamma = None

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

def reff_hinv_kpc(z_l, theta_eff, omega_m):
    chi_l = comoving_int(z_l, omega_m)
    d_l = (c / 100.0) * chi_l / (1.0 + z_l)
    return (theta_eff / arcsec_per_radian) * d_l * 1000.0

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
    sigma_err = np.sqrt(sigma_stat ** 2 + sigma_ap_corr ** 2)
    sigma_sys = sigma_sys_frac * sigma_obs
    return {"z_l": catalog["z_l"].to_numpy(float), "z_s": catalog["z_s"].to_numpy(float), "theta_E": catalog["theta_E_arcsec"].to_numpy(float), "theta_eff": theta_eff, "sigma_obs": sigma_obs, "sigma_measurement": sigma_err, "sigma_systematic": sigma_sys, "delta": catalog["delta"].to_numpy(float), "n_lenses": len(catalog)}

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

def log_s_prior(s):
    s = np.asarray(s, dtype=float)
    result = np.full(s.shape, -np.inf)
    valid = (s >= s_min) & (s <= s_max)
    result[valid] = -np.log(s_max - s_min)
    return result

def s_den_grid():
    x, w = leggauss(n_s_den)
    mid = 0.5 * (s_min + s_max)
    half = 0.5 * (s_max - s_min)
    nodes = mid + half * x
    return nodes, np.log(half) + np.log(w) + log_s_prior(nodes)
def s_obs_nodes(data): return data["sigma_obs"][:, None] + np.sqrt(2.0) * data["sigma_measurement"][:, None] * s_x[None, :]

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
    z = (gamma_nodes[:, None, :, :] - mean[:, :, None, None]) / sigma_gamma
    log_norm = log_trunc_norm(lower[:, None, :], upper[:, None, :], mean[:, :, None], sigma_gamma)
    return -0.5 * z ** 2 - np.log(sigma_gamma) - 0.5 * np.log(2.0 * np.pi) - log_norm[:, :, :, None]

def gamma_mean(pars, s_values, radius_eff, data):
    gamma_0, gamma_z, gamma_s, sigma_gamma = pars
    sigma_tilde = (s_values / 100.0) ** 2 / (radius_eff[:, None] / 10.0)
    return gamma_0 + gamma_z * data["z_l"][:, None] + gamma_s * np.log10(sigma_tilde)

def epl_sigma(omega_m, gamma_nodes, data):
    ratio = dist_ratio(data["z_l"], data["z_s"], omega_m)
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
    sigma2 = c ** 2 / (2.0 * np.sqrt(np.pi)) * (theta_E / arcsec_per_radian) / ratio[:, None, None] * (3.0 - delta) / ((xi - 2.0 * beta) * (3.0 - xi)) * F * (theta_eff / (2.0 * theta_E)) ** (2.0 - gamma_nodes)
    valid = (gamma_nodes > 1.0) & (xi > 1.0) & (xi < 3.0) & (xi - 2.0 * beta > 0.0) & (bracket > 0.0) & np.isfinite(sigma2) & (sigma2 > 0.0)
    return np.where(valid, np.sqrt(np.maximum(sigma2, 0.0)), np.nan), valid

def log_sgamma_prior(sigma_gamma): return -np.inf if not sigma_gamma_min < sigma_gamma < sigma_gamma_max else -0.5 * (sigma_gamma / sigma_gamma_prior_scale) ** 2

def make_context(omega_m, data, gamma_quad):
    radius_eff = reff_hinv_kpc(data["z_l"], data["theta_eff"], omega_m)
    s_num = s_obs_nodes(data)
    log_prior_num = log_s_prior(s_num)
    s_den_nodes, log_den_weights = s_den_grid()
    s_den = np.broadcast_to(s_den_nodes[None, :], (data["n_lenses"], len(s_den_nodes)))
    sigma_model, valid = epl_sigma(omega_m, gamma_quad["nodes"], data)
    model_sigma = data["sigma_systematic"][:, None, None, None]
    residual_num = s_num[:, :, None, None] - sigma_model[:, None, :, :]
    residual_den = s_den[:, :, None, None] - sigma_model[:, None, :, :]
    log_model_num = -0.5 * (residual_num ** 2 / model_sigma ** 2 + np.log(2.0 * np.pi * model_sigma ** 2))
    log_model_den = -0.5 * (residual_den ** 2 / model_sigma ** 2 + np.log(2.0 * np.pi * model_sigma ** 2))
    log_model_num = np.where(np.isfinite(log_prior_num)[:, :, None, None] & valid[:, None, :, :], log_model_num, -np.inf)
    log_model_den = np.where(valid[:, None, :, :], log_model_den, -np.inf)
    return {"radius_eff": radius_eff, "s_num": s_num, "log_prior_num": log_prior_num, "s_den": s_den, "log_den_weights": log_den_weights, "log_model_num": log_model_num, "log_model_den": log_model_den}

def log_target(pars, data, ctx, gamma_quad):
    prior = log_sgamma_prior(pars[3])
    if not np.isfinite(prior): return -np.inf
    mu_num = gamma_mean(pars, ctx["s_num"], ctx["radius_eff"], data)
    mu_den = gamma_mean(pars, ctx["s_den"], ctx["radius_eff"], data)
    log_p_gamma_num = log_gamma_pdf(gamma_quad["nodes"], gamma_quad["lower"], gamma_quad["upper"], mu_num, pars[3])
    log_p_gamma_den = log_gamma_pdf(gamma_quad["nodes"], gamma_quad["lower"], gamma_quad["upper"], mu_den, pars[3])
    log_gamma_num = logsumexp(gamma_quad["log_weights"][:, None, :, :] + log_p_gamma_num + ctx["log_model_num"], axis=3)
    log_gamma_den = logsumexp(gamma_quad["log_weights"][:, None, :, :] + log_p_gamma_den + ctx["log_model_den"], axis=3)
    log_n = logsumexp(log_s_w[None, :, None] + ctx["log_prior_num"][:, :, None] + log_gamma_num, axis=1)
    log_d = logsumexp(ctx["log_den_weights"][None, :, None] + log_gamma_den, axis=1)
    if not np.all(np.isfinite(log_n)) or not np.all(np.isfinite(log_d)): return -np.inf
    log_each_beta = np.sum(log_n - log_d, axis=0)
    return prior + logsumexp(log_each_beta + log_beta_w)

def init_worker(data, gamma_quad):
    global worker_data, worker_gamma
    worker_data = data
    worker_gamma = gamma_quad
    
def fit_omega(omega_m):
    ctx = make_context(float(omega_m), worker_data, worker_gamma)
    best = None
    for start_index, start in enumerate(starts):
        def objective(x):
            pars = np.array([x[0], x[1], x[2], np.exp(x[3])])
            val = log_target(pars, worker_data, ctx, worker_gamma)
            return 1.0e100 if not np.isfinite(val) else -val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(objective, start, method="Powell", bounds=bounds, options=opt_options)
        pars = np.array([result.x[0], result.x[1], result.x[2], np.exp(result.x[3])])
        val = log_target(pars, worker_data, ctx, worker_gamma)
        fit = {"omega_m": float(omega_m), "gamma_0": pars[0], "gamma_z": pars[1], "gamma_s": pars[2], "sigma_gamma": pars[3], "log_target": val, "success": bool(result.success), "nfev": int(result.nfev), "start": start_index}
        if best is None or fit["log_target"] > best["log_target"]: best = fit
    return best

def make_plot(table):
    table = table.sort_values("omega_m")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(table["omega_m"], table["delta_log_target"], marker="o", linewidth=1.8)
    ax.axvline(0.38, linestyle="--", linewidth=1.0, label=r"$\Omega_m=0.38$")
    ax.set_xlabel(r"$\Omega_m$")
    ax.set_ylabel(r"$\Delta\log\,target$")
    ax.set_xlim(0.0, 0.92)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("P3_profile.png", dpi=300, bbox_inches="tight")
    plt.show()
    
def main():
    start_time = time.perf_counter()
    catalog = load_catalog()
    data = prepare_data(catalog)
    gamma_quad = gamma_grid(data)
    print("P3 HIERARCHICAL PROFILE")
    print("Number of lenses =", data["n_lenses"])
    print("Parent prior: uniform in s, range =", [s_min, s_max], "km/s")
    print("Omega grid =", omega_grid)
    print("Processes =", n_processes)
    print()
    results = []
    with mp.Pool(processes=n_processes, initializer=init_worker, initargs=(data, gamma_quad)) as pool:
        with tqdm(total=len(omega_grid), desc="P3 profile", unit="fit") as progress:
            for i, result in enumerate(pool.imap_unordered(fit_omega, omega_grid), 1):
                results.append(result)
                print(i, "/", len(omega_grid), "Omega_m =", result["omega_m"], "target =", round(result["log_target"], 5), "sigma_gamma =", round(result["sigma_gamma"], 4), "success =", result["success"])
                progress.update(1)
    table = pd.DataFrame(results).sort_values("omega_m").reset_index(drop=True)
    table["delta_log_target"] = table["log_target"] - table["log_target"].max()
    best = table.loc[table["log_target"].idxmax()]
    row_038 = table.iloc[np.argmin(np.abs(table["omega_m"].to_numpy() - 0.38))]
    print()
    print("SUMMARY")
    print("Best Omega_m =", best["omega_m"])
    print("Delta log target at 0.38 =", round(float(row_038["delta_log_target"]), 4))
    print("Best gamma_0 =", round(float(best["gamma_0"]), 5))
    print("Best gamma_z =", round(float(best["gamma_z"]), 5))
    print("Best gamma_s =", round(float(best["gamma_s"]), 5))
    print("Best sigma_gamma =", round(float(best["sigma_gamma"]), 5))
    print("Boundary =", bool(best["omega_m"] == np.min(omega_grid)))
    print("All successful =", bool(table["success"].all()))
    print("Runtime minutes =", round((time.perf_counter() - start_time) / 60.0, 2))
    make_plot(table)

if __name__ == "__main__":
    mp.freeze_support()
    main()