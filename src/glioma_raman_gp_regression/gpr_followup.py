"""GPR follow-up experiments for patient-level IDH status regression.

These exploratory analyses keep patients as the split and scoring unit. The
Student-t fit is an iteratively reweighted Gaussian approximation, and the
heteroscedastic fit uses within-patient spectral spread only as a noise proxy.
Neither should be interpreted as a fully specified clinical probability model.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEEDS = (42, 101, 202)
N_SPLITS = 5


def load_patient_data(repo_root: str | Path | None = None) -> dict:
    """Load the preprocessed image spectra and construct patient-level data.

    ``X`` is the pixel-weighted patient mean used in notebook 03. ``entry_bags``
    contains one mean spectrum per sample entry, and ``pixel_groups`` retains
    views of within-entry spectra for the input-variability sensitivity check.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    candidates = (root / "data" / "preprocessed.npy", root.parent / "data" / "preprocessed.npy")
    data_path = next((path for path in candidates if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError("Could not find data/preprocessed.npy from the repository or notebook directory.")

    dataset = np.load(data_path, allow_pickle=True)
    channel_count = int(np.asarray(dataset[0]["data"]).shape[-1])
    grouped: dict[int, dict] = {}
    for item in dataset:
        patient_id = int(item["patient_id"])
        class_id = int(item["class_id"])
        target = int(class_id >= 4)  # 0 = IDH-mutant, 1 = IDH-wildtype
        if patient_id not in grouped:
            grouped[patient_id] = {
                "target": target,
                "sum": np.zeros(channel_count, dtype=np.float64),
                "n_pixels": 0,
                "entry_bags": [],
                "pixel_groups": [],
            }
        elif grouped[patient_id]["target"] != target:
            raise ValueError(f"Patient {patient_id} has inconsistent binary labels.")

        pixels = np.asarray(item["data"], dtype=np.float64).reshape(-1, channel_count)
        grouped[patient_id]["sum"] += pixels.sum(axis=0)
        grouped[patient_id]["n_pixels"] += len(pixels)
        grouped[patient_id]["entry_bags"].append(pixels.mean(axis=0))
        grouped[patient_id]["pixel_groups"].append(pixels)

    patient_ids = np.asarray(sorted(grouped), dtype=int)
    y = np.asarray([grouped[int(pid)]["target"] for pid in patient_ids], dtype=int)
    X = np.vstack([
        grouped[int(pid)]["sum"] / grouped[int(pid)]["n_pixels"] for pid in patient_ids
    ])
    entry_bags = [
        np.vstack(grouped[int(pid)]["entry_bags"]) for pid in patient_ids
    ]
    pixel_groups = [grouped[int(pid)]["pixel_groups"] for pid in patient_ids]
    entry_counts = np.asarray([len(grouped[int(pid)]["entry_bags"]) for pid in patient_ids])
    return {
        "data_path": data_path,
        "dataset": dataset,
        "patient_ids": patient_ids,
        "y": y,
        "X": X,
        "entry_bags": entry_bags,
        "pixel_groups": pixel_groups,
        "entry_counts": entry_counts,
        "channel_count": channel_count,
    }


def _kernel(n_components: int, ard: bool = True):
    length_scale = np.ones(n_components, dtype=float) if ard else 1.0
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e2))
    )


def _metric_row(y_true, mean, std, *, seed, method, fold=None):
    """Regression forecast metrics plus secondary IDH separation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    low = mean + norm.ppf(0.05) * std
    high = mean + norm.ppf(0.95) * std
    nlpd = np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((y_true - mean) / std) ** 2)
    row = {
        "seed": seed,
        "fold": fold,
        "method": method,
        "rmse": float(np.sqrt(mean_squared_error(y_true, mean))),
        "mae": float(mean_absolute_error(y_true, mean)),
        "normal_nlpd": float(nlpd),
        "coverage_90": float(np.mean((y_true >= low) & (y_true <= high))),
        "width_90": float(np.mean(high - low)),
        "accuracy_at_0_5": float(accuracy_score(y_true, mean >= 0.5)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, mean >= 0.5)),
        "roc_auc": float(roc_auc_score(y_true, mean)),
    }
    return row


def _student_metric_row(y_true, mean, scale, df, *, seed, method, fold=None):
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.maximum(np.asarray(scale, dtype=float), 1e-8)
    quantile = student_t.ppf(0.95, df)
    low, high = mean - quantile * scale, mean + quantile * scale
    log_score = student_t.logpdf((y_true - mean) / scale, df=df) - np.log(scale)
    return {
        "seed": seed,
        "fold": fold,
        "method": method,
        "rmse": float(np.sqrt(mean_squared_error(y_true, mean))),
        "mae": float(mean_absolute_error(y_true, mean)),
        "student_t_nlpd": float(-np.mean(log_score)),
        "coverage_90": float(np.mean((y_true >= low) & (y_true <= high))),
        "width_90": float(np.mean(high - low)),
        "accuracy_at_0_5": float(accuracy_score(y_true, mean >= 0.5)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, mean >= 0.5)),
        "roc_auc": float(roc_auc_score(y_true, mean)),
    }


def _aggregate_seed_metrics(rows: list[dict], metrics: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["seed", "method"], as_index=False)[list(metrics)].mean()
    summary = grouped.groupby("method")[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index(), grouped


def run_stability(data: dict, seeds: tuple[int, ...] = SEEDS, n_restarts: int = 2) -> dict:
    """Repeat patient CV while checking compact PCA and ARD complexity stability."""
    configs = [
        ("PCA 6 + RBF ARD", 6, True),
        ("PCA 8 + RBF ARD", 8, True),
        ("PCA 12 + RBF ARD", 12, True),
        ("PCA 20 + RBF ARD", 20, True),
        ("PCA 8 + RBF isotropic", 8, False),
        ("PCA 12 + RBF isotropic", 12, False),
    ]
    rows, optimizer_rows = [], []
    ids, y, X = data["patient_ids"], data["y"], data["X"]
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train, valid) in enumerate(splitter.split(X, y), start=1):
            pca = PCA(n_components=20, svd_solver="full")
            train_all = pca.fit_transform(X[train])
            valid_all = pca.transform(X[valid])
            for name, n_components, ard in configs:
                gp = GaussianProcessRegressor(
                    kernel=_kernel(n_components, ard),
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=n_restarts,
                    random_state=seed * 100 + fold,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(train_all[:, :n_components], y[train])
                mean, std = gp.predict(valid_all[:, :n_components], return_std=True)
                rows.append(_metric_row(y[valid], mean, std, seed=seed, method=name, fold=fold))
                theta, bounds = gp.kernel_.theta, gp.kernel_.bounds
                at_bound = np.isclose(theta, bounds[:, 0], atol=0.05, rtol=0.0) | np.isclose(
                    theta, bounds[:, 1], atol=0.05, rtol=0.0
                )
                optimizer_rows.append({
                    "seed": seed,
                    "fold": fold,
                    "method": name,
                    "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
                    "parameters_at_bound": int(at_bound.sum()),
                    "parameter_count": int(len(theta)),
                    "any_parameter_at_bound": bool(at_bound.any()),
                    "convergence_warnings": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
                })

    metrics = ("rmse", "mae", "normal_nlpd", "coverage_90", "width_90", "accuracy_at_0_5", "balanced_accuracy_at_0_5", "roc_auc")
    summary, per_seed = _aggregate_seed_metrics(rows, metrics)
    optimizer_frame = pd.DataFrame(optimizer_rows)
    optimizer_summary = optimizer_frame.groupby("method").agg(
        bound_fit_rate=("any_parameter_at_bound", "mean"),
        mean_bound_parameters=("parameters_at_bound", "mean"),
        mean_log_marginal_likelihood=("log_marginal_likelihood", "mean"),
        convergence_warning_fits=("convergence_warnings", lambda x: int((x > 0).sum())),
    ).reset_index()
    return {
        "summary": summary,
        "per_seed": per_seed,
        "fold_metrics": pd.DataFrame(rows),
        "optimizer_summary": optimizer_summary,
        "optimizer_fits": optimizer_frame,
    }


def _fit_reference_gpr(Z_train, y_train, seed: int):
    gp = GaussianProcessRegressor(
        kernel=_kernel(Z_train.shape[1], ard=True),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(Z_train, y_train)
    return gp


def _student_t_irls(Z_train, y_train, Z_valid, reference_gp, df: float = 4.0, max_iter: int = 10):
    """Robust IRLS approximation to Student-t observation noise.

    This is not exact Student-t likelihood inference. It freezes the Gaussian
    fold's signal kernel and noise scale, then iteratively downweights large
    standardized residuals using Student-t IRLS weights.
    """
    y_train = np.asarray(y_train, dtype=float)
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    y_scale = y_scale if y_scale > 1e-12 else 1.0
    noise_variance = max(float(reference_gp.kernel_.k2.noise_level), 1e-5)
    signal_kernel = reference_gp.kernel_.k1
    weights = np.ones(len(y_train), dtype=float)
    robust_gp = None
    for _ in range(max_iter):
        alpha = noise_variance / np.clip(weights, 0.05, 1.0)
        robust_gp = GaussianProcessRegressor(
            kernel=signal_kernel,
            alpha=alpha,
            normalize_y=True,
            optimizer=None,
        ).fit(Z_train, y_train)
        fitted = robust_gp.predict(Z_train)
        residual_standardized = (y_train - fitted) / y_scale
        proposed = (df + 1.0) / (df + residual_standardized**2 / noise_variance)
        proposed = np.clip(proposed, 0.05, 1.0)
        updated = 0.5 * weights + 0.5 * proposed
        if np.max(np.abs(updated - weights)) < 1e-3:
            weights = updated
            break
        weights = updated

    alpha = noise_variance / np.clip(weights, 0.05, 1.0)
    robust_gp = GaussianProcessRegressor(
        kernel=signal_kernel,
        alpha=alpha,
        normalize_y=True,
        optimizer=None,
    ).fit(Z_train, y_train)
    mean, latent_std = robust_gp.predict(Z_valid, return_std=True)
    predictive_scale = np.sqrt((latent_std / y_scale) ** 2 + noise_variance) * y_scale
    return mean, predictive_scale, weights, noise_variance


def _patient_pc_spread(pca: PCA, pixel_groups: list[np.ndarray]) -> float:
    """Mean within-patient PC variance, accumulated in chunks to limit memory."""
    count = 0
    sum_z = np.zeros(pca.n_components_, dtype=float)
    sum_z2 = np.zeros(pca.n_components_, dtype=float)
    for spectra in pixel_groups:
        for start in range(0, len(spectra), 4096):
            scores = pca.transform(spectra[start : start + 4096])
            count += len(scores)
            sum_z += scores.sum(axis=0)
            sum_z2 += np.square(scores).sum(axis=0)
    if count < 2:
        return 0.0
    variance = np.maximum((sum_z2 - np.square(sum_z) / count) / (count - 1), 0.0)
    return float(np.mean(variance))


def run_noise_and_robustness(data: dict, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Compare Gaussian, Student-t IRLS, and spread-proxy heteroscedastic GPR."""
    ids, y, X = data["patient_ids"], data["y"], data["X"]
    rows, robust_rows, weight_rows = [], [], []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train, valid) in enumerate(splitter.split(X, y), start=1):
            pca = PCA(n_components=8, svd_solver="full").fit(X[train])
            Z_train = pca.transform(X[train])
            Z_valid = pca.transform(X[valid])
            reference = _fit_reference_gpr(Z_train, y[train], seed * 100 + fold)
            base_mean, base_std = reference.predict(Z_valid, return_std=True)
            rows.append(_metric_row(y[valid], base_mean, base_std, seed=seed, method="Gaussian GPR, PCA 8 ARD", fold=fold))

            robust_mean, robust_scale, weights, noise_variance = _student_t_irls(
                Z_train, y[train], Z_valid, reference, df=4.0
            )
            robust_rows.append(_student_metric_row(
                y[valid], robust_mean, robust_scale, 4.0,
                seed=seed, method="Student-t IRLS approximation, df=4", fold=fold,
            ))
            for patient_id, weight in zip(ids[train], weights):
                weight_rows.append({"seed": seed, "fold": fold, "patient_id": int(patient_id), "robust_weight": float(weight)})

            # Within-patient PC spread is only a covariate variability proxy. It
            # combines spatial/biological heterogeneity with measurement noise.
            spreads = np.asarray([
                _patient_pc_spread(pca, data["pixel_groups"][position])
                for position in np.concatenate([train, valid])
            ])
            train_spread, valid_spread = spreads[: len(train)], spreads[len(train) :]
            median_spread = max(float(np.median(train_spread)), 1e-12)
            train_relative = np.clip(np.sqrt(np.maximum(train_spread, 1e-12) / median_spread), 0.5, 2.0)
            valid_relative = np.clip(np.sqrt(np.maximum(valid_spread, 1e-12) / median_spread), 0.5, 2.0)
            train_relative_mean = float(train_relative.mean())
            train_relative /= train_relative_mean
            valid_relative /= train_relative_mean

            noise_variance = max(float(reference.kernel_.k2.noise_level), 1e-5)
            hetero_gp = GaussianProcessRegressor(
                kernel=reference.kernel_.k1,
                alpha=noise_variance * train_relative,
                normalize_y=True,
                optimizer=None,
            ).fit(Z_train, y[train])
            hetero_mean, hetero_latent_std = hetero_gp.predict(Z_valid, return_std=True)
            y_scale = float(np.std(y[train])) or 1.0
            hetero_std = np.sqrt(
                hetero_latent_std**2 + noise_variance * valid_relative * y_scale**2
            )
            rows.append(_metric_row(
                y[valid], hetero_mean, hetero_std,
                seed=seed, method="Heteroscedastic GPR, spread proxy", fold=fold,
            ))

    metric_names = ("rmse", "mae", "normal_nlpd", "coverage_90", "width_90", "accuracy_at_0_5", "balanced_accuracy_at_0_5", "roc_auc")
    gaussian_summary, gaussian_per_seed = _aggregate_seed_metrics(rows, metric_names)
    student_metrics = ("rmse", "mae", "student_t_nlpd", "coverage_90", "width_90", "accuracy_at_0_5", "balanced_accuracy_at_0_5", "roc_auc")
    student_summary, student_per_seed = _aggregate_seed_metrics(robust_rows, student_metrics)
    return {
        "gaussian_summary": gaussian_summary,
        "gaussian_per_seed": gaussian_per_seed,
        "student_t_summary": student_summary,
        "student_t_per_seed": student_per_seed,
        "gaussian_fold_metrics": pd.DataFrame(rows),
        "student_t_fold_metrics": pd.DataFrame(robust_rows),
        "robust_weights": pd.DataFrame(weight_rows),
    }


def _bag_rbf_kernel(bags_a: list[np.ndarray], bags_b: list[np.ndarray], length_scale: float) -> np.ndarray:
    """Mean-embedding RBF kernel between patient bags; PSD by construction."""
    result = np.empty((len(bags_a), len(bags_b)), dtype=float)
    scale2 = max(float(length_scale), 1e-6) ** 2
    for i, bag_a in enumerate(bags_a):
        for j, bag_b in enumerate(bags_b):
            delta = bag_a[:, None, :] - bag_b[None, :, :]
            result[i, j] = np.exp(-0.5 * np.sum(delta * delta, axis=2) / scale2).mean()
    return result


def _fit_precomputed_gp(train_bags, valid_bags, y_train):
    """Fit amplitude, RBF length scale, and noise by training-fold evidence."""
    y_train = np.asarray(y_train, dtype=float)
    y_mean, y_scale = float(y_train.mean()), float(y_train.std())
    y_scale = y_scale if y_scale > 1e-12 else 1.0
    y_normalized = (y_train - y_mean) / y_scale

    def objective(theta):
        amplitude2, length_scale, noise2 = np.exp(theta)
        shape = _bag_rbf_kernel(train_bags, train_bags, length_scale)
        covariance = amplitude2 * shape + (noise2 + 1e-8) * np.eye(len(y_train))
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            alpha = cho_solve(factor, y_normalized, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e20
        logdet_half = np.log(np.diag(factor[0])).sum()
        return float(0.5 * y_normalized @ alpha + logdet_half + 0.5 * len(y_train) * np.log(2 * np.pi))

    starts = [np.log([1.0, length_scale, 0.1]) for length_scale in (0.5, 1.5)]
    fits = [minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(np.log(1e-3), np.log(1e3)), (np.log(0.1), np.log(10.0)), (np.log(1e-5), np.log(1e2))],
    ) for start in starts]
    optimum = min(fits, key=lambda fit: fit.fun)
    amplitude2, length_scale, noise2 = np.exp(optimum.x)
    K_train = amplitude2 * _bag_rbf_kernel(train_bags, train_bags, length_scale)
    K_train += (noise2 + 1e-8) * np.eye(len(y_train))
    factor = cho_factor(K_train, lower=True, check_finite=False)
    alpha = cho_solve(factor, y_normalized, check_finite=False)
    K_cross = amplitude2 * _bag_rbf_kernel(train_bags, valid_bags, length_scale)
    K_diag = amplitude2 * np.diag(_bag_rbf_kernel(valid_bags, valid_bags, length_scale)) + noise2
    mean_normalized = K_cross.T @ alpha
    projected = solve_triangular(factor[0], K_cross, lower=True, check_finite=False)
    variance_normalized = np.maximum(K_diag - np.sum(projected**2, axis=0), 1e-10)
    return (
        y_mean + y_scale * mean_normalized,
        y_scale * np.sqrt(variance_normalized),
        {
            "amplitude_variance": float(amplitude2),
            "length_scale": float(length_scale),
            "noise_variance": float(noise2),
            "negative_log_marginal_likelihood": float(optimum.fun),
        },
    )


def run_patient_bag_gpr(data: dict, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Compare one-mean-per-patient and entry-bag mean-embedding GPR kernels."""
    y, X = data["y"], data["X"]
    rows, hyper_rows = [], []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (train, valid) in enumerate(splitter.split(X, y), start=1):
            pca = PCA(n_components=8, svd_solver="full").fit(X[train])
            # Fit the scaler on one mean per training patient. Fitting it on
            # all entries would let repeated-entry patients carry extra weight.
            scaler = StandardScaler().fit(pca.transform(X[train]))
            entry_bags = [
                scaler.transform(pca.transform(data["entry_bags"][position]))
                for position in range(len(y))
            ]
            patient_scores = scaler.transform(pca.transform(X))
            patient_bags = [patient_scores[position : position + 1] for position in range(len(y))]

            for method, all_bags in (("Patient-mean RBF GPR", patient_bags), ("Patient-entry bag-kernel GPR", entry_bags)):
                mean, std, hyper = _fit_precomputed_gp(
                    [all_bags[position] for position in train],
                    [all_bags[position] for position in valid],
                    y[train],
                )
                rows.append(_metric_row(y[valid], mean, std, seed=seed, method=method, fold=fold))
                hyper_rows.append({"seed": seed, "fold": fold, "method": method, **hyper})

    metrics = ("rmse", "mae", "normal_nlpd", "coverage_90", "width_90", "accuracy_at_0_5", "balanced_accuracy_at_0_5", "roc_auc")
    summary, per_seed = _aggregate_seed_metrics(rows, metrics)
    return {
        "summary": summary,
        "per_seed": per_seed,
        "fold_metrics": pd.DataFrame(rows),
        "hyperparameters": pd.DataFrame(hyper_rows),
    }


def run_all(data: dict, seeds: tuple[int, ...] = SEEDS) -> dict:
    """Run all four GPR-focused follow-up experiments."""
    return {
        "stability": run_stability(data, seeds=seeds),
        "noise_robustness": run_noise_and_robustness(data, seeds=seeds),
        "patient_bag": run_patient_bag_gpr(data, seeds=seeds),
    }


def print_results(results: dict) -> None:
    """Print compact tables suitable for notebook output and the companion note."""
    print("Repeated patient-level GPR stability (mean and SD across split seeds):")
    print(results["stability"]["summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nStability results by split seed:")
    print(results["stability"]["per_seed"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nKernel optimization diagnostics:")
    print(results["stability"]["optimizer_summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nGaussian and heteroscedastic GPR (mean and SD across split seeds):")
    print(results["noise_robustness"]["gaussian_summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nGaussian and heteroscedastic results by split seed:")
    print(results["noise_robustness"]["gaussian_per_seed"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nStudent-t IRLS approximation (mean and SD across split seeds):")
    print(results["noise_robustness"]["student_t_summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nStudent-t results by split seed:")
    print(results["noise_robustness"]["student_t_per_seed"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPatient-mean and entry-bag GPR (mean and SD across split seeds):")
    print(results["patient_bag"]["summary"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPatient-mean and entry-bag results by split seed:")
    print(results["patient_bag"]["per_seed"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nMean robust IRLS weight by patient (values below 1 indicate downweighting):")
    weights = results["noise_robustness"]["robust_weights"].groupby("patient_id")["robust_weight"].mean().sort_values()
    print(weights.head(10).to_string(float_format=lambda x: f"{x:.3f}"))
