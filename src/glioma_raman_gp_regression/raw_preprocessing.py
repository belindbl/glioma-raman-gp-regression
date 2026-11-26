"""Raw APOLLO-format data conversion and GPR preprocessing experiments."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
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


RAW_CHANNELS = 1738
PROCESSED_CHANNELS = 1167
# APOLLO reports removed intervals 0--248.651529 and 1797.210--2697.804 cm^-1.
# These half-open raw-channel slices produce the reported 1167 retained channels.
APOLLO_RETAINED_SLICES = ((104, 907), (1374, 1738))
DEFAULT_SEEDS = (42, 101, 202)


def _parse_raw_identity(path: Path) -> tuple[int, int, str]:
    patient_match = re.match(r"^HF-(\d+)", path.stem, flags=re.IGNORECASE)
    class_match = re.fullmatch(r"LGm-(\d+)", path.parent.name, flags=re.IGNORECASE)
    if patient_match is None or class_match is None:
        raise ValueError(f"Cannot parse patient/class identity from {path}.")
    patient_id, class_id = int(patient_match.group(1)), int(class_match.group(1))
    if class_id < 1 or class_id > 6:
        raise ValueError(f"Unexpected LGm class folder {path.parent.name!r}.")
    return patient_id, class_id, path.stem


def _array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    chunk_size = 8 * 1024 * 1024
    for start in range(0, len(view), chunk_size):
        digest.update(view[start : start + chunk_size])
    return digest.hexdigest()


def _alignment_rows(raw_root: Path, preprocessed_path: Path) -> list[dict]:
    """Match all raw files to current entries and validate IDs and dimensions."""
    processed = np.load(preprocessed_path, allow_pickle=True)
    expected = []
    seen_processed = set()
    for item in processed:
        name = str(item["name"])
        if name in seen_processed:
            raise ValueError(f"Duplicate sample name in current dataset: {name}")
        seen_processed.add(name)
        expected.append({
            "name": name,
            "patient_id": int(item["patient_id"]),
            "class_id": int(item["class_id"]),
            "processed_shape": tuple(np.asarray(item["data"]).shape),
        })
    del processed
    gc.collect()

    raw_files = list(Path(raw_root).rglob("*.npy"))
    raw_by_name = {}
    for path in raw_files:
        identity = _parse_raw_identity(path)
        name = identity[2]
        if name in raw_by_name:
            raise ValueError(f"Duplicate raw sample name: {name}")
        raw_by_name[name] = (path, identity)

    if len(raw_by_name) != len(expected):
        raise ValueError(f"Raw/current counts differ: {len(raw_by_name)} raw, {len(expected)} processed.")
    if set(raw_by_name) != {row["name"] for row in expected}:
        missing = sorted({row["name"] for row in expected} - set(raw_by_name))
        extra = sorted(set(raw_by_name) - {row["name"] for row in expected})
        raise ValueError(f"Sample names do not match; missing raw={missing}; extra raw={extra}.")

    rows = []
    for row in expected:
        path, (patient_id, class_id, name) = raw_by_name[row["name"]]
        if patient_id != row["patient_id"] or class_id != row["class_id"]:
            raise ValueError(
                f"Identity mismatch for {name}: raw patient/class={patient_id}/{class_id}, "
                f"current={row['patient_id']}/{row['class_id']}."
            )
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3 or raw.shape[-1] != RAW_CHANNELS:
            raise ValueError(f"Unexpected raw array shape for {name}: {raw.shape}.")
        if row["processed_shape"] != (*raw.shape[:2], PROCESSED_CHANNELS):
            raise ValueError(
                f"Spatial/channel dimensions do not align for {name}: "
                f"raw={raw.shape}, processed={row['processed_shape']}."
            )
        rows.append({
            **row,
            "raw_shape": tuple(raw.shape),
            "pixel_count": int(raw.shape[0] * raw.shape[1]),
            "raw_dtype": str(raw.dtype),
            "source_file": path.relative_to(raw_root).as_posix(),
            "source_path": path,
        })
        del raw
    return rows


def merge_raw_dataset(
    raw_root: str | Path,
    preprocessed_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge raw files into an ordered object-array .npy matching current IDs.

    The merged records retain raw float32 image arrays and use the current
    dataset's sample order. The output also includes a relative source path.
    Existing output files are never overwritten.
    """
    raw_root, preprocessed_path, output_path = map(Path, (raw_root, preprocessed_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged dataset: {output_path}")
    rows = _alignment_rows(raw_root, preprocessed_path)
    merged = np.empty(len(rows), dtype=object)
    hashes = {}
    for index, row in enumerate(rows):
        raw = np.load(row["source_path"], allow_pickle=False)
        hashes[row["name"]] = _array_sha256(raw)
        merged[index] = {
            "data": raw,
            "name": row["name"],
            "class_id": row["class_id"],
            "patient_id": row["patient_id"],
            "source_file": row["source_file"],
        }
        del raw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, merged, allow_pickle=True)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    del merged
    gc.collect()

    # Re-open the merged archive and verify metadata plus bitwise array content.
    saved = np.load(output_path, allow_pickle=True)
    for item in saved:
        name = str(item["name"])
        row = next(r for r in rows if r["name"] == name)
        if int(item["patient_id"]) != row["patient_id"] or int(item["class_id"]) != row["class_id"]:
            raise AssertionError(f"Saved metadata changed for {name}.")
        if str(item["source_file"]) != row["source_file"]:
            raise AssertionError(f"Saved source filename changed for {name}.")
        if tuple(item["data"].shape) != row["raw_shape"]:
            raise AssertionError(f"Saved dimensions changed for {name}.")
        if _array_sha256(item["data"]) != hashes[name]:
            raise AssertionError(f"Raw pixel values changed during merge for {name}.")
    manifest = pd.DataFrame([
        {key: value for key, value in row.items() if key != "source_path"}
        | {"sha256_raw_array": hashes[row["name"]]}
        for row in rows
    ])
    manifest.to_csv(output_path.with_name("raw_merged_manifest.csv"), index=False)
    del saved
    gc.collect()
    return manifest


def audit_merged_dataset(
    raw_root: str | Path,
    preprocessed_path: str | Path,
    merged_path: str | Path,
) -> pd.DataFrame:
    """Recheck every merged record against its raw file and current entry."""
    rows = _alignment_rows(Path(raw_root), Path(preprocessed_path))
    merged = np.load(merged_path, allow_pickle=True)
    by_name = {str(item["name"]): item for item in merged}
    if len(by_name) != len(rows):
        raise ValueError(f"Merged archive has {len(by_name)} unique names; expected {len(rows)}.")
    results = []
    for row in rows:
        item = by_name.get(row["name"])
        if item is None:
            raise ValueError(f"Merged archive is missing {row['name']}.")
        source = np.load(row["source_path"], mmap_mode="r", allow_pickle=False)
        matched = (
            int(item["patient_id"]) == row["patient_id"]
            and int(item["class_id"]) == row["class_id"]
            and str(item["source_file"]) == row["source_file"]
            and tuple(item["data"].shape) == row["raw_shape"]
            and _array_sha256(item["data"]) == _array_sha256(source)
        )
        if not matched:
            raise AssertionError(f"Merged archive does not match source for {row['name']}.")
        results.append({
            "name": row["name"],
            "patient_id": row["patient_id"],
            "class_id": row["class_id"],
            "raw_shape": row["raw_shape"],
            "processed_shape": row["processed_shape"],
            "array_content_exact": True,
        })
        del source
    del merged
    gc.collect()
    return pd.DataFrame(results)


def _airpls_baseline(spectrum: np.ndarray, lam: float = 4.0, porder: int = 1, itermax: int = 15) -> np.ndarray:
    """airPLS asymmetric baseline, using the APOLLO paper's lambda=4/order=1."""
    y = np.asarray(spectrum, dtype=np.float64)
    n = len(y)
    if porder < 1 or porder >= n:
        raise ValueError("porder must be between 1 and spectrum length - 1.")
    diagonals = [np.ones(n), -np.ones(n)]
    difference = sparse.diags(diagonals, [0, 1], shape=(n - 1, n), format="csc")
    if porder > 1:
        difference = difference
        for _ in range(1, porder):
            difference = difference[1:, 1:] - difference[:-1, 1:]
    penalty = float(lam) * (difference.T @ difference)
    weights = np.ones(n, dtype=np.float64)
    baseline = y.copy()
    for iteration in range(1, itermax + 1):
        baseline = spsolve(sparse.diags(weights, 0, format="csc") + penalty, weights * y)
        residual = y - baseline
        negative = residual < 0
        negative_sum = float(np.abs(residual[negative]).sum())
        if negative_sum < 0.001 * float(np.abs(y).sum()) or not negative.any():
            break
        weights[~negative] = 0.0
        weights[negative] = np.exp(iteration * np.abs(residual[negative]) / max(negative_sum, 1e-12))
        endpoint_weight = np.exp(iteration * np.max(residual[negative]) / max(negative_sum, 1e-12))
        weights[0] = endpoint_weight
        weights[-1] = endpoint_weight
    return baseline


def apollo_preprocess_spectrum(raw_spectrum: np.ndarray, lam: float = 4.0) -> np.ndarray:
    """Crop, airPLS-correct, clip, concatenate, and L2-normalize one raw spectrum."""
    raw_spectrum = np.asarray(raw_spectrum, dtype=np.float64)
    if raw_spectrum.shape != (RAW_CHANNELS,):
        raise ValueError(f"Expected {RAW_CHANNELS} raw channels, got {raw_spectrum.shape}.")
    segments = []
    for start, stop in APOLLO_RETAINED_SLICES:
        retained = raw_spectrum[start:stop]
        corrected = retained - _airpls_baseline(retained, lam=lam, porder=1)
        segments.append(np.maximum(corrected, 0.0))
    result = np.concatenate(segments)
    if result.shape[0] != PROCESSED_CHANNELS:
        raise AssertionError(f"APOLLO channel crop produced {len(result)} rather than {PROCESSED_CHANNELS}.")
    norm_value = float(np.linalg.norm(result))
    if norm_value > 0:
        result /= norm_value
    return result


def verify_apollo_preprocessing(
    raw_merged_path: str | Path,
    preprocessed_path: str | Path,
    samples_per_entry: int = 5,
) -> pd.DataFrame:
    """Compare sampled raw pixels processed by APOLLO against the current file."""
    raw = np.load(raw_merged_path, allow_pickle=True)
    processed = np.load(preprocessed_path, allow_pickle=True)
    processed_by_name = {str(item["name"]): item for item in processed}
    results = []
    for item in raw:
        name = str(item["name"])
        current = processed_by_name[name]
        raw_flat = np.asarray(item["data"]).reshape(-1, RAW_CHANNELS)
        current_flat = np.asarray(current["data"]).reshape(-1, PROCESSED_CHANNELS)
        sample_indices = np.unique(np.linspace(0, len(raw_flat) - 1, min(samples_per_entry, len(raw_flat)), dtype=int))
        errors, correlations = [], []
        for index in sample_indices:
            result = apollo_preprocess_spectrum(raw_flat[index])
            target = current_flat[index]
            errors.append(float(np.max(np.abs(result - target))))
            denominator = np.linalg.norm(result) * np.linalg.norm(target)
            correlations.append(float(np.dot(result, target) / denominator) if denominator else 1.0)
        results.append({
            "name": name,
            "patient_id": int(item["patient_id"]),
            "class_id": int(item["class_id"]),
            "sampled_pixels": len(sample_indices),
            "max_absolute_error": max(errors),
            "median_cosine_similarity": float(np.median(correlations)),
        })
    del raw, processed
    gc.collect()
    return pd.DataFrame(results)


def _normalize_rows(values: np.ndarray, center: bool = False, scale: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if center:
        values = values - values.mean(axis=1, keepdims=True)
    if scale:
        row_sd = values.std(axis=1, keepdims=True)
        values = values / np.maximum(row_sd, 1e-12)
    row_norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(row_norm, 1e-12)


def build_preprocessing_features(
    raw_merged_path: str | Path,
    preprocessed_path: str | Path,
    chunk_size: int = 4096,
) -> dict:
    """Aggregate APOLLO and candidate spectrum-wise preprocessing to patients."""
    raw_entries = np.load(raw_merged_path, allow_pickle=True)
    processed_entries = np.load(preprocessed_path, allow_pickle=True)
    raw_by_name = {str(item["name"]): item for item in raw_entries}
    patient_ids = sorted({int(item["patient_id"]) for item in processed_entries})
    targets = {}
    variants = (
        "APOLLO current",
        "Raw crop + clip + L2 (no airPLS)",
        "APOLLO + pixel SNV",
        "APOLLO + Savitzky-Golay smooth",
        "APOLLO + Savitzky-Golay first derivative",
    )
    sums = {variant: {pid: np.zeros(PROCESSED_CHANNELS, dtype=np.float64) for pid in patient_ids} for variant in variants}
    counts = {pid: 0 for pid in patient_ids}

    for processed_item in processed_entries:
        name = str(processed_item["name"])
        patient_id = int(processed_item["patient_id"])
        target = int(processed_item["class_id"] >= 4)
        if patient_id in targets and targets[patient_id] != target:
            raise ValueError(f"Patient {patient_id} has inconsistent IDH targets.")
        targets[patient_id] = target
        processed_flat = np.asarray(processed_item["data"]).reshape(-1, PROCESSED_CHANNELS)
        raw_flat = np.asarray(raw_by_name[name]["data"]).reshape(-1, RAW_CHANNELS)
        if len(processed_flat) != len(raw_flat):
            raise ValueError(f"Pixel counts differ for {name}.")

        for start in range(0, len(processed_flat), chunk_size):
            stop = min(start + chunk_size, len(processed_flat))
            apollo = np.asarray(processed_flat[start:stop], dtype=np.float64)
            raw_chunk = np.asarray(raw_flat[start:stop], dtype=np.float64)
            raw_crop = np.concatenate(
                [raw_chunk[:, left:right] for left, right in APOLLO_RETAINED_SLICES], axis=1
            )
            raw_crop = _normalize_rows(np.maximum(raw_crop, 0.0))
            snv = _normalize_rows(apollo, center=True, scale=True)
            smooth = savgol_filter(apollo, window_length=11, polyorder=3, deriv=0, axis=1, mode="interp")
            smooth = _normalize_rows(smooth)
            derivative = savgol_filter(apollo, window_length=11, polyorder=3, deriv=1, axis=1, mode="interp")
            derivative = _normalize_rows(derivative)

            sums["APOLLO current"][patient_id] += apollo.sum(axis=0)
            sums["Raw crop + clip + L2 (no airPLS)"][patient_id] += raw_crop.sum(axis=0)
            sums["APOLLO + pixel SNV"][patient_id] += snv.sum(axis=0)
            sums["APOLLO + Savitzky-Golay smooth"][patient_id] += smooth.sum(axis=0)
            sums["APOLLO + Savitzky-Golay first derivative"][patient_id] += derivative.sum(axis=0)
            counts[patient_id] += len(apollo)

    X_by_variant = {
        variant: np.vstack([sums[variant][pid] / counts[pid] for pid in patient_ids])
        for variant in variants
    }
    y = np.asarray([targets[pid] for pid in patient_ids], dtype=int)
    del raw_entries, processed_entries, raw_by_name, sums
    gc.collect()
    return {"patient_ids": np.asarray(patient_ids), "y": y, "X_by_variant": X_by_variant, "pixel_counts": counts}


def evaluate_preprocessing_gpr(
    features: dict,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    n_components: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed PCA-8 ARD GPR under repeated patient-level folds."""
    patient_ids, y = features["patient_ids"], features["y"]
    seed_rows = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for variant, X in features["X_by_variant"].items():
            out_of_fold_mean = np.zeros(len(y), dtype=float)
            out_of_fold_std = np.zeros(len(y), dtype=float)
            for fold, (train, valid) in enumerate(splitter.split(X, y), start=1):
                # Match notebook 03's PCA solver selection and repeat its seed
                # within a CV run so the seed-42 APOLLO baseline is comparable.
                pca = PCA(n_components=n_components, random_state=seed).fit(X[train])
                Z_train, Z_valid = pca.transform(X[train]), pca.transform(X[valid])
                kernel = (
                    ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF(np.ones(n_components))
                    + WhiteKernel(0.1, noise_level_bounds=(1e-5, 1e2))
                )
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(Z_train, y[train])
                mean, std = gp.predict(Z_valid, return_std=True)
                out_of_fold_mean[valid] = mean
                out_of_fold_std[valid] = std

            # Pool one held-out prediction per patient before calculating
            # nonlinear scores such as AUC, matching notebook 03's protocol.
            out_of_fold_std = np.maximum(out_of_fold_std, 1e-8)
            low = out_of_fold_mean + norm.ppf(0.05) * out_of_fold_std
            high = out_of_fold_mean + norm.ppf(0.95) * out_of_fold_std
            nlpd = np.mean(
                0.5 * np.log(2 * np.pi * out_of_fold_std**2)
                + 0.5 * ((y - out_of_fold_mean) / out_of_fold_std) ** 2
            )
            seed_rows.append({
                "seed": seed,
                "preprocessing": variant,
                "rmse": float(np.sqrt(mean_squared_error(y, out_of_fold_mean))),
                "mae": float(mean_absolute_error(y, out_of_fold_mean)),
                "normal_nlpd": float(nlpd),
                "coverage_90": float(np.mean((y >= low) & (y <= high))),
                "width_90": float(np.mean(high - low)),
                "accuracy_at_0_5": float(accuracy_score(y, out_of_fold_mean >= 0.5)),
                "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, out_of_fold_mean >= 0.5)),
                "roc_auc": float(roc_auc_score(y, out_of_fold_mean)),
            })
    by_seed = pd.DataFrame(seed_rows)
    metric_columns = [
        "rmse", "mae", "normal_nlpd", "coverage_90", "width_90",
        "accuracy_at_0_5", "balanced_accuracy_at_0_5", "roc_auc",
    ]
    summary = by_seed.groupby("preprocessing")[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index(), by_seed
