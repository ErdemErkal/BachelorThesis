# ruff: noqa: E402
import argparse
import json
import os
import sys
import tempfile
import warnings
from copy import deepcopy
from types import SimpleNamespace

sys.dont_write_bytecode = True
SURVIVAL_OUTPUT_DIR = os.path.join("results", "survival_outputs")
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLED"] = "true"


class QuietTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        try:
            self._finalizer.detach()
        except Exception:
            pass


tempfile.TemporaryDirectory = QuietTemporaryDirectory

import numpy as np
import pandas as pd
from icp import ConformalSurvDist
from icp.scorer import QuantileRegressionNC
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from SurvivalEVAL.Evaluations.util import check_monotonicity
from utils.util_survival import (
    format_pred_sksurv,
    make_mono_quantiles,
    survival_data_split,
    survival_to_quantile,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="veteran",
        help="Dataset name in results/datasets, or a CSV path.",
    )
    parser.add_argument(
        "--test-data", default=None, help="Optional shifted test data path."
    )
    parser.add_argument("--splits", default=None, help="Optional splits JSON path.")
    parser.add_argument(
        "--split-id", type=int, default=0, help="Index into splits['train']."
    )
    parser.add_argument(
        "--ess", default="1.0", help="ESS value for the shifted test filename."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frac-train", type=float, default=0.6)
    parser.add_argument("--model", choices=["AFT", "CGSA", "CoxPH"], default="AFT")
    parser.add_argument(
        "--density-mode",
        choices=["estimated", "oracle"],
        default="estimated",
        help="How to compute source->target density weights for weighted CSD.",
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=9
    )
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument(
        "--oracle-clip-percentile",
        type=float,
        default=1.0,
        help=(
            "Percentile at which to cap oracle density weights "
            "(e.g., 0.95). Set to 1.0 for no clipping."
        ),
    )
    parser.add_argument(
        "--decensor-method",
        default="sampling",
        choices=["uncensored", "margin", "PO", "sampling"],
    )
    parser.add_argument("--penalizer", type=float, default=0.01)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz filename under results/survival_outputs, or a full path.",
    )
    args = parser.parse_args()
    if not 0 < args.frac_train < 1:
        parser.error("--frac-train must be between 0 and 1.")
    if not 0 < args.oracle_clip_percentile <= 1.0:
        parser.error("--oracle-clip-percentile must be in the range (0, 1].")
    return args


def survival_output_path(filename_or_path):
    if not filename_or_path.lower().endswith(".npz"):
        filename_or_path = f"{filename_or_path}.npz"
    if os.path.isabs(filename_or_path) or os.path.dirname(filename_or_path):
        return filename_or_path
    return os.path.join(SURVIVAL_OUTPUT_DIR, filename_or_path)


def resolve_paths(args):
    if os.path.exists(args.dataset):
        dataset_path = args.dataset
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    else:
        dataset_name = args.dataset
        dataset_path = f"results/datasets/{dataset_name}.csv"

    test_path = (
        args.test_data
        or f"results/create_distribution_shift/{dataset_name}_split_{args.split_id}_ess_{args.ess}.tsv"
    )
    splits_path = args.splits or f"results/create_splits/{dataset_name}.json"
    output_name = (
        args.output
        or f"{dataset_name}_split_{args.split_id}_ess_{args.ess}_seed_{args.seed}_{args.model}.npz"
    )
    output_path = survival_output_path(output_name)
    return dataset_name, dataset_path, test_path, splits_path, output_path


def add_survival_columns(features, source):
    frame = features.copy()
    frame["time"] = source["time"].astype(float).values
    frame["event"] = source["event"].astype(int).values
    return frame


def make_model_frames(data_train, data_val, data_test):
    # Fit preprocessing on the training split, then reuse it for calibration and test.
    feature_cols = [col for col in data_train.columns if col not in ["time", "event"]]
    x_train_raw = data_train[feature_cols].copy()
    x_val_raw = data_val[feature_cols].copy()
    x_test_raw = data_test[feature_cols].copy()

    num_cols = []
    cat_cols = []
    for col in feature_cols:
        converted = [
            pd.to_numeric(frame[col], errors="coerce")
            for frame in [x_train_raw, x_val_raw, x_test_raw]
        ]
        original = [frame[col] for frame in [x_train_raw, x_val_raw, x_test_raw]]
        numeric_like = all(
            (src.isna() | conv.notna()).all() for src, conv in zip(original, converted)
        )
        if numeric_like:
            num_cols.append(col)
            x_train_raw[col], x_val_raw[col], x_test_raw[col] = converted
        else:
            cat_cols.append(col)

    frames = []
    if num_cols:
        train_num = x_train_raw[num_cols].astype(float)
        val_num = x_val_raw[num_cols].astype(float)
        test_num = x_test_raw[num_cols].astype(float)

        medians = train_num.median()
        train_num = train_num.fillna(medians)
        val_num = val_num.fillna(medians)
        test_num = test_num.fillna(medians)

        scaler = StandardScaler()
        train_num = pd.DataFrame(scaler.fit_transform(train_num), columns=num_cols)
        val_num = pd.DataFrame(scaler.transform(val_num), columns=num_cols)
        test_num = pd.DataFrame(scaler.transform(test_num), columns=num_cols)
        frames.append((train_num, val_num, test_num))

    if cat_cols:
        # Keep dummy columns aligned when validation/test lacks a training category.
        train_cat_raw = x_train_raw[cat_cols].fillna("missing").astype(str)
        val_cat_raw = x_val_raw[cat_cols].fillna("missing").astype(str)
        test_cat_raw = x_test_raw[cat_cols].fillna("missing").astype(str)

        train_cat = pd.get_dummies(train_cat_raw, drop_first=True, dtype=float)
        val_cat = pd.get_dummies(val_cat_raw, drop_first=True, dtype=float).reindex(
            columns=train_cat.columns, fill_value=0
        )
        test_cat = pd.get_dummies(test_cat_raw, drop_first=True, dtype=float).reindex(
            columns=train_cat.columns, fill_value=0
        )
        frames.append((train_cat, val_cat, test_cat))

    if not frames:
        raise ValueError("No feature columns found.")

    x_train = pd.concat([part[0].reset_index(drop=True) for part in frames], axis=1)
    x_val = pd.concat([part[1].reset_index(drop=True) for part in frames], axis=1)
    x_test = pd.concat([part[2].reset_index(drop=True) for part in frames], axis=1)
    return (
        add_survival_columns(x_train, data_train),
        add_survival_columns(x_val, data_val),
        add_survival_columns(x_test, data_test),
    )


def add_time_zero(surv, time_coordinates):
    if len(time_coordinates) > 0 and time_coordinates[0] == 0:
        return surv, time_coordinates
    return (
        np.concatenate([np.ones((surv.shape[0], 1)), surv], axis=1),
        np.concatenate([np.array([0.0]), time_coordinates], axis=0),
    )


def make_structured_survival_target(time_values, event_values):
    y = np.zeros(time_values.shape[0], dtype=[("event", bool), ("time", float)])
    y["event"] = event_values.astype(bool)
    y["time"] = time_values.astype(float)
    return y


def fit_density_ratio_model(
    source_features, target_features, seed, target_sample_weights=None
):
    source_x = source_features.values
    target_x = target_features.values
    x_weight = np.vstack([source_x, target_x])
    y_weight = np.concatenate([np.zeros(source_x.shape[0]), np.ones(target_x.shape[0])])

    weight_model = LogisticRegression(max_iter=1000, random_state=seed)
    source_sample_weights = np.ones(source_x.shape[0], dtype=float)
    if target_sample_weights is None:
        target_sample_weights = np.ones(target_x.shape[0], dtype=float)
    else:
        target_sample_weights = np.asarray(target_sample_weights, dtype=float)
        if target_sample_weights.ndim != 1:
            raise ValueError("target_sample_weights must be one-dimensional.")
        if target_sample_weights.shape[0] != target_x.shape[0]:
            raise ValueError(
                "target_sample_weights length must match target feature rows."
            )
        if np.any(~np.isfinite(target_sample_weights)) or np.any(
            target_sample_weights < 0
        ):
            raise ValueError(
                "target_sample_weights must contain finite non-negative values."
            )
        target_weight_mean = float(np.mean(target_sample_weights))
        if not np.isfinite(target_weight_mean) or target_weight_mean <= 0:
            raise ValueError(
                "target_sample_weights must have a positive finite mean."
            )
        # Keep class mass comparable while preserving relative target weighting.
        target_sample_weights = target_sample_weights / target_weight_mean

    sample_weights = np.concatenate([source_sample_weights, target_sample_weights])
    weight_model.fit(x_weight, y_weight, sample_weight=sample_weights)

    source_mass = float(np.sum(source_sample_weights))
    target_mass = float(np.sum(target_sample_weights))
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Target class mass must be positive and finite.")
    source_to_target_prior_ratio = source_mass / target_mass
    return weight_model, source_to_target_prior_ratio


def weighted_conformal_cutoffs(
    calibration_scores, calibration_weights, test_weights, quantile_levels
):
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    calibration_weights = np.asarray(calibration_weights, dtype=float)
    test_weights = np.asarray(test_weights, dtype=float)
    quantile_levels = np.asarray(quantile_levels, dtype=float)

    if calibration_scores.ndim == 1:
        calibration_scores = calibration_scores[:, np.newaxis]
    if calibration_scores.ndim != 2:
        raise ValueError("Calibration scores must be one- or two-dimensional.")
    if calibration_weights.ndim != 1:
        raise ValueError("Calibration weights must be one-dimensional.")
    if calibration_weights.shape[0] != calibration_scores.shape[0]:
        raise ValueError("Calibration scores and weights must have matching rows.")
    if calibration_scores.shape[1] not in [1, quantile_levels.shape[0]]:
        raise ValueError("Calibration score columns must match quantile levels.")

    cal_weight_sum = calibration_weights.sum()
    if not np.isfinite(cal_weight_sum) or cal_weight_sum <= 0:
        raise ValueError("Calibration weights must have positive finite sum.")

    cutoffs = np.empty((test_weights.shape[0], quantile_levels.shape[0]))
    for j, level in enumerate(quantile_levels):
        score_col = calibration_scores[:, 0]
        if calibration_scores.shape[1] > 1:
            score_col = calibration_scores[:, j]

        order = np.argsort(score_col)
        sorted_scores = score_col[order]
        sorted_weight_cdf = np.cumsum(calibration_weights[order])

        threshold = (1 - level) * (cal_weight_sum + test_weights)
        idx = np.searchsorted(sorted_weight_cdf, threshold, side="left")
        out_of_bounds = idx == sorted_scores.shape[0]
        idx_safe = np.clip(idx, 0, sorted_scores.shape[0] - 1)

        cutoff_j = sorted_scores[idx_safe]
        cutoff_j[out_of_bounds] = np.inf
        cutoffs[:, j] = cutoff_j

    return cutoffs


def main():
    args_cli = parse_args()
    dataset_name, dataset_path, test_path, splits_path, output_path = resolve_paths(
        args_cli
    )

    class PreFitQuantileRegressionNC(QuantileRegressionNC):
        def __init__(
            self,
            model,
            args=argparse.Namespace,
            weight_model=None,
            source_to_target_prior_ratio=1.0,
            probability_clip=1e-6,
            density_mode="estimated",
            oracle_test_weights=None,
            oracle_cal_weights=None,
        ):
            super().__init__(model, args)
            self.weight_model = weight_model
            self.source_to_target_prior_ratio = source_to_target_prior_ratio
            self.probability_clip = probability_clip
            self.density_mode = density_mode
            self.oracle_test_weights = (
                None
                if oracle_test_weights is None
                else np.asarray(oracle_test_weights, dtype=float)
            )
            self.oracle_cal_weights = (
                None
                if oracle_cal_weights is None
                else np.asarray(oracle_cal_weights, dtype=float)
            )
            self.calibration_scores = None
            self.calibration_weights = None
            self.calibration_density_features = None
            self.test_density_features = None

        def fit(self, train_set, val_set):
            return self

        def predict_nc(
            self,
            x: np.ndarray,
            quantile_levels: np.ndarray,
            feature_names: list[str] = None,
        ) -> np.ndarray:
            if not isinstance(
                self.model, (CoxPHSurvivalAnalysis, GradientBoostingSurvivalAnalysis)
            ):
                return super().predict_nc(x, quantile_levels, feature_names)

            batch_size = 16384
            num_batches = x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)
            quantile_batches = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, x.shape[0])
                x_batch = x[start_idx:end_idx]
                pred_surv = self.model.predict_survival_function(x_batch)
                surv_prob, time_coordinates = format_pred_sksurv(pred_surv)
                if time_coordinates[0] != 0:
                    time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
                    surv_prob = np.concatenate(
                        [np.ones([surv_prob.shape[0], 1]), surv_prob], axis=1
                    )
                time_coordinates = np.repeat(
                    time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0
                )
                quantile_batch = survival_to_quantile(
                    surv_prob, time_coordinates, quantile_levels, self.args.interpolate
                )
                quantile_batches.append(quantile_batch)
            return np.vstack(quantile_batches)

        def set_density_feature_context(
            self, calibration_features: np.ndarray | None, test_features: np.ndarray | None
        ):
            self.calibration_density_features = (
                None
                if calibration_features is None
                else np.asarray(calibration_features, dtype=float)
            )
            self.test_density_features = (
                None
                if test_features is None
                else np.asarray(test_features, dtype=float)
            )

        def density_ratio(self, x, is_test=False):
            if self.density_mode == "oracle":
                if is_test:
                    if self.oracle_test_weights is None:
                        raise ValueError(
                            "Oracle density mode requires oracle test weights."
                        )
                    if self.oracle_test_weights.shape[0] != x.shape[0]:
                        raise ValueError(
                            "Oracle test weights length does not match test samples."
                        )
                    return self.oracle_test_weights
                if self.oracle_cal_weights is None:
                    raise ValueError(
                        "Oracle density mode requires oracle calibration weights."
                    )
                if self.oracle_cal_weights.shape[0] != x.shape[0]:
                    raise ValueError(
                        "Oracle calibration weights length does not match samples."
                    )
                return self.oracle_cal_weights
            if self.weight_model is None:
                return np.ones(x.shape[0], dtype=float)
            target_prob = self.weight_model.predict_proba(x)[:, 1]
            target_prob = np.clip(
                target_prob, self.probability_clip, 1 - self.probability_clip
            )
            odds = target_prob / (1 - target_prob)
            weights = odds * self.source_to_target_prior_ratio
            return np.clip(weights, 0.0, 50.0)

        def score(
            self,
            feature_df: pd.DataFrame,
            t: np.ndarray,
            e: np.ndarray,
            quantile_levels: np.ndarray,
            n_sample: int = 1000,
        ):
            x = feature_df.values
            x_names = feature_df.columns.tolist()
            y = np.stack([t, e], axis=1)

            quantile_predictions = self.predict_nc(x, quantile_levels, x_names)
            density_features = x
            if (
                self.density_mode == "estimated"
                and self.calibration_density_features is not None
            ):
                if self.calibration_density_features.shape[0] != x.shape[0]:
                    raise ValueError(
                        "Calibration density features row count does not match "
                        "calibration samples."
                    )
                density_features = self.calibration_density_features
            weights = self.density_ratio(density_features, is_test=False)

            if n_sample is not None:
                quantile_predictions = np.repeat(quantile_predictions, n_sample, axis=0)
                weights = np.repeat(weights, n_sample) / n_sample

            if quantile_predictions.shape[0] != y.shape[0]:
                raise ValueError("Sample size does not match.")

            self.calibration_scores = self.err_func.apply(quantile_predictions, y)
            self.calibration_weights = weights.astype(float)
            return self.calibration_scores

        def weighted_error_distribution(self, x, quantile_levels):
            if self.calibration_scores is None or self.calibration_weights is None:
                raise ValueError("Weighted CSD prediction requires calibration first.")

            density_features = x
            if self.density_mode == "estimated" and self.test_density_features is not None:
                if self.test_density_features.shape[0] != x.shape[0]:
                    raise ValueError(
                        "Test density features row count does not match test samples."
                    )
                density_features = self.test_density_features
            test_weights = self.density_ratio(density_features, is_test=True)
            return weighted_conformal_cutoffs(
                self.calibration_scores,
                self.calibration_weights,
                test_weights,
                quantile_levels,
            )

        def predict(
            self,
            x: np.ndarray,
            conformal_scores: np.ndarray,
            feature_names: list[str] = None,
            quantile_levels=None,
        ):
            quantile_predictions = self.predict_nc(x, quantile_levels, feature_names)
            error_dist = self.weighted_error_distribution(x, quantile_levels)
            quantile_predictions = quantile_predictions - error_dist
            quantile_levels, quantile_predictions = make_mono_quantiles(
                quantile_levels,
                quantile_predictions,
                method=self.args.mono_method,
                seed=self.args.seed,
            )
            assert np.all(
                quantile_predictions >= 0
            ), "Quantile predictions contain negative."
            assert check_monotonicity(
                quantile_predictions
            ), "Quantile predictions are not monotonic."
            return quantile_predictions

    np.random.seed(args_cli.seed)

    def summarize_weights(label: str, weights: np.ndarray) -> None:
        weights = np.asarray(weights, dtype=float)
        percentiles = [50, 90, 95, 99]
        quantile_vals = np.percentile(weights, percentiles)
        summary = {
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            **{
                f"p{percentile}": float(value)
                for percentile, value in zip(percentiles, quantile_vals)
            },
        }
        summary_str = ", ".join(f"{k}={v:.6g}" for k, v in summary.items())
        print(f"Oracle weight stats ({label}): {summary_str}")

    def extract_tilt_feature_name(shift_tsv_path: str) -> str:
        try:
            tilt_meta = pd.read_csv(shift_tsv_path, sep="\t", usecols=["tilt_feature"])
        except ValueError as exc:
            raise ValueError(
                "Shift TSV must include a 'tilt_feature' column."
            ) from exc
        tilt_values = (
            tilt_meta["tilt_feature"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
        )
        if tilt_values.shape[0] != 1:
            raise ValueError(
                "Expected exactly one non-empty tilt_feature value in shift TSV."
            )
        return tilt_values[0]

    def is_natural_shift(tilt_name: str) -> bool:
        return str(tilt_name).strip().lower() in {"none", "natural", "__none__"}

    # Load full dataset and shifted test rows.
    data = pd.read_csv(dataset_path).rename(columns={"status": "event"})
    splits = json.loads(open(splits_path).read())
    shifted_test = pd.read_csv(test_path, sep="\t").rename(columns={"status": "event"})
    # Split-bookkeeping only; must not enter survival / density models.
    data = data.drop(columns=["source_dataset"], errors="ignore")
    shifted_test = shifted_test.drop(columns=["source_dataset"], errors="ignore")
    if "dataset_index" not in shifted_test.columns:
        raise ValueError("Shift TSV must include a 'dataset_index' column.")
    raw_dataset_index = shifted_test["dataset_index"].to_numpy()
    if np.any(pd.isna(raw_dataset_index)):
        raise ValueError("Shift TSV contains missing dataset_index values.")
    test_dataset_index = raw_dataset_index.astype(int)
    if np.any(test_dataset_index < 0) or np.any(test_dataset_index >= len(data)):
        raise ValueError("Shift TSV dataset_index values are out of bounds.")

    tilt_feature_name = extract_tilt_feature_name(test_path)
    natural_shift = is_natural_shift(tilt_feature_name)
    if natural_shift and args_cli.density_mode == "oracle":
        raise ValueError(
            "Natural transfer shift has no oracle weights; use --density-mode estimated."
        )
    if not natural_shift:
        if tilt_feature_name not in data.columns:
            raise ValueError(
                f"tilt_feature '{tilt_feature_name}' is missing from dataset columns."
            )
        if tilt_feature_name in {"time", "event"}:
            raise ValueError("tilt_feature cannot be a survival target column.")

    data_test = shifted_test.drop(
        columns=["dataset_index", "tilt_feature", "true_density_weight"],
        errors="ignore",
    ).copy()
    if not natural_shift and tilt_feature_name not in data_test.columns:
        raise ValueError(
            f"tilt_feature '{tilt_feature_name}' is missing from split test frame."
        )

    oracle_eval_test_weights = np.ones(data_test.shape[0], dtype=float)
    oracle_weight_map = None
    if not natural_shift:
        if test_path.lower().endswith(".tsv"):
            oracle_weight_path = test_path[: -len(".tsv")] + "_weights.tsv"
        else:
            oracle_weight_path = f"{test_path}_weights.tsv"
        if not os.path.exists(oracle_weight_path):
            raise ValueError(
                f"Missing oracle weight file: {oracle_weight_path}. "
                "Run create_distribution_shift first."
            )
        oracle_weights_df = pd.read_csv(oracle_weight_path, sep="\t")
        required_cols = {"dataset_index", "oracle_weight"}
        if not required_cols.issubset(oracle_weights_df.columns):
            raise ValueError(
                "Oracle weight file must contain columns: "
                "'dataset_index' and 'oracle_weight'."
            )
        oracle_weight_map = (
            oracle_weights_df[["dataset_index", "oracle_weight"]]
            .drop_duplicates(subset=["dataset_index"], keep="last")
            .set_index("dataset_index")["oracle_weight"]
            .sort_index()
        )
        oracle_eval_test_weights = oracle_weight_map.reindex(test_dataset_index).to_numpy(
            dtype=float
        )
        if np.any(~np.isfinite(oracle_eval_test_weights)):
            raise ValueError(
                "Oracle weight map does not cover all shifted test dataset indices."
            )
        summarize_weights("oracle_eval_test", oracle_eval_test_weights)

    oracle_test_weights = None
    oracle_cal_weights = None
    oracle_weight_cap = None
    oracle_normalization_factor = None
    if args_cli.density_mode == "oracle":
        raw_test_weights = np.asarray(oracle_eval_test_weights, dtype=float)
        if raw_test_weights.ndim != 1:
            raise ValueError("Oracle test weights must be one-dimensional.")
        if np.any(~np.isfinite(raw_test_weights)):
            raise ValueError("Oracle test weights must be finite before clipping.")
        summarize_weights("raw_test", raw_test_weights)

        if args_cli.oracle_clip_percentile >= 1.0:
            oracle_test_weights = raw_test_weights
            oracle_normalization_factor = 1.0
            oracle_weight_cap = np.inf
        else:
            clip_percentile_val = args_cli.oracle_clip_percentile * 100.0
            oracle_weight_cap = np.percentile(raw_test_weights, clip_percentile_val)
            clipped_test_weights = np.clip(
                raw_test_weights, a_min=0.0, a_max=oracle_weight_cap
            )
            summarize_weights("clipped_test", clipped_test_weights)

            oracle_normalization_factor = float(np.mean(clipped_test_weights))
            if (
                not np.isfinite(oracle_normalization_factor)
                or oracle_normalization_factor <= 0
            ):
                raise ValueError(
                    "Oracle normalization factor must be positive and finite "
                    "(mean of clipped oracle test weights)."
                )
            oracle_test_weights = clipped_test_weights / oracle_normalization_factor
        if oracle_test_weights.ndim != 1:
            raise ValueError("Final oracle test weights must be one-dimensional.")
        if oracle_test_weights.shape[0] != data_test.shape[0]:
            raise ValueError("Final oracle test weights length mismatch.")
        summarize_weights("re_normalized_test", oracle_test_weights)

    if data_test.shape[0] < 20:
        raise ValueError(
            f"Shifted test set too small ({data_test.shape[0]} samples). "
            "Re-run with a larger base pool or milder shift."
        )
    if data_test.shape[0] < args_cli.n_quantiles * 10:
        warnings.warn(
            "Shifted test set has fewer than n_quantiles * 10 samples "
            f"({data_test.shape[0]} < {args_cli.n_quantiles * 10}); "
            "quantile-based calibration may be noisy.",
            RuntimeWarning,
        )

    train_ix = np.array(splits["train"][args_cli.split_id])
    train_split = data.iloc[train_ix, :]

    # Split the saved training subset into model-training and calibration parts.
    data_train, data_val, _ = survival_data_split(
        train_split,
        stratify_colname="both",
        frac_train=args_cli.frac_train,
        frac_val=1 - args_cli.frac_train,
        frac_test=0.0,
        random_state=args_cli.seed,
    )
    if not natural_shift and tilt_feature_name not in data_val.columns:
        raise ValueError(
            f"tilt_feature '{tilt_feature_name}' is missing from calibration frame."
        )
    if args_cli.density_mode == "oracle":
        raw_cal_weights = oracle_weight_map.reindex(data_val.index.values).to_numpy(
            dtype=float
        )
        if raw_cal_weights.ndim != 1:
            raise ValueError("Oracle calibration weights must be one-dimensional.")
        if np.any(~np.isfinite(raw_cal_weights)):
            raise ValueError(
                "Oracle weight map does not cover all calibration indices."
            )
        summarize_weights("raw_calibration", raw_cal_weights)
        if args_cli.oracle_clip_percentile >= 1.0:
            oracle_cal_weights = raw_cal_weights
        else:
            clipped_cal_weights = np.clip(
                raw_cal_weights, a_min=0.0, a_max=oracle_weight_cap
            )
            summarize_weights("clipped_calibration", clipped_cal_weights)
            oracle_cal_weights = clipped_cal_weights / oracle_normalization_factor
        if oracle_cal_weights.ndim != 1:
            raise ValueError("Final oracle calibration weights must be one-dimensional.")
        if oracle_cal_weights.shape[0] != data_val.shape[0]:
            raise ValueError("Final oracle calibration weights length mismatch.")
        summarize_weights("re_normalized_calibration", oracle_cal_weights)

    # Natural transfer: keep all covariates. Adversarial ESS: drop the tilted feature.
    if natural_shift:
        train_frame, val_frame, test_frame = make_model_frames(
            data_train, data_val, data_test
        )
    else:
        train_frame, val_frame, test_frame = make_model_frames(
            data_train.drop(columns=[tilt_feature_name]),
            data_val.drop(columns=[tilt_feature_name]),
            data_test.drop(columns=[tilt_feature_name]),
        )

    # Store event/time arrays needed later by SurvivalEVAL.
    t_train = train_frame["time"].values
    e_train = train_frame["event"].values.astype(int)
    t_val = val_frame["time"].values
    e_val = val_frame["event"].values.astype(int)
    t_test = test_frame["time"].values
    e_test = test_frame["event"].values.astype(int)
    t_train_ref = np.concatenate([t_train, t_val])
    e_train_ref = np.concatenate([e_train, e_val])
    x_test = test_frame.drop(columns=["time", "event"]).values

    # Fit baseline Weibull AFT and save its predicted survival curves.
    if args_cli.model == "AFT":
        model = WeibullAFTFitter(penalizer=args_cli.penalizer)
        model.fit(train_frame, duration_col="time", event_col="event")
        surv_df = model.predict_survival_function(
            test_frame.drop(columns=["time", "event"])
        )
        surv_test = surv_df.values.T
        time_coordinates = surv_df.index.values
    elif args_cli.model == "CGSA":
        x_train = train_frame.drop(columns=["time", "event"]).values
        y_train = make_structured_survival_target(
            train_frame["time"].values, train_frame["event"].values
        )
        model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=0.1,
            n_estimators=100,
            random_state=args_cli.seed,
        )
        model.fit(x_train, y_train)
        pred_surv = model.predict_survival_function(
            test_frame.drop(columns=["time", "event"]).values
        )
        surv_test, time_coordinates = format_pred_sksurv(pred_surv)
    elif args_cli.model == "CoxPH":
        x_train = train_frame.drop(columns=["time", "event"]).values
        y_train = make_structured_survival_target(
            train_frame["time"].values, train_frame["event"].values
        )
        model = CoxPHSurvivalAnalysis(alpha=args_cli.penalizer)
        model.fit(x_train, y_train)
        pred_surv = model.predict_survival_function(
            test_frame.drop(columns=["time", "event"]).values
        )
        surv_test, time_coordinates = format_pred_sksurv(pred_surv)
    else:
        raise ValueError(f"Unsupported model: {args_cli.model}")

    surv_test, time_coordinates = add_time_zero(surv_test, time_coordinates)

    # Reuse the fitted AFT model inside the conformal calibrators.
    scorer_args = SimpleNamespace(
        model=args_cli.model,
        n_quantiles=args_cli.n_quantiles,
        decensor_method=args_cli.decensor_method,
        n_sample=args_cli.n_sample,
        use_train=False,
        verbose=False,
        interpolate="Pchip",
        mono_method="ceil",
        seed=args_cli.seed,
    )

    weight_model = None
    source_to_target_prior_ratio = 1.0
    if args_cli.density_mode == "estimated":
        if natural_shift:
            # Natural Rot→GBSG: estimate density ratio on all model covariates.
            density_val = val_frame.drop(columns=["time", "event"])
            density_test = test_frame.drop(columns=["time", "event"])
        else:
            density_val = data_val[[tilt_feature_name]]
            density_test = data_test[[tilt_feature_name]]
        weight_model, source_to_target_prior_ratio = fit_density_ratio_model(
            density_val,
            density_test,
            args_cli.seed,
            target_sample_weights=None,
        )

    nc_csd = PreFitQuantileRegressionNC(
        deepcopy(model),
        scorer_args,
        weight_model=weight_model,
        source_to_target_prior_ratio=source_to_target_prior_ratio,
        density_mode=args_cli.density_mode,
        oracle_test_weights=oracle_test_weights,
        oracle_cal_weights=oracle_cal_weights,
    )
    if args_cli.density_mode == "estimated":
        nc_csd.set_density_feature_context(
            density_val.values,
            density_test.values,
        )
    else:
        nc_csd.set_density_feature_context(
            data_val[[tilt_feature_name]].values,
            data_test[[tilt_feature_name]].values,
        )
    icp_csd = ConformalSurvDist(
        nc_csd,
        condition=None,
        decensor_method=args_cli.decensor_method,
        n_quantiles=args_cli.n_quantiles,
    )
    icp_csd.fit(train_frame, val_frame)
    icp_csd.calibrate(val_frame)
    q_levels_csd, q_preds_csd = icp_csd.predict(x_test)

    # --- DIAGNOSTIC EXPORT (one row per test sample) ---
    mid_idx = args_cli.n_quantiles // 2
    test_features_df = test_frame.drop(columns=["time", "event"])
    diag_quantile_levels = np.linspace(
        1 / (args_cli.n_quantiles + 1),
        args_cli.n_quantiles / (args_cli.n_quantiles + 1),
        args_cli.n_quantiles,
    )
    test_q_preds = nc_csd.predict_nc(
        test_features_df.values,
        diag_quantile_levels,
        test_features_df.columns.tolist(),
    )
    test_y = np.stack([t_test, e_test], axis=1)
    test_scores = nc_csd.err_func.apply(test_q_preds, test_y)
    if test_scores.ndim != 2:
        raise ValueError("Expected 2D conformity scores for diagnostic export.")
    if mid_idx < 0 or mid_idx >= test_scores.shape[1]:
        raise ValueError("Median score index out of bounds for diagnostic export.")
    median_test_scores = test_scores[:, mid_idx]

    diag_df = pd.DataFrame(
        {
            "dataset_index": shifted_test["dataset_index"].to_numpy(dtype=int),
            "tilt_feature_name": tilt_feature_name,
            "tilt_feature_val": (
                np.full(data_test.shape[0], np.nan)
                if natural_shift
                else shifted_test[tilt_feature_name].to_numpy()
            ),
            "time": t_test,
            "event": e_test,
            "oracle_weight": oracle_eval_test_weights,
            "conformity_score": median_test_scores,
        }
    )
    if diag_df.shape[0] != t_test.shape[0] or diag_df.shape[0] != shifted_test.shape[0]:
        raise ValueError("Diagnostic TSV must contain exactly one row per test sample.")

    diag_dir = os.path.join("results", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    diag_filename = (
        f"{dataset_name}_split_{args_cli.split_id}_ess_{args_cli.ess}_"
        f"seed_{args_cli.seed}_{args_cli.model}_weighted_diagnostics.tsv"
    )
    diag_path = os.path.join(diag_dir, diag_filename)
    diag_df.to_csv(diag_path, sep="\t", index=False)
    print(f"Saved diagnostics TSV to {diag_path}")
    # -------------------------

    # Save predictions and labels; evaluation is handled by single_split_eval.py.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        baseline_survival=surv_test,
        baseline_time_coordinates=time_coordinates,
        csd_q_levels=q_levels_csd,
        csd_q_preds=q_preds_csd,
        x_test=x_test,
        t_test=t_test,
        e_test=e_test,
        oracle_test_weights=oracle_eval_test_weights,
        t_train_ref=t_train_ref,
        e_train_ref=e_train_ref,
        baseline_method=np.array(f"Baseline {args_cli.model}"),
        csd_method=np.array(f"{args_cli.model} + weighted CSD"),
        metadata=np.array(json.dumps({**vars(args_cli), "dataset_name": dataset_name})),
    )
    print(f"Saved survival outputs to {output_path}")


if __name__ == "__main__":
    main()
