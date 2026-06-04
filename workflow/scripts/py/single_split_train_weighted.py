# ruff: noqa: E402
import argparse
import json
import os
import sys
import tempfile
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
from icp import CSDiPOT, ConformalSurvDist
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from SurvivalEVAL.Evaluations.util import check_monotonicity
from utils.util_survival import make_mono_quantiles, survival_data_split


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
    parser.add_argument("--model", choices=["AFT"], default="AFT")
    parser.add_argument("--n-quantiles", type=int, default=9)
    parser.add_argument("--n-sample", type=int, default=1000)
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


def fit_density_ratio_model(source_features, target_features, seed):
    source_x = source_features.values
    target_x = target_features.values
    x_weight = np.vstack([source_x, target_x])
    y_weight = np.concatenate(
        [np.zeros(source_x.shape[0]), np.ones(target_x.shape[0])]
    )

    weight_model = LogisticRegression(max_iter=1000, random_state=seed)
    weight_model.fit(x_weight, y_weight)
    source_to_target_prior_ratio = source_x.shape[0] / target_x.shape[0]
    return weight_model, source_to_target_prior_ratio


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
        ):
            super().__init__(model, args)
            self.weight_model = weight_model
            self.source_to_target_prior_ratio = source_to_target_prior_ratio
            self.probability_clip = probability_clip
            self.calibration_scores = None
            self.calibration_weights = None

        def fit(self, train_set, val_set):
            return self

        def density_ratio(self, x):
            if self.weight_model is None:
                return np.ones(x.shape[0])
            target_prob = self.weight_model.predict_proba(x)[:, 1]
            target_prob = np.clip(
                target_prob, self.probability_clip, 1 - self.probability_clip
            )
            odds = target_prob / (1 - target_prob)
            return odds * self.source_to_target_prior_ratio

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
            weights = self.density_ratio(x)

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

            test_weights = self.density_ratio(x)
            cal_weights = self.calibration_weights
            cal_weight_sum = cal_weights.sum()
            if not np.isfinite(cal_weight_sum) or cal_weight_sum <= 0:
                raise ValueError("Calibration weights must have positive finite sum.")

            quantile_levels = np.asarray(quantile_levels)
            errors = np.empty((x.shape[0], quantile_levels.shape[0]))
            for j, level in enumerate(quantile_levels):
                order = np.argsort(self.calibration_scores[:, j])
                sorted_scores = self.calibration_scores[order, j]
                sorted_weight_cdf = np.cumsum(cal_weights[order])

                threshold = (1 - level) * (cal_weight_sum + test_weights)
                idx = np.searchsorted(sorted_weight_cdf, threshold, side="left")

                out_of_bounds = idx == sorted_scores.shape[0]

                idx_safe = np.clip(idx, 0, sorted_scores.shape[0] - 1)

                error_j = sorted_scores[idx_safe]
                error_j[out_of_bounds] = np.inf
                errors[:, j] = error_j
            return errors

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
            assert np.all(quantile_predictions >= 0), (
                "Quantile predictions contain negative."
            )
            assert check_monotonicity(quantile_predictions), (
                "Quantile predictions are not monotonic."
            )
            return quantile_predictions

    class PreFitSurvivalPredictionNC(SurvivalPredictionNC):
        def fit(self, train_set, val_set):
            return self

    np.random.seed(args_cli.seed)

    # Load the original dataset, shifted test split, and saved train indices.
    data = pd.read_csv(dataset_path).rename(columns={"status": "event"})
    data_test = pd.read_csv(test_path, sep="\t").rename(columns={"status": "event"})
    splits = json.loads(open(splits_path).read())

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

    train_frame, val_frame, test_frame = make_model_frames(
        data_train, data_val, data_test
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

    weight_model, source_to_target_prior_ratio = fit_density_ratio_model(
        val_frame.drop(columns=["time", "event"]),
        test_frame.drop(columns=["time", "event"]),
        args_cli.seed,
    )

    nc_csd = PreFitQuantileRegressionNC(
        deepcopy(model),
        scorer_args,
        weight_model=weight_model,
        source_to_target_prior_ratio=source_to_target_prior_ratio,
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

    # CSD-iPOT uses survival probabilities at observed times as calibration scores.
    nc_ipot = PreFitSurvivalPredictionNC(deepcopy(model), scorer_args)
    icp_ipot = CSDiPOT(
        nc_ipot,
        decensor_method=args_cli.decensor_method,
        n_percentile=args_cli.n_quantiles,
    )
    icp_ipot.fit(train_frame, val_frame)
    icp_ipot.calibrate(val_frame)
    q_levels_ipot, q_preds_ipot = icp_ipot.predict(x_test)

    # Save predictions and labels; evaluation is handled by single_split_eval.py.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        baseline_survival=surv_test,
        baseline_time_coordinates=time_coordinates,
        csd_q_levels=q_levels_csd,
        csd_q_preds=q_preds_csd,
        ipot_q_levels=q_levels_ipot,
        ipot_q_preds=q_preds_ipot,
        x_test=x_test,
        t_test=t_test,
        e_test=e_test,
        t_train_ref=t_train_ref,
        e_train_ref=e_train_ref,
        baseline_method=np.array(f"Baseline {args_cli.model}"),
        csd_method=np.array(f"{args_cli.model} + weighted CSD"),
        ipot_method=np.array(f"{args_cli.model} + CSD-iPOT"),
        metadata=np.array(json.dumps({**vars(args_cli), "dataset_name": dataset_name})),
    )
    print(f"Saved survival outputs to {output_path}")


if __name__ == "__main__":
    main()
