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
from icp import ConformalSurvDist, CSDiPOT
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from utils.util_survival import format_pred_sksurv, survival_data_split, survival_to_quantile


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


def make_structured_survival_target(time_values, event_values):
    y = np.zeros(time_values.shape[0], dtype=[("event", bool), ("time", float)])
    y["event"] = event_values.astype(bool)
    y["time"] = time_values.astype(float)
    return y


def main():
    args_cli = parse_args()
    dataset_name, dataset_path, test_path, splits_path, output_path = resolve_paths(
        args_cli
    )

    class PreFitQuantileRegressionNC(QuantileRegressionNC):
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

    class PreFitSurvivalPredictionNC(SurvivalPredictionNC):
        def fit(self, train_set, val_set):
            return self

        def predict_nc(self, x: np.ndarray, feature_names: list[str] = None):
            if not isinstance(
                self.model, (CoxPHSurvivalAnalysis, GradientBoostingSurvivalAnalysis)
            ):
                return super().predict_nc(x, feature_names)

            pred_surv = self.model.predict_survival_function(x)
            surv_prob, time_coordinates = format_pred_sksurv(pred_surv)
            if time_coordinates[0] != 0:
                time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
                surv_prob = np.concatenate(
                    [np.ones([surv_prob.shape[0], 1]), surv_prob], axis=1
                )
            time_coordinates = np.repeat(
                time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0
            )
            return surv_prob, time_coordinates

    np.random.seed(args_cli.seed)

    def extract_tilt_feature_name(shift_tsv_path: str) -> str:
        try:
            tilt_meta = pd.read_csv(shift_tsv_path, sep="\t", usecols=["tilt_feature"])
        except ValueError as exc:
            raise ValueError(
                "Shift TSV must include a 'tilt_feature' column."
            ) from exc
        tilt_values = (
            tilt_meta["tilt_feature"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
        )
        if tilt_values.shape[0] != 1:
            raise ValueError(
                "Expected exactly one non-empty tilt_feature value in shift TSV."
            )
        return tilt_values[0]

    def is_natural_shift(tilt_name: str) -> bool:
        return str(tilt_name).strip().lower() in {"none", "natural", "__none__"}

    # Load the original dataset, shifted test split, and saved train indices.
    data = pd.read_csv(dataset_path).rename(columns={"status": "event"})
    data_test = pd.read_csv(test_path, sep="\t").rename(columns={"status": "event"})
    # Split-bookkeeping only; must not enter survival / density models.
    data = data.drop(columns=["source_dataset"], errors="ignore")
    data_test = data_test.drop(columns=["source_dataset"], errors="ignore")
    tilt_feature_name = extract_tilt_feature_name(test_path)
    natural_shift = is_natural_shift(tilt_feature_name)
    if not natural_shift:
        if tilt_feature_name not in data.columns:
            raise ValueError(
                f"tilt_feature '{tilt_feature_name}' is missing from dataset columns."
            )
        if tilt_feature_name in {"time", "event"}:
            raise ValueError("tilt_feature cannot be a survival target column.")
    oracle_test_weights = None
    if "true_density_weight" in data_test.columns:
        oracle_test_weights = data_test["true_density_weight"].to_numpy(dtype=float)
        data_test = data_test.drop(columns=["true_density_weight"])
    if "dataset_index" in data_test.columns:
        data_test = data_test.drop(columns=["dataset_index"])
    if "tilt_feature" in data_test.columns:
        data_test = data_test.drop(columns=["tilt_feature"])
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
            random_state=args_cli.seed
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

    nc_csd = PreFitQuantileRegressionNC(deepcopy(model), scorer_args)
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
        oracle_test_weights=oracle_test_weights
        if oracle_test_weights is not None
        else np.ones_like(t_test, dtype=float),
        t_train_ref=t_train_ref,
        e_train_ref=e_train_ref,
        baseline_method=np.array(f"Baseline {args_cli.model}"),
        csd_method=np.array(f"{args_cli.model} + CSD"),
        ipot_method=np.array(f"{args_cli.model} + CSD-iPOT"),
        metadata=np.array(json.dumps({**vars(args_cli), "dataset_name": dataset_name})),
    )
    print(f"Saved survival outputs to {output_path}")


if __name__ == "__main__":
    main()
