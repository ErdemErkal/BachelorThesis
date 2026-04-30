import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

sys.dont_write_bytecode = True
SURVIVAL_OUTPUT_DIR = os.path.join("results", "survival_outputs")
sys.path.append("MakeSurvivalCalibratedAgain")
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

from icp import CSDiPOT, ConformalSurvDist  # noqa: E402
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC  # noqa: E402
from utils.util_survival import format_pred_sksurv, survival_data_split  # noqa: E402


def run_command(command):
    subprocess.check_call(command, shell=True)


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
        "--split-id", type=int, default=1, help="Index into splits['train']."
    )
    parser.add_argument(
        "--ess", default="1.0", help="ESS value for the shifted test filename."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["AFT", "GB"], default="AFT")
    parser.add_argument("--n-quantiles", type=int, default=9)
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument(
        "--decensor-method",
        default="sampling",
        choices=["uncensored", "margin", "PO", "sampling"],
    )
    parser.add_argument("--penalizer", type=float, default=0.01)
    parser.add_argument("--gb-estimators", type=int, default=100)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz filename under results/survival_outputs, or a full path.",
    )
    parser.add_argument("--skip-install", action="store_true")
    return parser.parse_args()


def survival_output_path(filename_or_path):
    if not filename_or_path.lower().endswith(".npz"):
        filename_or_path = f"{filename_or_path}.npz"
    if os.path.isabs(filename_or_path):
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
        or f"{dataset_name}_split_{args.split_id}_ess_{args.ess}_{args.model}.npz"
    )
    output_path = survival_output_path(output_name)
    return dataset_name, dataset_path, test_path, splits_path, output_path


def setup_environment(skip_install):
    if not os.path.exists("MakeSurvivalCalibratedAgain"):
        raise FileNotFoundError("MakeSurvivalCalibratedAgain is missing.")

    if not skip_install:
        run_command(f'"{sys.executable}" -m pip install --upgrade pip')
        run_command(
            f'"{sys.executable}" -m pip install -r MakeSurvivalCalibratedAgain/requirements.txt'
        )
        run_command(f'"{sys.executable}" -m pip install -U ucimlrepo')
        run_command(f'"{sys.executable}" -m pip install SurvivalEVAL')


def add_survival_columns(features, source):
    frame = features.copy()
    frame["time"] = source["time"].astype(float).values
    frame["event"] = source["event"].astype(int).values
    return frame


def make_model_frames(data_train, data_val, data_test):
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


def make_sksurv_labels(time, event):
    labels = np.empty(dtype=[("cens", bool), ("time", np.float64)], shape=time.shape[0])
    labels["cens"] = event.astype(bool)
    labels["time"] = time
    return labels


def add_time_zero(surv, time_coordinates):
    if len(time_coordinates) > 0 and time_coordinates[0] == 0:
        return surv, time_coordinates
    return (
        np.concatenate([np.ones((surv.shape[0], 1)), surv], axis=1),
        np.concatenate([np.array([0.0]), time_coordinates], axis=0),
    )


def main():
    args_cli = parse_args()
    dataset_name, dataset_path, test_path, splits_path, output_path = resolve_paths(
        args_cli
    )
    setup_environment(args_cli.skip_install)

    class PreFitQuantileRegressionNC(QuantileRegressionNC):
        def fit(self, train_set, val_set):
            return self

    class PreFitSurvivalPredictionNC(SurvivalPredictionNC):
        def fit(self, train_set, val_set):
            return self

    np.random.seed(args_cli.seed)

    data = pd.read_csv(dataset_path).rename(columns={"status": "event"})
    data_test = pd.read_csv(test_path, sep="\t").rename(columns={"status": "event"})
    splits = json.loads(open(splits_path).read())

    train_ix = np.array(splits["train"][args_cli.split_id])
    train_split = data.iloc[train_ix, :]
    data_train, data_val, _ = survival_data_split(
        train_split,
        stratify_colname="both",
        frac_train=0.8,
        frac_val=0.2,
        frac_test=0.0,
        random_state=args_cli.seed,
    )

    train_frame, val_frame, test_frame = make_model_frames(
        data_train, data_val, data_test
    )

    t_train = train_frame["time"].values
    e_train = train_frame["event"].values.astype(int)
    t_val = val_frame["time"].values
    e_val = val_frame["event"].values.astype(int)
    t_test = test_frame["time"].values
    e_test = test_frame["event"].values.astype(int)
    t_train_ref = np.concatenate([t_train, t_val])
    e_train_ref = np.concatenate([e_train, e_val])
    x_test = test_frame.drop(columns=["time", "event"]).values

    if args_cli.model == "AFT":
        model = WeibullAFTFitter(penalizer=args_cli.penalizer)
        model.fit(train_frame, duration_col="time", event_col="event")
        surv_df = model.predict_survival_function(
            test_frame.drop(columns=["time", "event"])
        )
        surv_test = surv_df.values.T
        time_coordinates = surv_df.index.values
    elif args_cli.model == "GB":
        model = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=args_cli.gb_estimators,
            random_state=args_cli.seed,
        )
        model.fit(
            train_frame.drop(columns=["time", "event"]).values,
            make_sksurv_labels(t_train, e_train),
        )
        surv_test, time_coordinates = format_pred_sksurv(
            model.predict_survival_function(x_test)
        )
    else:
        raise ValueError(f"Unsupported model: {args_cli.model}")

    surv_test, time_coordinates = add_time_zero(surv_test, time_coordinates)

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

    nc_ipot = PreFitSurvivalPredictionNC(deepcopy(model), scorer_args)
    icp_ipot = CSDiPOT(
        nc_ipot,
        decensor_method=args_cli.decensor_method,
        n_percentile=args_cli.n_quantiles,
    )
    icp_ipot.fit(train_frame, val_frame)
    icp_ipot.calibrate(val_frame)
    q_levels_ipot, q_preds_ipot = icp_ipot.predict(x_test)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        baseline_survival=surv_test,
        baseline_time_coordinates=time_coordinates,
        csd_q_levels=q_levels_csd,
        csd_q_preds=q_preds_csd,
        ipot_q_levels=q_levels_ipot,
        ipot_q_preds=q_preds_ipot,
        t_test=t_test,
        e_test=e_test,
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
