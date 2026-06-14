import argparse
import os
import sys

import numpy as np
import pandas as pd
from SurvivalEVAL import QuantileRegEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration
import CondCalEvaluation

sys.dont_write_bytecode = True
SURVIVAL_OUTPUT_DIR = os.path.join("results", "survival_outputs")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="veteran_split_1_ess_1.0_AFT.npz",
        help="Prediction .npz filename under results/survival_outputs, or a full path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def survival_output_path(filename_or_path):
    if not filename_or_path.lower().endswith(".npz"):
        filename_or_path = f"{filename_or_path}.npz"
    if os.path.isabs(filename_or_path) or os.path.dirname(filename_or_path):
        return filename_or_path
    return os.path.join(SURVIVAL_OUTPUT_DIR, filename_or_path)


def scalar_text(value):
    return str(value.item()) if hasattr(value, "item") else str(value)


def d_calibration_compat(pred_probs, event_indicators, num_bins):
    _, pvalue, dcal_hist = d_calibration(pred_probs, event_indicators, num_bins)
    return pvalue, dcal_hist


def d_calibration_statistic(evaluator, num_bins=10):
    predict_probs = evaluator.predict_probability_from_curve(evaluator.event_times)
    statistic, _, _ = d_calibration(
        pred_probs=predict_probs,
        event_indicators=evaluator.event_indicators,
        num_bins=num_bins,
    )
    return statistic


def worst_slab_calibration(evaluator, x_test, t_test, e_test, seed):
    np.random.seed(seed)
    CondCalEvaluation.d_calibration = d_calibration_compat
    predict_probs = evaluator.predict_probability_from_curve(t_test)
    return CondCalEvaluation.wsc_xcal(
        x_test, e_test, predict_probs, random_state=seed
    )


def main():
    args = parse_args()
    seed = args.seed

    # Load saved model outputs and the event/time arrays used for evaluation.
    input_path = survival_output_path(args.input)
    outputs = np.load(input_path, allow_pickle=False)
    t_test = outputs["t_test"]
    e_test = outputs["e_test"]
    t_train_ref = outputs["t_train_ref"]
    e_train_ref = outputs["e_train_ref"]
    x_test = outputs["x_test"]

    # Baseline AFT outputs are survival curves over time coordinates.
    evl_base = SurvivalEvaluator(
        outputs["baseline_survival"],
        outputs["baseline_time_coordinates"],
        t_test,
        e_test,
        t_train_ref,
        e_train_ref,
        predict_time_method="Median",
        interpolation="Pchip",
    )

    # CSD and CSD-iPOT outputs are calibrated quantile predictions.
    evl_csd = QuantileRegEvaluator(
        outputs["csd_q_preds"],
        outputs["csd_q_levels"],
        t_test,
        e_test,
        t_train_ref,
        e_train_ref,
        predict_time_method="Median",
        interpolation="Pchip",
    )

    evl_ipot = QuantileRegEvaluator(
        outputs["ipot_q_preds"],
        outputs["ipot_q_levels"],
        t_test,
        e_test,
        t_train_ref,
        e_train_ref,
        predict_time_method="Median",
        interpolation="Pchip",
    )

    # Report the same metrics for baseline and calibrated outputs.
    results = pd.DataFrame(
        [
            {
                "method": scalar_text(outputs["baseline_method"]),
                "Harrell_CI": evl_base.concordance(method="Harrell")[0],
                "D_CAL_statistic": d_calibration_statistic(evl_base),
                "WS_cal": worst_slab_calibration(
                    evl_base, x_test, t_test, e_test, seed
                ),
            },
            {
                "method": scalar_text(outputs["csd_method"]),
                "Harrell_CI": evl_csd.concordance(method="Harrell")[0],
                "D_CAL_statistic": d_calibration_statistic(evl_csd),
                "WS_cal": worst_slab_calibration(
                    evl_csd, x_test, t_test, e_test, seed
                ),
            },
            {
                "method": scalar_text(outputs["ipot_method"]),
                "Harrell_CI": evl_ipot.concordance(method="Harrell")[0],
                "D_CAL_statistic": d_calibration_statistic(evl_ipot),
                "WS_cal": worst_slab_calibration(
                    evl_ipot, x_test, t_test, e_test, seed
                ),
            },
        ]
    )

    print(f"{os.path.basename(input_path)}:")
    print(results)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        results.to_csv(args.output_csv, index=False)
        print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
