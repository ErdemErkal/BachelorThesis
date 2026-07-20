import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from SurvivalEVAL import QuantileRegEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.DistributionCalibration import (
    create_censor_hist as survivaleval_create_censor_hist,
)
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
    parser.add_argument(
        "--ws-m",
        type=int,
        default=200,
        help="Number of random directions for worst-slab search (smaller is faster).",
    )
    parser.add_argument(
        "--ws-delta",
        type=float,
        default=0.5,
        help="Minimum slab mass fraction for worst-slab search (larger is faster).",
    )
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


def weighted_d_calibration(pred_probs, event_indicators, weights, num_bins=10):
    pred_probs = np.asarray(pred_probs, dtype=float)
    event_indicators = np.asarray(event_indicators, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if pred_probs.shape[0] != event_indicators.shape[0] or pred_probs.shape[0] != weights.shape[0]:
        raise ValueError("pred_probs, event_indicators, and weights must have same length.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative.")
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have positive finite sum.")

    quantile = np.linspace(1, 0, num_bins + 1)
    event_mask = event_indicators.astype(bool)
    event_probs = pred_probs[event_mask]
    event_weights = weights[event_mask]
    event_position = np.digitize(event_probs, quantile)
    event_position[event_position == 0] = 1

    event_hist = np.zeros(num_bins, dtype=float)
    for i, pos in enumerate(event_position):
        event_hist[pos - 1] += event_weights[i]

    censor_mask = ~event_mask
    censored_probs = pred_probs[censor_mask]
    censored_weights = weights[censor_mask]
    censor_hist = np.zeros(num_bins, dtype=float)
    for i, prob in enumerate(censored_probs):
        censor_hist += survivaleval_create_censor_hist(prob, num_bins) * censored_weights[i]

    raw_hist = event_hist + censor_hist
    ess = (weight_sum**2) / np.sum(weights**2)
    total_raw = float(np.sum(raw_hist))
    if total_raw <= 0:
        raise ValueError("Weighted D-calibration histogram has non-positive mass.")
    scaling_factor = ess / total_raw
    scaled_hist = raw_hist * scaling_factor
    expected_counts = np.ones(num_bins, dtype=float) * (ess / num_bins)
    statistic, pvalue = chisquare(scaled_hist, f_exp=expected_counts)
    return statistic, pvalue, raw_hist


def xcal_from_hist(hist):
    hist = np.asarray(hist, dtype=float)
    if hist.ndim != 1:
        raise ValueError("xcal_from_hist expects a 1D histogram.")
    total = float(np.sum(hist))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    num_bins = hist.shape[0]
    if num_bins <= 1:
        return 0.0
    cdf = np.cumsum(hist / total)
    optimal = (np.arange(num_bins) + 1) / num_bins
    dof = 1.0 / (num_bins - 1)
    return dof * np.sum(np.square(cdf - optimal))


def sample_sphere(n, p, rng):
    v = rng.standard_normal((p, n))
    norm = np.linalg.norm(v, axis=0)
    norm[norm == 0] = 1.0
    v /= norm
    return v.T


def weighted_wsc_v(X, event_indicators, predict_probs, weights, num_bins, delta, v):
    weights = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative in weighted_wsc_v.")

    quantile = np.linspace(1, 0, num_bins + 1)
    position = np.digitize(predict_probs, quantile)
    position[position == 0] = 1

    binning = np.zeros((len(predict_probs), num_bins), dtype=float)
    for i in range(len(predict_probs)):
        if event_indicators[i]:
            binning[i, position[i] - 1] += weights[i]
        else:
            binning[i, :] = (
                survivaleval_create_censor_hist(predict_probs[i], num_bins) * weights[i]
            )

    n = len(predict_probs)
    z = np.dot(X, v)
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    binning_ordered = binning[z_order, :]
    weights_ordered = weights[z_order]

    cumsum_hist = np.cumsum(binning_ordered, axis=0)
    cumsum_weights = np.cumsum(weights_ordered)
    total_weight = float(cumsum_weights[-1])
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("weighted_wsc_v requires positive finite total sample weight.")
    min_mass = delta * total_weight

    ai_best = 0
    bi_best = n - 1
    xcal_max = xcal_from_hist(cumsum_hist[-1, :])

    for ai in range(n):
        for bi in range(ai, n):
            slab_weight = cumsum_weights[bi] - (
                cumsum_weights[ai - 1] if ai > 0 else 0.0
            )
            if slab_weight < min_mass:
                continue
            if ai == 0:
                slab_hist = cumsum_hist[bi, :]
            else:
                slab_hist = cumsum_hist[bi, :] - cumsum_hist[ai - 1, :]
            slab_xcal = xcal_from_hist(slab_hist)
            if slab_xcal > xcal_max:
                ai_best = ai
                bi_best = bi
                xcal_max = slab_xcal
    return xcal_max, z_sorted[ai_best], z_sorted[bi_best]


def weighted_worst_slab(
    X, event_indicators, predict_probs, weights, num_bins, delta=0.33, M=1000, random_state=42
):
    rng = np.random.default_rng(random_state)
    V = sample_sphere(M, p=X.shape[1], rng=rng)
    wsc_list = np.zeros(M, dtype=float)
    a_list = np.zeros(M, dtype=float)
    b_list = np.zeros(M, dtype=float)
    for m in range(M):
        wsc_list[m], a_list[m], b_list[m] = weighted_wsc_v(
            X, event_indicators, predict_probs, weights, num_bins, delta, V[m]
        )
    idx_star = np.argmax(wsc_list)
    return wsc_list[idx_star], V[idx_star], a_list[idx_star], b_list[idx_star]


def weighted_wsc_xcal(
    X,
    event_indicators,
    predict_probs,
    weights,
    num_bins=10,
    delta=0.33,
    test_size=0.5,
    M=1000,
    random_state=42,
):
    X = np.asarray(X, dtype=float)
    event_indicators = np.asarray(event_indicators, dtype=int)
    predict_probs = np.asarray(predict_probs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative in weighted_wsc_xcal.")
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("weighted_wsc_xcal requires positive finite total sample weight.")

    clipped_weights = np.clip(weights, a_min=1e-12, a_max=None)
    log_weights = np.log(clipped_weights)
    stratify_labels = None
    quantiles = np.quantile(log_weights, np.linspace(0, 1, 6))
    quantiles = np.unique(quantiles)
    if quantiles.size > 2:
        candidate_labels = np.digitize(log_weights, quantiles[1:-1], right=True)
        label_counts = np.bincount(candidate_labels, minlength=quantiles.size - 1)
        if np.all(label_counts >= 2):
            stratify_labels = candidate_labels

    (
        X_train,
        X_test,
        e_train,
        e_test,
        pred_train,
        pred_test,
        w_train,
        w_test,
    ) = train_test_split(
        X,
        event_indicators,
        predict_probs,
        weights,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    _, v_star, a_star, b_star = weighted_worst_slab(
        X_train,
        e_train,
        pred_train,
        w_train,
        num_bins,
        delta=delta,
        M=M,
        random_state=random_state,
    )
    z = np.dot(X_test, v_star)
    idx = np.where((z >= a_star) * (z <= b_star))[0]
    if idx.size == 0:
        return np.nan
    slab_test_mass = float(np.sum(w_test[idx]))
    total_test_mass = float(np.sum(w_test))
    if (
        not np.isfinite(total_test_mass)
        or total_test_mass <= 0
        or not np.isfinite(slab_test_mass)
        or slab_test_mass < delta * total_test_mass
    ):
        return np.nan
    _, _, dcal_hist = weighted_d_calibration(
        pred_test[idx], e_test[idx], w_test[idx], num_bins=num_bins
    )
    return xcal_from_hist(dcal_hist)


def worst_slab_calibration(evaluator, x_test, t_test, e_test, seed, delta=0.5, M=200):
    np.random.seed(seed)
    CondCalEvaluation.d_calibration = d_calibration_compat
    predict_probs = evaluator.predict_probability_from_curve(t_test)
    return CondCalEvaluation.wsc_xcal(
        x_test, e_test, predict_probs, random_state=seed, delta=delta, test_size=0.5, M=M
    )


def weighted_worst_slab_calibration(
    evaluator, x_test, t_test, e_test, weights, seed, num_bins=10, delta=0.5, M=200
):
    predict_probs = evaluator.predict_probability_from_curve(t_test)
    return weighted_wsc_xcal(
        x_test,
        e_test,
        predict_probs,
        weights,
        num_bins=num_bins,
        delta=delta,
        M=M,
        random_state=seed,
    )


def main():
    args = parse_args()

    # Load saved model outputs and the event/time arrays used for evaluation.
    input_path = survival_output_path(args.input)
    outputs = np.load(input_path, allow_pickle=False)
    metadata = {}
    if "metadata" in outputs.files:
        try:
            metadata = json.loads(str(outputs["metadata"].item()))
        except Exception:
            metadata = {}
    density_mode = metadata.get("density_mode", "na")
    ablation_label = f"density={density_mode}"
    t_test = outputs["t_test"]
    e_test = outputs["e_test"]
    t_train_ref = outputs["t_train_ref"]
    e_train_ref = outputs["e_train_ref"]
    test_weights = (
        outputs["oracle_test_weights"]
        if "oracle_test_weights" in outputs.files
        else np.ones_like(t_test, dtype=float)
    )

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

    # CSD outputs are calibrated quantile predictions.
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

    # Report weighted-evaluation-focused metrics for baseline and weighted CSD.
    base_probs = evl_base.predict_probability_from_curve(t_test)
    base_w_dcal, _, _ = weighted_d_calibration(base_probs, e_test, test_weights)
    csd_probs = evl_csd.predict_probability_from_curve(t_test)
    csd_w_dcal, _, _ = weighted_d_calibration(csd_probs, e_test, test_weights)

    rows = [
        {
            "method": scalar_text(outputs["baseline_method"]),
            "density_mode": density_mode,
            "ablation_label": ablation_label,
            "Antolini_CI": evl_base.concordance_time_dependent(method="Antolini")[0],
            "IBS": evl_base.integrated_brier_score(),
            "D_CAL_statistic": d_calibration_statistic(evl_base),
            "D_CAL_weighted_statistic": base_w_dcal,
        },
        {
            "method": scalar_text(outputs["csd_method"]),
            "density_mode": density_mode,
            "ablation_label": ablation_label,
            "Antolini_CI": evl_csd.concordance_time_dependent(method="Antolini")[0],
            "IBS": evl_csd.integrated_brier_score(),
            "D_CAL_statistic": d_calibration_statistic(evl_csd),
            "D_CAL_weighted_statistic": csd_w_dcal,
        },
    ]
    results = pd.DataFrame(rows)

    print(f"{os.path.basename(input_path)}:")
    print(results)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        results.to_csv(args.output_csv, index=False)
        print(f"Saved metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
