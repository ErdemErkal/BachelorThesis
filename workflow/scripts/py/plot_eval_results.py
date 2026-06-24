"""Create calibration violin plots from eval metric CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRIC_RE = re.compile(
    r"^(?P<dataset>.+)_split_(?P<split_id>\d+)_ess_(?P<ess>[^_]+)"
    r"_seed_(?P<seed>\d+)_(?P<run_label>.+)\.csv$"
)

DEFAULT_WEIGHTING_SUFFIXES = {
    "unweighted": "",
    "weighted": "_weighted",
}


def comparison_methods(model: str) -> dict[str, list[str]]:
    return {
        "csd": [
            f"Baseline {model}",
            f"{model} + CSD",
            f"{model} + weighted CSD",
        ],
        "ipot": [
            f"Baseline {model}",
            f"{model} + CSD-iPOT",
            f"{model} + weighted CSD-iPOT",
        ],
    }


def methods_for_weighting(model: str, weighting: str) -> list[str]:
    baseline = f"Baseline {model}"
    if weighting == "unweighted":
        return [baseline, f"{model} + CSD", f"{model} + CSD-iPOT"]
    if weighting == "weighted":
        return [baseline, f"{model} + weighted CSD", f"{model} + weighted CSD-iPOT"]

    methods = [baseline]
    for comparison_order in comparison_methods(model).values():
        methods.extend(comparison_order[1:])
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read results/eval_metrics CSVs and create calibration violin plots "
            "for each model, both per dataset and pooled across datasets."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="results/eval_metrics",
        type=Path,
        help="Directory containing eval metric CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots/eval_violin",
        type=Path,
        help="Directory where plot images will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for eval metric files inside --input-dir.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        help=(
            "Metric column to plot. Can be repeated. If omitted, plots "
            "D_CAL_statistic and WS_cal when available."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format.",
    )
    parser.add_argument(
        "--dpi",
        default=200,
        type=int,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--ess-order",
        default="desc",
        choices=["asc", "desc"],
        help="Sort ESS values on the x axis.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Limit which datasets are pooled. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Limit plotting to one model. Can be repeated.",
    )
    parser.add_argument(
        "--weighting",
        action="append",
        help="Limit plotting to one weighting mode. Can be repeated.",
    )
    parser.add_argument(
        "--weighting-suffix",
        action="append",
        metavar="MODE=SUFFIX",
        help=(
            "Map a weighting mode to its filename suffix. Can be repeated. "
            "Defaults to unweighted= and weighted=_weighted."
        ),
    )
    parser.add_argument(
        "--threshold",
        default=0.05,
        type=float,
        help="Optional horizontal reference line. Use a negative value to disable.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_weighting_suffixes(pairs: list[str] | None) -> dict[str, str]:
    if not pairs:
        return DEFAULT_WEIGHTING_SUFFIXES.copy()

    suffixes: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected weighting suffix as MODE=SUFFIX, got {pair!r}")
        mode, suffix = pair.split("=", 1)
        if not mode:
            raise ValueError(f"Weighting mode cannot be empty in {pair!r}")
        suffixes[mode] = suffix
    return suffixes


def split_run_label(
    run_label: str, weighting_suffixes: dict[str, str]
) -> tuple[str, str]:
    non_empty_suffixes = sorted(
        ((mode, suffix) for mode, suffix in weighting_suffixes.items() if suffix),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for mode, suffix in non_empty_suffixes:
        if run_label.endswith(suffix):
            return run_label[: -len(suffix)], mode

    for mode, suffix in weighting_suffixes.items():
        if suffix == "":
            return run_label, mode

    return run_label, "unweighted"


def deduplicate_baseline_rows(data: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = data["method"].eq("Baseline " + data["model"].astype(str))
    if not baseline_mask.any():
        return data

    baseline = data[baseline_mask].copy()
    non_baseline = data[~baseline_mask]

    baseline["_weighting_rank"] = baseline["weighting"].ne("unweighted").astype(int)
    baseline = baseline.sort_values(
        [
            "dataset",
            "split_id",
            "ess",
            "seed",
            "model",
            "method",
            "_weighting_rank",
            "source_file",
        ],
        kind="mergesort",
    )
    baseline = baseline.drop_duplicates(
        subset=["dataset", "split_id", "ess", "seed", "model", "method"],
        keep="first",
    ).drop(columns="_weighting_rank")

    return pd.concat([non_baseline, baseline], ignore_index=True)


def read_eval_metrics(
    input_dir: Path,
    pattern: str,
    metric_candidates: list[str],
    weighting_suffixes: dict[str, str],
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    weightings: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for csv_path in sorted(input_dir.glob(pattern)):
        match = METRIC_RE.match(csv_path.name)
        if not match:
            continue

        metadata = match.groupdict()
        model, weighting = split_run_label(metadata["run_label"], weighting_suffixes)
        if datasets and metadata["dataset"] not in datasets:
            continue
        if models and model not in models:
            continue
        if weightings and weighting not in weightings:
            continue

        metrics = pd.read_csv(csv_path)
        metric_cols = [col for col in metric_candidates if col in metrics.columns]
        if not metric_cols:
            continue

        methods_to_keep = set(methods_for_weighting(model, weighting))
        metrics = metrics[metrics["method"].isin(methods_to_keep)].copy()
        if metrics.empty:
            continue

        metrics["dataset"] = metadata["dataset"]
        metrics["split_id"] = int(metadata["split_id"])
        metrics["ess"] = float(metadata["ess"])
        metrics["ess_label"] = metadata["ess"]
        metrics["seed"] = int(metadata["seed"])
        metrics["model"] = model
        metrics["weighting"] = weighting
        metrics["run_label"] = metadata["run_label"]
        metrics["source_file"] = csv_path.name
        for metric_col in metric_cols:
            metrics[metric_col] = pd.to_numeric(metrics[metric_col], errors="coerce")
        rows.append(
            metrics[
                [
                    "method",
                    *metric_cols,
                    "dataset",
                    "split_id",
                    "ess",
                    "ess_label",
                    "seed",
                    "model",
                    "weighting",
                    "run_label",
                    "source_file",
                ]
            ]
        )

    if not rows:
        raise ValueError(
            f"No eval metric CSVs matched {input_dir / pattern} with any of "
            f"these columns: {metric_candidates}"
        )

    return deduplicate_baseline_rows(pd.concat(rows, ignore_index=True))


def available_metrics(data: pd.DataFrame, requested_metrics: list[str]) -> list[str]:
    metrics = [metric for metric in requested_metrics if metric in data.columns]
    if not metrics:
        raise ValueError(
            f"None of the requested metric columns exist: {requested_metrics}"
        )
    return metrics


def metric_label(metric: str) -> str:
    labels = {
        "D_CAL_pvalue": "D-calibration p-value",
        "D_CAL_statistic": "D-calibration test statistic",
        "WS_cal": "Worst-slab calibration score",
        "WorstSlab_D_CAL": "Worst-slab D-calibration score",
    }
    return labels.get(metric, metric)


def plot_model_scope(
    data: pd.DataFrame,
    model: str,
    comparison: str,
    method_order: list[str],
    metric: str,
    output_dir: Path,
    output_format: str,
    dpi: int,
    ess_order: str,
    threshold: float,
    dataset: str | None = None,
) -> Path | None:
    subset = data[data["model"] == model].dropna(subset=[metric]).copy()
    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]
    subset = subset[subset["method"].isin(method_order)]
    if subset.empty:
        return None

    hue_order = [method for method in method_order if method in set(subset["method"])]

    reverse = ess_order == "desc"
    ess_values = sorted(subset["ess"].unique(), reverse=reverse)
    ess_labels = [
        subset.loc[subset["ess"] == ess, "ess_label"].iloc[0] for ess in ess_values
    ]

    width = max(8, len(ess_labels) * 1.8)
    height = 5.5
    fig, ax = plt.subplots(figsize=(width, height))

    sns.violinplot(
        data=subset,
        x="ess_label",
        y=metric,
        hue="method",
        hue_order=hue_order,
        order=ess_labels,
        cut=0,
        inner="quartile",
        density_norm="width",
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="ess_label",
        y=metric,
        hue="method",
        hue_order=hue_order,
        order=ess_labels,
        dodge=True,
        jitter=0.18,
        alpha=0.35,
        size=2,
        linewidth=0,
        legend=False,
        ax=ax,
    )

    if threshold >= 0:
        ax.axhline(threshold, color="0.25", linestyle="--", linewidth=0.9, alpha=0.7)

    n_datasets = subset["dataset"].nunique()
    if dataset is None:
        scope_label = f"pooled across {n_datasets} dataset(s)"
        output_stem = (
            f"{safe_name(model)}_{safe_name(comparison)}"
            f"_pooled_{safe_name(metric)}_violin"
        )
    else:
        scope_label = dataset
        output_stem = (
            f"{safe_name(model)}_{safe_name(comparison)}"
            f"_{safe_name(dataset)}_{safe_name(metric)}_violin"
        )
    ax.set_title(f"{model} {comparison} - {metric_label(metric)} - {scope_label}")
    ax.set_xlabel("ESS ratio")
    ax.set_ylabel(metric_label(metric))
    if metric == "D_CAL_pvalue":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Method", loc="best", frameon=False)

    fig.tight_layout()
    output_path = output_dir / f"{output_stem}.{output_format}"
    fig.savefig(output_path, dpi=dpi if output_format == "png" else None)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    requested_metrics = args.metric or ["D_CAL_statistic", "WS_cal"]
    weighting_suffixes = parse_weighting_suffixes(args.weighting_suffix)
    data = read_eval_metrics(
        args.input_dir,
        args.pattern,
        requested_metrics,
        weighting_suffixes=weighting_suffixes,
        datasets=set(args.dataset) if args.dataset else None,
        models=set(args.model) if args.model else None,
        weightings=set(args.weighting) if args.weighting else None,
    )
    metrics = available_metrics(data, requested_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        metric_data = data.dropna(subset=[metric])
        if metric_data.empty:
            continue
        for model in sorted(metric_data["model"].unique()):
            comparisons = comparison_methods(model)
            model_data = metric_data[metric_data["model"] == model]
            for comparison, method_order in comparisons.items():
                for dataset in sorted(model_data["dataset"].unique()):
                    output_path = plot_model_scope(
                        data=metric_data,
                        model=model,
                        comparison=comparison,
                        method_order=method_order,
                        dataset=dataset,
                        metric=metric,
                        output_dir=args.output_dir,
                        output_format=args.format,
                        dpi=args.dpi,
                        ess_order=args.ess_order,
                        threshold=args.threshold if metric == "D_CAL_pvalue" else -1,
                    )
                    if output_path is not None:
                        written.append(output_path)

                output_path = plot_model_scope(
                    data=metric_data,
                    model=model,
                    comparison=comparison,
                    method_order=method_order,
                    metric=metric,
                    output_dir=args.output_dir,
                    output_format=args.format,
                    dpi=args.dpi,
                    ess_order=args.ess_order,
                    threshold=args.threshold if metric == "D_CAL_pvalue" else -1,
                )
                if output_path is not None:
                    written.append(output_path)

    print(f"Wrote {len(written)} plot(s) to {args.output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
