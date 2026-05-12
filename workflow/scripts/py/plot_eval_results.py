"""Create pooled calibration violin plots from eval metric CSV files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_RE = re.compile(
    r"^(?P<dataset>.+)_split_(?P<split_id>\d+)_ess_(?P<ess>[^_]+)_seed_(?P<seed>\d+)_(?P<model>.+)\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read results/eval_metrics CSVs and create calibration violin "
            "plot for each model, pooling all datasets together."
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
            "D_CAL_pvalue and WS_cal when available."
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
        "--threshold",
        default=0.05,
        type=float,
        help="Optional horizontal reference line. Use a negative value to disable.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def read_eval_metrics(
    input_dir: Path,
    pattern: str,
    metric_candidates: list[str],
    datasets: set[str] | None = None,
    models: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for csv_path in sorted(input_dir.glob(pattern)):
        match = METRIC_RE.match(csv_path.name)
        if not match:
            continue

        metadata = match.groupdict()
        if datasets and metadata["dataset"] not in datasets:
            continue
        if models and metadata["model"] not in models:
            continue

        metrics = pd.read_csv(csv_path)
        metric_cols = [col for col in metric_candidates if col in metrics.columns]
        if not metric_cols:
            continue

        metrics["dataset"] = metadata["dataset"]
        metrics["split_id"] = int(metadata["split_id"])
        metrics["ess"] = float(metadata["ess"])
        metrics["ess_label"] = metadata["ess"]
        metrics["seed"] = int(metadata["seed"])
        metrics["model"] = metadata["model"]
        metrics["source_file"] = csv_path.name
        for metric_col in metric_cols:
            metrics[metric_col] = pd.to_numeric(metrics[metric_col], errors="coerce")
        rows.append(metrics[["method", *metric_cols, "dataset", "split_id", "ess", "ess_label", "seed", "model", "source_file"]])

    if not rows:
        raise ValueError(
            f"No eval metric CSVs matched {input_dir / pattern} with any of "
            f"these columns: {metric_candidates}"
        )

    return pd.concat(rows, ignore_index=True)


def available_metrics(data: pd.DataFrame, requested_metrics: list[str]) -> list[str]:
    metrics = [metric for metric in requested_metrics if metric in data.columns]
    if not metrics:
        raise ValueError(f"None of the requested metric columns exist: {requested_metrics}")
    return metrics


def metric_label(metric: str) -> str:
    labels = {
        "D_CAL_pvalue": "D-calibration p-value",
        "WS_cal": "Worst-slab calibration score",
        "WorstSlab_D_CAL": "Worst-slab D-calibration score",
    }
    return labels.get(metric, metric)


def plot_model(
    data: pd.DataFrame,
    model: str,
    metric: str,
    output_dir: Path,
    output_format: str,
    dpi: int,
    ess_order: str,
    threshold: float,
) -> Path:
    subset = data[data["model"] == model].dropna(subset=[metric]).copy()
    if subset.empty:
        raise ValueError(f"No finite {metric} values found for model {model}")

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
    ax.set_title(f"{model} - {metric_label(metric)} - pooled across {n_datasets} dataset(s)")
    ax.set_xlabel("ESS ratio")
    ax.set_ylabel(metric_label(metric))
    if metric == "D_CAL_pvalue":
        ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Method", loc="best", frameon=False)

    fig.tight_layout()
    output_path = output_dir / f"{safe_name(model)}_pooled_{safe_name(metric)}_violin.{output_format}"
    fig.savefig(output_path, dpi=dpi if output_format == "png" else None)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    requested_metrics = args.metric or ["D_CAL_pvalue", "WS_cal"]
    data = read_eval_metrics(
        args.input_dir,
        args.pattern,
        requested_metrics,
        datasets=set(args.dataset) if args.dataset else None,
        models=set(args.model) if args.model else None,
    )
    metrics = available_metrics(data, requested_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        metric_data = data.dropna(subset=[metric])
        if metric_data.empty:
            continue
        for model in sorted(metric_data["model"].unique()):
            written.append(
                plot_model(
                    data=metric_data,
                    model=model,
                    metric=metric,
                    output_dir=args.output_dir,
                    output_format=args.format,
                    dpi=args.dpi,
                    ess_order=args.ess_order,
                    threshold=args.threshold if metric == "D_CAL_pvalue" else -1,
                )
            )

    print(f"Wrote {len(written)} plot(s) to {args.output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
