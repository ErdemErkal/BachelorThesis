import json
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.stderr = sys.stdout = open(snakemake.log[0], "w")


def main(data_path, split_path, seed, n_splits, k_time_bins):
    dataset = pd.read_csv(data_path)
    if "source_dataset" not in dataset.columns:
        raise ValueError("Cross dataset file must include 'source_dataset' column.")

    source = dataset["source_dataset"].astype(str).str.lower()
    train_idx = dataset.index[source == "rotterdam"].to_numpy()
    test_pool_idx = dataset.index[source == "gbsg"].to_numpy()

    if train_idx.size == 0 or test_pool_idx.size == 0:
        raise ValueError(
            "Cross dataset must contain both rotterdam (train) and gbsg (test) rows."
        )
    if n_splits < 2 or n_splits > test_pool_idx.size:
        raise ValueError(
            f"cross_n_splits={n_splits} invalid for gbsg test pool size {test_pool_idx.size}."
        )

    # Stratify GBSG test folds the same way as make_splits.py:
    # joint strata over event indicator and within-event time quantile bins.
    test_pool = dataset.loc[test_pool_idx].copy()
    if "event" in test_pool.columns:
        event = test_pool["event"].to_numpy()
    elif "status" in test_pool.columns:
        event = test_pool["status"].to_numpy()
    else:
        raise ValueError("Cross dataset must contain 'event' or 'status'.")
    time = test_pool["time"].to_numpy()

    # Calculate time quantiles separately for censored and uncensored to prevent
    # the censoring distribution from skewing the event time bins.
    time_bins = pd.Series(index=test_pool.index, dtype=float)
    for e in [0, 1]:
        mask = event == e
        if mask.sum() > 0:
            time_bins[mask] = pd.qcut(
                time[mask], q=k_time_bins, labels=False, duplicates="drop"
            )

    stratify_label = [f"{e}_{t}" for e, t in zip(event, time_bins)]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_splits = []
    test_splits = []
    for _, test_local_idx in kf.split(test_pool_idx, stratify_label):
        # Always train on the full Rotterdam set; only GBSG is fold-split.
        train_splits.append(train_idx.tolist())
        test_splits.append(test_pool_idx[test_local_idx].tolist())

    with open(split_path, "w") as fp:
        json.dump({"train": train_splits, "test": test_splits}, fp)


main(
    data_path=snakemake.input[0],
    split_path=snakemake.output[0],
    seed=snakemake.params["seed"],
    n_splits=snakemake.params["n_splits"],
    k_time_bins=snakemake.params["k_time_bins"],
)
