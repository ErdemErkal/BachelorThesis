import sys
import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

sys.stderr = sys.stdout = open(snakemake.log[0], "w")


def main(
    data_paths, split_paths, seed, n_splits, do_shuffle, n_repeats, k_time_bins
) -> int:
    assert len(data_paths) == len(split_paths)
    if n_repeats == 1:
        kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=do_shuffle,
            random_state=seed if do_shuffle else None,
        )
    else:
        if not do_shuffle:
            raise ValueError("Repeated stratified splits require do_shuffle=True.")
        kf = RepeatedStratifiedKFold(
            n_repeats=n_repeats,
            n_splits=n_splits,
            random_state=seed,
        )
    for data_ix in range(len(data_paths)):
        dataset = pd.read_csv(data_paths[data_ix])
        if "event" in dataset.columns:
            event_col = "event"
        elif "status" in dataset.columns:
            event_col = "status"
        else:
            raise ValueError(
                f"{data_paths[data_ix]} must contain an 'event' or 'status' column."
            )
        event = dataset[event_col].values
        time = dataset["time"].values

        # Calculate time quantiles separately for censored and uncensored to prevent
        # the censoring distribution from skewing the event time bins.
        time_bins = pd.Series(index=dataset.index, dtype=float)
        for e in [0, 1]:
            mask = event == e
            if mask.sum() > 0:
                time_bins[mask] = pd.qcut(
                    time[mask], q=k_time_bins, labels=False, duplicates="drop"
                )

        stratify_label = [f"{e}_{t}" for e, t in zip(event, time_bins)]

        train_list = []
        test_list = []
        for train_ix, test_ix in kf.split(dataset, stratify_label):
            train_list.append(train_ix.tolist())
            test_list.append(test_ix.tolist())

        json_dict = {
            "train": train_list,
            "test": test_list,
        }
        with open(split_paths[data_ix], "w") as fp:
            json.dump(json_dict, fp)
    return 0


status = main(
    data_paths=snakemake.input,
    split_paths=snakemake.output,
    seed=snakemake.params["seed"],
    n_splits=snakemake.params["n_splits"],
    do_shuffle=snakemake.params["do_shuffle"],
    n_repeats=snakemake.params["n_repeats"],
    k_time_bins=snakemake.params["k_time_bins"],
)
