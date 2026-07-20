"""Export the GBSG test fold as-is (natural Rot→GBSG shift; no adversarial tilt)."""
import json
import sys

import pandas as pd

sys.stderr = sys.stdout = open(snakemake.log[0], "w")


def main(data_path, splits_path, split_ix, test_out):
    data = pd.read_csv(data_path)
    with open(splits_path) as fp:
        splits = json.load(fp)

    test_idx = splits["test"][int(split_ix)]
    if not test_idx:
        raise ValueError(f"Empty test split for split_ix={split_ix}.")

    test = data.iloc[test_idx].copy()
    test.insert(0, "dataset_index", test_idx)
    # Sentinel: no adversarial tilt feature; train scripts treat this as natural shift.
    test["tilt_feature"] = "none"

    test.to_csv(test_out, sep="\t", index=False)
    print(
        f"Wrote natural test split {split_ix}: {len(test)} rows -> {test_out} "
        "(no oracle weights)."
    )


main(
    data_path=snakemake.input["data"],
    splits_path=snakemake.input["splits"],
    split_ix=snakemake.params["split_ix"],
    test_out=snakemake.output["test"],
)
