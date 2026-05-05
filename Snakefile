import glob
import os

configfile: "config/config.yaml"

ESS_RATIOS = config["ess_ratios"]

DATASETS = config["datasets"]


include: "workflow/rules/download_data.smk"
include: "workflow/rules/create_splits.smk"
include: "workflow/rules/create_distribution_shift.smk"
include: "workflow/rules/train_eval.smk"


rule all:
    input:
        expand(
            "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv",
            dataset=DATASETS,
            split_ix=[i for i in range(config["n_splits"])],
            ess_ratio=ESS_RATIOS
        )
