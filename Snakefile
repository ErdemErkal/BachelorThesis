import glob
import os

configfile: "config/config.yaml"

ESS_RATIOS = config["ess_ratios"]

DATASETS = config["datasets"]

SPLIT_IDS = range(config["n_splits"])
SEEDS = config.get("seeds", [config["seed"]])
MODELS = config.get("models", ["AFT"])


include: "workflow/rules/download_data.smk"
include: "workflow/rules/create_splits.smk"
include: "workflow/rules/create_distribution_shift.smk"
include: "workflow/rules/train_eval.smk"


rule all:
    input:
        expand(
            "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv",
            dataset=DATASETS,
            split_ix=SPLIT_IDS,
            ess_ratio=ESS_RATIOS
        )


rule all_train:
    input:
        expand(
            "results/survival_outputs/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.npz",
            dataset=DATASETS,
            split_id=SPLIT_IDS,
            ess=ESS_RATIOS,
            seed=SEEDS,
            model=MODELS,
        )


rule all_eval:
    input:
        expand(
            "results/eval_metrics/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.csv",
            dataset=DATASETS,
            split_id=SPLIT_IDS,
            ess=ESS_RATIOS,
            seed=SEEDS,
            model=MODELS,
        )


rule all_train_eval:
    input:
        expand(
            "results/survival_outputs/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.npz",
            dataset=DATASETS,
            split_id=SPLIT_IDS,
            ess=ESS_RATIOS,
            seed=SEEDS,
            model=MODELS,
        ),
        expand(
            "results/eval_metrics/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.csv",
            dataset=DATASETS,
            split_id=SPLIT_IDS,
            ess=ESS_RATIOS,
            seed=SEEDS,
            model=MODELS,
        )
