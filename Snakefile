configfile: "config/config.yaml"

ESS_RATIOS = config["ess_ratios"]
DATASETS = config["datasets"]
SPLIT_IDS = range(config["n_splits"] * config["n_repeats"])
SEEDS = config.get("seeds", [config["seed"]])
MODELS = config.get("models", ["AFT"])
PLOT_METRICS = config.get("plot_metrics", ["D_CAL_pvalue", "WS_cal"])
PLOT_FORMAT = config.get("plot_format", "png")
PLOT_OUTPUT_DIR = "results/plots/eval_violin"
PLOT_METRIC_ARGS = " ".join(f"--metric {metric}" for metric in PLOT_METRICS)

DATASET_OUTPUT = "results/datasets/{dataset}.csv"
SPLIT_OUTPUT = "results/create_splits/{dataset}.json"
SHIFT_OUTPUT = "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv"
TRAIN_OUTPUT = "results/survival_outputs/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.npz"
EVAL_OUTPUT = "results/eval_metrics/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.csv"
PLOT_OUTPUT = PLOT_OUTPUT_DIR + "/{model}_pooled_{metric}_violin.{plot_format}"

DATASET_TARGETS = expand(
    DATASET_OUTPUT,
    dataset=DATASETS,
)

SPLIT_TARGETS = expand(
    SPLIT_OUTPUT,
    dataset=DATASETS,
)

SHIFT_TARGETS = expand(
    SHIFT_OUTPUT,
    dataset=DATASETS,
    split_ix=SPLIT_IDS,
    ess_ratio=ESS_RATIOS,
)

FROZEN_DATA_TARGETS = DATASET_TARGETS + SPLIT_TARGETS + SHIFT_TARGETS

TRAIN_TARGETS = expand(
    TRAIN_OUTPUT,
    dataset=DATASETS,
    split_id=SPLIT_IDS,
    ess=ESS_RATIOS,
    seed=SEEDS,
    model=MODELS,
)

EVAL_TARGETS = expand(
    EVAL_OUTPUT,
    dataset=DATASETS,
    split_id=SPLIT_IDS,
    ess=ESS_RATIOS,
    seed=SEEDS,
    model=MODELS,
)

PLOT_TARGETS = expand(
    PLOT_OUTPUT,
    model=MODELS,
    metric=PLOT_METRICS,
    plot_format=[PLOT_FORMAT],
)


rule all:
    input:
        PLOT_TARGETS


rule frozen_data:
    input:
        FROZEN_DATA_TARGETS


rule shifts:
    input:
        SHIFT_TARGETS


rule train:
    input:
        TRAIN_TARGETS


rule eval:
    input:
        EVAL_TARGETS


rule plot:
    input:
        PLOT_TARGETS


include: "workflow/rules/train_eval.smk"
include: "workflow/rules/plot_eval.smk"
