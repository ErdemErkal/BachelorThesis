configfile: "config/config.yaml"

DEFAULT_TRAIN_SCRIPTS = {
    "unweighted": "workflow/scripts/py/single_split_train.py",
    "weighted": "workflow/scripts/py/single_split_train_weighted.py",
}
DEFAULT_WEIGHTING_SUFFIXES = {
    "unweighted": "",
    "weighted": "_weighted",
}
TRAIN_SCRIPTS = config.get("train_scripts", DEFAULT_TRAIN_SCRIPTS)
WEIGHTING_SUFFIXES = config.get("weighting_suffixes", DEFAULT_WEIGHTING_SUFFIXES)

ESS_RATIOS = config["ess_ratios"]
DATASETS = config["datasets"]
SPLIT_IDS = range(config["n_splits"] * config["n_repeats"])
SEEDS = config.get("seeds", [config["seed"]])
MODELS = config.get("models", ["AFT"])
WEIGHTING_MODES = config.get("weighting_modes", list(TRAIN_SCRIPTS.keys()))
PLOT_METRICS = config.get("plot_metrics", ["D_CAL_pvalue", "WS_cal"])
PLOT_FORMAT = config.get("plot_format", "png")
PLOT_OUTPUT_DIR = "results/plots/eval_violin"
PLOT_PATTERN = config.get("plot_pattern", "*.csv")
PLOT_METRIC_ARGS = " ".join(f"--metric {metric}" for metric in PLOT_METRICS)
PLOT_WEIGHTING_SUFFIX_ARGS = " ".join(
    f"--weighting-suffix {mode}={WEIGHTING_SUFFIXES[mode]}"
    for mode in WEIGHTING_MODES
)

DATASET_OUTPUT = "results/datasets/{dataset}.csv"
SPLIT_OUTPUT = "results/create_splits/{dataset}.json"
SHIFT_OUTPUT = "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv"
TRAIN_OUTPUT = (
    "results/survival_outputs/"
    "{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{run_label}.npz"
)
EVAL_OUTPUT = (
    "results/eval_metrics/"
    "{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{run_label}.csv"
)
PLOT_OUTPUT = PLOT_OUTPUT_DIR + "/{model}_pooled_{metric}_violin.{plot_format}"
DATASET_PLOT_OUTPUT = (
    PLOT_OUTPUT_DIR + "/{model}_{dataset}_{metric}_violin.{plot_format}"
)

wildcard_constraints:
    split_id=r"\d+",
    seed=r"\d+",


RUNS = []
for model in MODELS:
    for weighting in WEIGHTING_MODES:
        if weighting not in WEIGHTING_SUFFIXES:
            raise ValueError(
                f"No filename suffix configured for weighting mode '{weighting}'."
            )
        RUNS.append(
            {
                "model": model,
                "weighting": weighting,
                "label": f"{model}{WEIGHTING_SUFFIXES[weighting]}",
            }
        )

RUN_LABELS = [run["label"] for run in RUNS]
RUN_BY_LABEL = {run["label"]: run for run in RUNS}
if len(RUN_BY_LABEL) != len(RUN_LABELS):
    raise ValueError("Duplicate run labels generated from models and weighting modes.")


def run_for(wildcards):
    if wildcards.run_label not in RUN_BY_LABEL:
        raise ValueError(f"No run configured for label '{wildcards.run_label}'.")
    return RUN_BY_LABEL[wildcards.run_label]


def train_script_for(wildcards):
    weighting = run_for(wildcards)["weighting"]
    if weighting not in TRAIN_SCRIPTS:
        raise ValueError(
            f"No train script configured for weighting mode '{weighting}'."
        )
    return TRAIN_SCRIPTS[weighting]


def model_for(wildcards):
    return run_for(wildcards)["model"]


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
    run_label=RUN_LABELS,
)

EVAL_TARGETS = expand(
    EVAL_OUTPUT,
    dataset=DATASETS,
    split_id=SPLIT_IDS,
    ess=ESS_RATIOS,
    seed=SEEDS,
    run_label=RUN_LABELS,
)

POOLED_PLOT_TARGETS = expand(
    PLOT_OUTPUT,
    model=MODELS,
    metric=PLOT_METRICS,
    plot_format=[PLOT_FORMAT],
)

DATASET_PLOT_TARGETS = expand(
    DATASET_PLOT_OUTPUT,
    model=MODELS,
    dataset=DATASETS,
    metric=PLOT_METRICS,
    plot_format=[PLOT_FORMAT],
)

PLOT_TARGETS = POOLED_PLOT_TARGETS + DATASET_PLOT_TARGETS


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
