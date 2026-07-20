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
WEIGHTED_DENSITY_MODES = config.get(
    "weighted_density_modes", ["estimated", "oracle"]
)
PLOT_METRICS = config.get("plot_metrics", ["D_CAL_pvalue", "WS_cal"])
PLOT_FORMAT = config.get("plot_format", "png")
PLOT_OUTPUT_DIR = "results/plots/eval_violin"
PLOT_PATTERN = config.get("plot_pattern", "*.csv")
PLOT_METRIC_ARGS = " ".join(f"--metric {metric}" for metric in PLOT_METRICS)
PLOT_WEIGHTING_SUFFIX_ARGS = " ".join(
    f"--weighting-suffix {mode}={WEIGHTING_SUFFIXES[mode]}" for mode in WEIGHTING_MODES
)

DATASET_OUTPUT = "results/datasets/{dataset}.csv"
SPLIT_OUTPUT = "results/create_splits/{dataset}.json"
SHIFT_OUTPUT = (
    "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv"
)
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
        if weighting == "weighted":
            for density_mode in WEIGHTED_DENSITY_MODES:
                RUNS.append(
                    {
                        "model": model,
                        "weighting": weighting,
                        "density_mode": density_mode,
                        "label": (
                            f"{model}__dens_{density_mode}"
                            f"{WEIGHTING_SUFFIXES[weighting]}"
                        ),
                    }
                )
        else:
            RUNS.append(
                {
                    "model": model,
                    "weighting": weighting,
                    "density_mode": None,
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


def extra_train_args_for(wildcards):
    run = run_for(wildcards)
    if run["weighting"] != "weighted":
        return ""
    return f"--density-mode {run['density_mode']}"


EVAL_TARGETS = expand(
    EVAL_OUTPUT,
    dataset=DATASETS,
    split_id=SPLIT_IDS,
    ess=ESS_RATIOS,
    seed=SEEDS,
    run_label=RUN_LABELS,
)

CROSS_DATASET_NAME = config.get("cross_dataset_name", "rotterdam_gbsg")
CROSS_N_SPLITS = config.get("cross_n_splits", 5)
CROSS_SPLIT_IDS = range(CROSS_N_SPLITS)
CROSS_ESS = config.get("cross_ess_ratios", ["natural"])
# Natural shift has no oracle weights — weighted runs use estimated density only.
CROSS_RUN_LABELS = [
    run["label"]
    for run in RUNS
    if run["weighting"] == "unweighted" or run["density_mode"] == "estimated"
]
CROSS_EVAL_TARGETS = expand(
    EVAL_OUTPUT,
    dataset=[CROSS_DATASET_NAME],
    split_id=CROSS_SPLIT_IDS,
    ess=CROSS_ESS,
    seed=SEEDS,
    run_label=CROSS_RUN_LABELS,
)

PLOT_TARGETS = expand(
    PLOT_OUTPUT,
    model=MODELS,
    metric=PLOT_METRICS,
    plot_format=[PLOT_FORMAT],
) + expand(
    DATASET_PLOT_OUTPUT,
    model=MODELS,
    dataset=DATASETS,
    metric=PLOT_METRICS,
    plot_format=[PLOT_FORMAT],
)


rule all:
    input:
        EVAL_TARGETS,
        CROSS_EVAL_TARGETS,


include: "workflow/rules/download_data.smk"
include: "workflow/rules/create_splits.smk"
include: "workflow/rules/create_distribution_shift.smk"
include: "workflow/rules/train_eval.smk"
include: "workflow/rules/cross_dataset.smk"
