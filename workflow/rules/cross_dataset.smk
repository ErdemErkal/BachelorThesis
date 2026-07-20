CROSS_DATASET_NAME = config.get("cross_dataset_name", "rotterdam_gbsg")
CROSS_N_SPLITS = config.get("cross_n_splits", 5)
CROSS_SPLIT_IDS = range(CROSS_N_SPLITS)
# Natural domain shift only — no ESS adversarial grid.
CROSS_ESS = config.get("cross_ess_ratios", ["natural"])


# Load survival::rotterdam + survival::gbsg and write a harmonized CSV.
rule prepare_cross_dataset_rotterdam_gbsg:
    output:
        unionized=f"results/datasets/{CROSS_DATASET_NAME}.csv",
    log:
        f"logs/prepare_cross_dataset/{CROSS_DATASET_NAME}.log",
    conda:
        "../envs/r/create_distribution_shift.yaml"
    script:
        "../scripts/r/prepare_cross_dataset_rotterdam_gbsg.R"


rule create_cross_splits_rotterdam_gbsg:
    input:
        f"results/datasets/{CROSS_DATASET_NAME}.csv"
    output:
        f"results/create_splits/{CROSS_DATASET_NAME}.json"
    params:
        seed=config["seed"],
        n_splits=CROSS_N_SPLITS,
        k_time_bins=config.get("k_time_bins", 3),
    log:
        f"logs/create_splits/{CROSS_DATASET_NAME}.log"
    conda:
        "../envs/py/base.yaml"
    script:
        "../scripts/py/make_cross_splits.py"


ruleorder: create_cross_splits_rotterdam_gbsg > create_splits


# Passthrough test fold — no adversarial tilt / no oracle weights.
rule create_cross_natural_test:
    input:
        data=f"results/datasets/{CROSS_DATASET_NAME}.csv",
        splits=f"results/create_splits/{CROSS_DATASET_NAME}.json",
    output:
        test=(
            "results/create_distribution_shift/"
            f"{CROSS_DATASET_NAME}_split_{{split_ix}}_ess_natural.tsv"
        ),
    params:
        split_ix=lambda wildcards: int(wildcards.split_ix),
    log:
        f"logs/create_distribution_shift/{CROSS_DATASET_NAME}_split_{{split_ix}}_ess_natural.log",
    conda:
        "../envs/py/base.yaml"
    script:
        "../scripts/py/prepare_natural_test_split.py"


ruleorder: create_cross_natural_test > create_distribution_shift


rule cross_dataset_train_eval:
    input:
        CROSS_EVAL_TARGETS
