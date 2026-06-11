configfile: "config/config.yaml"

rule create_distribution_shift:
    input:
        data="results/datasets/{dataset}.csv",
        splits="results/create_splits/{dataset}.json"
    output:
        tilted_test = "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv"
    params:
        split_ix = lambda wildcards: int(wildcards.split_ix),
        subsample_ratio = config.get("subsample_ratio", 0.5),
        seed = config["seed"]
    log:
        "logs/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.log"
    conda:
        "../envs/r/create_distribution_shift.yaml"
    script:
        "../scripts/r/create_distribution_shift.R"

