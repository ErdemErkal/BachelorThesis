configfile: "config/config.yaml"

rule create_distribution_shift:
    input:
        data="results/datasets/{dataset}.csv",
        splits="results/create_splits/{dataset}.json"
    output:
        test_with_weights = "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.tsv",
        oracle_weights = "results/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}_weights.tsv"
    params:
        split_ix = lambda wildcards: int(wildcards.split_ix),
        seed = config["seed"]
    log:
        "logs/create_distribution_shift/{dataset}_split_{split_ix}_ess_{ess_ratio}.log"
    conda:
        "../envs/r/create_distribution_shift.yaml"
    script:
        "../scripts/r/create_distribution_shift.R"

