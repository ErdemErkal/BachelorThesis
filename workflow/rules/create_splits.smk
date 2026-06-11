configfile: "config/config.yaml"


rule create_splits:
    input:
        "results/datasets/{dataset}.csv"
    output:
        "results/create_splits/{dataset}.json"
    params:
        seed=config["seed"],
        n_splits=config["n_splits"],
        do_shuffle=config["do_shuffle"],
        n_repeats=config["n_repeats"],
        k_time_bins=config.get("k_time_bins", 3)
    log:
        "logs/create_splits/{dataset}.log"
    conda:
        "../envs/py/base.yaml"
    script:
        "../scripts/py/make_splits.py"
