rule single_split_train:
    input:
        dataset="results/datasets/{dataset}.csv",
        splits="results/create_splits/{dataset}.json",
        test_data="results/create_distribution_shift/{dataset}_split_{split_id}_ess_{ess}.tsv"
    output:
        npz="results/survival_outputs/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.npz"
    log:
        "logs/single_split_train/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.log"
    conda:
        "../envs/py/train.yaml"
    shell:
        """
        python workflow/scripts/py/single_split_train.py \
            --dataset {input.dataset} \
            --splits {input.splits} \
            --test-data {input.test_data} \
            --split-id {wildcards.split_id} \
            --ess {wildcards.ess} \
            --seed {wildcards.seed} \
            --model {wildcards.model} \
            --output {output.npz} > {log} 2>&1
        """

rule single_split_eval:
    input:
        npz="results/survival_outputs/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.npz"
    output:
        csv="results/eval_metrics/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.csv"
    log:
        "logs/single_split_eval/{dataset}_split_{split_id}_ess_{ess}_seed_{seed}_{model}.log"
    conda:
        "../envs/py/eval.yaml"
    shell:
        """
        python workflow/scripts/py/single_split_eval.py \
            --input {input.npz} \
            --seed {wildcards.seed} \
            --output-csv {output.csv} > {log} 2>&1
        """
