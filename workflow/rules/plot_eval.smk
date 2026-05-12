rule plot_eval_results:
    input:
        EVAL_TARGETS
    output:
        PLOT_TARGETS
    log:
        "logs/plot_eval_results.log"
    params:
        input_dir="results/eval_metrics",
        output_dir=PLOT_OUTPUT_DIR,
        metrics=PLOT_METRIC_ARGS,
        output_format=PLOT_FORMAT
    conda:
        "../envs/py/plot.yaml"
    shell:
        """
        python workflow/scripts/py/plot_eval_results.py \
            --input-dir {params.input_dir} \
            --output-dir {params.output_dir} \
            --format {params.output_format} \
            {params.metrics} > {log} 2>&1
        """
