rule download_data:
    output:
        expand("results/datasets/{ds}.csv", ds=DATASETS)
    log:
        "logs/clone_repo.log"
    params:
        repo_url=config["repo_url"],
        repo_commit=config["repo_commit"]
    shell:
        """
        git clone {params.repo_url} temp_repo > {log} 2>&1
        cd temp_repo
        git checkout {params.repo_commit} >> ../{log} 2>&1
        cd ..
        
        mkdir -p results/datasets >> {log} 2>&1
        for ds in {DATASETS}; do
            cp temp_repo/datasets/$ds.csv results/datasets/ >> {log} 2>&1
        done
        rm -rf temp_repo >> {log} 2>&1
        """