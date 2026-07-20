# Setup logging
log_file <- file(snakemake@log[[1]], open = "wt")
sink(log_file, type = "message")
sink(log_file, type = "output")

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(vroom)
  library(rjson)
})

#' Calculate Effective Sample Size (ESS) Ratio with Numerical Stability
#' Based on heuristic: (sum(w)^2) / sum(w^2) [cite: 261]
get_ess_ratio <- function(gamma, scores, n) {
  # Subtract max score for numerical stability in exp()
  # This prevents Inf values while maintaining the same ratio
  shifted_scores <- gamma * scores
  max_s <- max(shifted_scores)
  weights <- exp(shifted_scores - max_s)
  
  ess <- (sum(weights)^2) / sum(weights^2)
  return(ess / n)
}

safe_zscore <- function(values, center, scale) {
  if (!is.finite(scale) || scale <= 0) {
    return(rep(0, length(values)))
  }
  z <- (values - center) / scale
  z[!is.finite(z)] <- 0
  z
}

generate_tilted_data <- function(csv_path,
                                 json_path,
                                 split_ix,
                                 target_ess_ratio,
                                 output_path,
                                 seed = 42) {
  set.seed(seed)
  
  # 1. Load data and JSON splits
  full_data <- vroom(csv_path)
  splits <- fromJSON(file = json_path)
  
  # 2. Extract indices
  idx_train <- unlist(splits$train[[split_ix + 1]]) + 1
  idx_test  <- unlist(splits$test[[split_ix + 1]]) + 1

  # Align with benchmark / Python naming; never use split-source labels as features.
  if ("event" %in% names(full_data) && !("status" %in% names(full_data))) {
    full_data <- full_data %>% rename(status = event)
  }
  drop_meta <- intersect(c("source_dataset", "dataset_index"), names(full_data))
  if (length(drop_meta) > 0) {
    full_data <- full_data %>% select(-all_of(drop_meta))
  }
  
  data_train_raw <- full_data[idx_train, ]
  data_test_pool_raw <- full_data[idx_test, ]
  
  # 3. Adversarial Error Tilt (fit only on training pool to avoid leakage).
  cox_base <- coxph(
    Surv(time, status) ~ . - time - status,
    data = data_train_raw
  )

  abs_errors <- abs(residuals(cox_base, type = "deviance"))
  error_lm <- lm(abs_errors ~ . - time - status, data = data_train_raw)

  error_coefs <- coef(error_lm)[-1]
  error_coefs[is.na(error_coefs)] <- 0

  if (length(error_coefs) == 0) {
    stop("Adversarial error model produced no usable coefficients.")
  }

  # Restrict to truly continuous features (> 10 unique values).
  min_unique_values <- 10

  continuous_vars <- names(data_train_raw)[
    sapply(data_train_raw, function(x) length(unique(x)) >= min_unique_values)
  ]
  continuous_vars <- setdiff(continuous_vars, c("time", "status"))

  continuous_coefs <- error_coefs[names(error_coefs) %in% continuous_vars]
  continuous_coefs[is.na(continuous_coefs)] <- 0

  if (length(continuous_coefs) == 0) {
    stop(
      "Adversarial error model produced no usable continuous coefficients (>= 10 unique values)."
    )
  }

  strongest_feature <- names(continuous_coefs)[which.max(abs(continuous_coefs))]
  message(
    "Strongest continuous feature driving error: ",
    strongest_feature,
    " (coef: ",
    continuous_coefs[strongest_feature],
    ")"
  )

  masked_coefs <- rep(0, length(error_coefs))
  names(masked_coefs) <- names(error_coefs)
  masked_coefs[strongest_feature] <- error_coefs[strongest_feature]
  error_coefs <- masked_coefs

  train_design <- model.matrix(error_lm)[, -1, drop = FALSE]
  terms_x <- delete.response(terms(error_lm))
  test_design_raw <- model.matrix(
    terms_x,
    data = data_test_pool_raw,
    xlev = .getXlevels(terms(error_lm), model.frame(error_lm))
  )
  if ("(Intercept)" %in% colnames(test_design_raw)) {
    test_design_raw <- test_design_raw[, colnames(test_design_raw) != "(Intercept)", drop = FALSE]
  }

  common_cols <- intersect(colnames(train_design), colnames(test_design_raw))
  test_design <- matrix(0, nrow = nrow(data_test_pool_raw), ncol = ncol(train_design))
  colnames(test_design) <- colnames(train_design)
  if (length(common_cols) > 0) {
    test_design[, common_cols] <- test_design_raw[, common_cols, drop = FALSE]
  }

  error_coefs <- error_coefs[colnames(train_design)]
  error_coefs[is.na(error_coefs)] <- 0

  raw_train_scores <- as.numeric(train_design %*% error_coefs)
  raw_test_scores <- as.numeric(test_design %*% error_coefs)

  # Stable z-score using the training score distribution.
  train_center <- mean(raw_train_scores)
  train_scale <- sd(raw_train_scores)
  if (!is.finite(train_scale) || train_scale <= 0) {
    train_scores <- rep(0, length(raw_train_scores))
    test_pool_scores <- rep(0, length(raw_test_scores))
  } else {
    train_scores <- (raw_train_scores - train_center) / train_scale
    test_pool_scores <- (raw_test_scores - train_center) / train_scale
    train_scores[!is.finite(train_scores)] <- 0
    test_pool_scores[!is.finite(test_pool_scores)] <- 0
  }
  message("Adversarial tilt uses ", length(error_coefs), " error-model features.")
  
  # 4. Solve for Gamma with Error Handling
  
  if (target_ess_ratio >= 1.0) {
    gamma_opt <- 0
  } else {
    # Narrow interval check to ensure uniroot doesn't hit numerical limits immediately
    opt_res <- uniroot(
      function(g) get_ess_ratio(g, train_scores, nrow(data_train_raw)) - target_ess_ratio,
      interval = c(0.0, 10.0), extendInt = "yes", tol = 1e-5
    )
    gamma_opt <- opt_res$root
  }
  
  # 5. Apply shift as weights only on fixed rows (no sampling).
  full_design_raw <- model.matrix(
    terms_x,
    data = full_data,
    xlev = .getXlevels(terms(error_lm), model.frame(error_lm))
  )
  if ("(Intercept)" %in% colnames(full_design_raw)) {
    full_design_raw <- full_design_raw[, colnames(full_design_raw) != "(Intercept)", drop = FALSE]
  }
  full_design <- matrix(0, nrow = nrow(full_data), ncol = ncol(train_design))
  colnames(full_design) <- colnames(train_design)
  common_full_cols <- intersect(colnames(full_design_raw), colnames(train_design))
  if (length(common_full_cols) > 0) {
    full_design[, common_full_cols] <- full_design_raw[, common_full_cols, drop = FALSE]
  }

  raw_full_scores <- as.numeric(full_design %*% error_coefs)
  full_scores <- safe_zscore(raw_full_scores, train_center, train_scale)

  shifted_full_scores <- gamma_opt * full_scores
  raw_full_weights <- exp(shifted_full_scores - max(shifted_full_scores))
  train_weight_mean <- mean(raw_full_weights[idx_train])
  if (!is.finite(train_weight_mean) || train_weight_mean <= 0) {
    stop("Oracle weights must have positive finite train-split mean.")
  }
  oracle_weights_full <- raw_full_weights / train_weight_mean

  # Write per-index oracle weights for calibration/test lookup in Python.
  weights_output_path <- if (grepl("\\.tsv$", output_path)) {
    sub("\\.tsv$", "_weights.tsv", output_path)
  } else {
    paste0(output_path, "_weights.tsv")
  }
  full_weights_df <- data.frame(
    dataset_index = seq_len(nrow(full_data)) - 1,
    oracle_weight = as.numeric(oracle_weights_full),
    tilt_feature = strongest_feature,
    split_ix = split_ix,
    ess_ratio = target_ess_ratio
  )
  vroom_write(full_weights_df, weights_output_path, delim = "\t")
  message("Oracle weights written to: ", weights_output_path)

  # Keep all test rows; downstream evaluation handles weighting directly.
  selected_pool_idx <- seq_len(nrow(data_test_pool_raw))

  shifted_output <- data_test_pool_raw[selected_pool_idx, , drop = FALSE]
  parent_dataset_idx <- idx_test[selected_pool_idx]
  shifted_output$dataset_index <- parent_dataset_idx - 1
  shifted_output$tilt_feature <- strongest_feature
  shifted_output$true_density_weight <- as.numeric(oracle_weights_full[parent_dataset_idx])

  vroom_write(shifted_output, output_path, delim = "\t")
  message("Test rows with oracle weights written to: ", output_path)
}

generate_tilted_data(
  csv_path = snakemake@input[["data"]],
  json_path = snakemake@input[["splits"]],
  split_ix = as.numeric(snakemake@params[["split_ix"]]),
  target_ess_ratio = as.numeric(snakemake@wildcards[["ess_ratio"]]),
  output_path = snakemake@output[["test_with_weights"]],
  seed = as.numeric(snakemake@params[["seed"]])
)

sink(type = "message"); sink(type = "output")
