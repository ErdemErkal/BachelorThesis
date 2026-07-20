# Setup logging
log_file <- file(snakemake@log[[1]], open = "wt")
sink(log_file, type = "message")
sink(log_file, type = "output")

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
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

generate_tilted_data <- function(csv_path,
                                 json_path,
                                 split_ix,
                                 target_ess_ratio,
                                 output_path,
                                 tilt_base_coefs = c(1, -1, 1),
                                 seed = 42) {
  set.seed(seed)
  
  # 1. Load data and JSON splits
  full_data <- vroom(csv_path)
  splits <- fromJSON(file = json_path)
  
  # 2. Extract indices
  idx_train <- unlist(splits$train[[split_ix + 1]]) + 1
  idx_test  <- unlist(splits$test[[split_ix + 1]]) + 1
  
  data_train_raw <- full_data[idx_train, ]
  data_test_pool_raw <- full_data[idx_test, ]
  
  # 3. Handle One-Hot Encoding
  target_cols <- c("time", "status")
  feature_names <- setdiff(colnames(full_data), target_cols)
  formula_str <- as.formula(paste("~ 0 +", paste(feature_names, collapse = " + ")))
  
  train_mat_raw <- model.matrix(formula_str, data = data_train_raw)
  test_pool_mat_raw <- model.matrix(formula_str, data = data_test_pool_raw)
  
  # Align columns
  common_cols <- intersect(colnames(train_mat_raw), colnames(test_pool_mat_raw))
  test_aligned <- matrix(0, nrow = nrow(data_test_pool_raw), ncol = ncol(train_mat_raw))
  colnames(test_aligned) <- colnames(train_mat_raw)
  test_aligned[, common_cols] <- test_pool_mat_raw[, common_cols]
  
  # 4. Scaling
  train_mat <- scale(train_mat_raw)
  train_center <- attr(train_mat, "scaled:center")
  train_scale <- attr(train_mat, "scaled:scale")
  test_pool_mat <- scale(test_aligned, center = train_center, scale = train_scale)
  
  # 5. Feature Selection
  cox_model <- cv.glmnet(
    train_mat,
    Surv(data_train_raw$time, data_train_raw$status),
    family = "cox",
    alpha = 0
  )
  coeffs <- as.matrix(coef(cox_model, s = "lambda.min"))
  valid_coeffs <- coeffs[!is.na(coeffs[,1]), 1, drop=FALSE]
  top_covs <- names(sort(abs(valid_coeffs[, 1]), decreasing = TRUE))
  top_covs <- top_covs[1:min(length(top_covs), length(tilt_base_coefs))]
  
  message("Selected top covariates for tilting: ", paste(top_covs, collapse = ", "))
  
  current_tilt_coefs <- tilt_base_coefs[1:length(top_covs)]
  
  # 6. Solve for Gamma with Error Handling
  train_scores <- train_mat[, top_covs, drop = FALSE] %*% matrix(current_tilt_coefs, ncol = 1)
  
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
  
  # 7. Apply shift with rejection sampling (no replacement).
  test_pool_scores <- test_pool_mat[, top_covs, drop = FALSE] %*% matrix(current_tilt_coefs, ncol = 1)
  
  # Re-apply numerical stability for weight and acceptance calculations.
  shifted_test_scores <- gamma_opt * test_pool_scores
  raw_test_weights <- exp(shifted_test_scores - max(shifted_test_scores))

  # Rejection sampling acceptance probabilities are naturally in [0, 1].
  acceptance_probs <- as.numeric(raw_test_weights)
  uniform_draws <- runif(n = nrow(data_test_pool_raw))
  accepted_indices <- which(uniform_draws <= acceptance_probs)

  shifted_output <- data_test_pool_raw[accepted_indices, ]

  # Normalize to E_source[w(X)] ~= 1 so oracle weights stay on the same
  # importance-ratio scale expected by weighted conformal prediction.
  oracle_density_weights <- raw_test_weights / mean(raw_test_weights)
  shifted_output$true_density_weight <- as.numeric(
    oracle_density_weights[accepted_indices]
  )

  if (nrow(shifted_output) < 50) {
    stop(
      paste0(
        "Shifted test set is too small for reliable evaluation (",
        nrow(shifted_output),
        " rows). Re-run with a milder shift or larger base test pool."
      )
    )
  }

  vroom_write(shifted_output, output_path, delim = "\t")
  message("Shifted data written to: ", output_path)
}

generate_tilted_data(
  csv_path = snakemake@input[["data"]],
  json_path = snakemake@input[["splits"]],
  split_ix = as.numeric(snakemake@params[["split_ix"]]),
  target_ess_ratio = as.numeric(snakemake@wildcards[["ess_ratio"]]),
  output_path = snakemake@output[["tilted_test"]],
  seed = as.numeric(snakemake@params[["seed"]])
)

sink(type = "message"); sink(type = "output")
