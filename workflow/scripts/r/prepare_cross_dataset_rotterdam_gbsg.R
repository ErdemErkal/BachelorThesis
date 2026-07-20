# Load Rotterdam + GBSG from the survival package and harmonize to a shared schema.
# Size is categorical in Rotterdam and continuous (mm) in GBSG; bin GBSG to match.
suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(readr)
  library(stringr)
})

output_path <- snakemake@output[["unionized"]]

data(rotterdam, package = "survival", envir = environment())
data(gbsg, package = "survival", envir = environment())

size_levels <- c("<=20", "20-50", ">50")

normalize_size_factor <- function(x) {
  x_chr <- str_replace_all(as.character(x), "\\s+", "")
  x_chr <- dplyr::case_when(
    x_chr %in% c("<=20", "<20", "0-20", "1-20") ~ "<=20",
    x_chr %in% c("20-50", "21-50") ~ "20-50",
    x_chr %in% c(">50", ">=50", "50+") ~ ">50",
    TRUE ~ x_chr
  )
  factor(x_chr, levels = size_levels)
}

# Same cutpoints as survival vignette / Royston & Altman validation.
bin_size_mm <- function(size_mm) {
  size_mm <- as.numeric(size_mm)
  if (any(!is.finite(size_mm))) {
    stop("GBSG size contains non-finite values after numeric conversion.")
  }
  cut(
    size_mm,
    breaks = c(-Inf, 20, 50, Inf),
    labels = size_levels,
    right = TRUE
  )
}

# Recurrence-free endpoint aligned with GBSG status (RFS).
rotterdam_std <- tibble(
  age = as.numeric(rotterdam$age),
  meno = as.integer(rotterdam$meno),
  size = normalize_size_factor(rotterdam$size),
  grade = as.integer(rotterdam$grade),
  nodes = as.numeric(rotterdam$nodes),
  pgr = as.numeric(rotterdam$pgr),
  er = as.numeric(rotterdam$er),
  hormon = as.integer(rotterdam$hormon),
  time = as.numeric(pmin(rotterdam$rtime, rotterdam$dtime)),
  status = as.integer(pmax(rotterdam$recur, rotterdam$death)),
  source_dataset = "rotterdam"
)

gbsg_std <- tibble(
  age = as.numeric(gbsg$age),
  meno = as.integer(gbsg$meno),
  size = bin_size_mm(gbsg$size),
  grade = as.integer(gbsg$grade),
  nodes = as.numeric(gbsg$nodes),
  pgr = as.numeric(gbsg$pgr),
  er = as.numeric(gbsg$er),
  hormon = as.integer(gbsg$hormon),
  time = as.numeric(gbsg$rfstime),
  status = as.integer(gbsg$status),
  source_dataset = "gbsg"
)

# Drop Rotterdam-only fields (year, chemo, pid, …) by construction.
unionized <- bind_rows(rotterdam_std, gbsg_std)

if (any(is.na(unionized$size))) {
  stop("Size harmonization produced NA values; check size encodings.")
}
if (any(!is.finite(unionized$time)) || any(unionized$time <= 0)) {
  stop("Invalid survival times after harmonization.")
}
if (!all(unionized$status %in% c(0L, 1L))) {
  stop("Event indicators must be binary 0/1 after harmonization.")
}

message(
  "Loaded survival::rotterdam + survival::gbsg. Unionized rows: ", nrow(unionized),
  " (rotterdam=", sum(unionized$source_dataset == "rotterdam"),
  ", gbsg=", sum(unionized$source_dataset == "gbsg"), ")"
)
message("Shared covariates: age, meno, size, grade, nodes, pgr, er, hormon")
message("Size levels: ", paste(levels(unionized$size), collapse = ", "))

write_csv(unionized, output_path)
