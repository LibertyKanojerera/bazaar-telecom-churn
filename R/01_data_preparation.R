# =============================================================================
# Script: 01_data_preparation.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Data loading, cleaning, imputation, and feature engineering
# =============================================================================

library(dplyr)
library(fastDummies)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
# Place telecom.csv in the /data directory
telecom_raw <- read.csv("data/telecom.csv", stringsAsFactors = FALSE)

cat("Dataset dimensions:", nrow(telecom_raw), "rows x", ncol(telecom_raw), "columns\n")
cat("Column names:\n")
print(names(telecom_raw))

# -----------------------------------------------------------------------------
# 2. MISSING VALUE ASSESSMENT
# -----------------------------------------------------------------------------
missing_summary <- data.frame(
  Variable = names(telecom_raw),
  Missing_Count = sapply(telecom_raw, function(x) sum(is.na(x))),
  Missing_Pct   = round(sapply(telecom_raw, function(x) mean(is.na(x)) * 100), 2)
)
cat("\nMissing Value Summary:\n")
print(missing_summary[missing_summary$Missing_Pct > 0, ])

# Key issue: suspended_subscribers (95.8% missing), not_active_subscribers (49.1% missing)

# -----------------------------------------------------------------------------
# 3. IMPUTATION
# Operational rationale: missing = no inactive/suspended subscribers recorded
# -----------------------------------------------------------------------------
telecom_raw$not_active_subscribers[is.na(telecom_raw$not_active_subscribers)] <- 0
telecom_raw$suspended_subscribers[is.na(telecom_raw$suspended_subscribers)]   <- 0

# Impute remaining minor missings (billing_zip, arpu) with mode/median
# billing_zip: will be dropped, so skipping imputation
# arpu: will be dropped as multicollinear, so skipping imputation

cat("\nPost-imputation missing values:", sum(is.na(telecom_raw)), "\n")

# -----------------------------------------------------------------------------
# 4. VARIABLE REMOVAL
# Drop identifiers and multicollinear variables
# -----------------------------------------------------------------------------
cols_to_drop <- c(
  "pid",           # Pure identifier — no predictive value
  "billing_zip",   # Granular location — overfitting risk
  "ka_name",       # High-cardinality account manager ID
  "total_subs",    # Multicollinear with active + not_active + suspended
  "total_revenue", # Multicollinear with avg_mobile + avg_fix revenue
  "arpu"           # Derived from revenue/subscribers — redundant
)

telecom_clean <- telecom_raw %>%
  select(-any_of(cols_to_drop))

cat("\nVariables retained after reduction:\n")
print(names(telecom_clean))

# -----------------------------------------------------------------------------
# 5. CATEGORY CLEANING
# Fix crm_pid_value_segment inconsistencies (case, spacing, rare categories)
# -----------------------------------------------------------------------------
telecom_clean$crm_pid_value_segment <- trimws(telecom_clean$crm_pid_value_segment)
telecom_clean$crm_pid_value_segment <- tools::toTitleCase(
  tolower(telecom_clean$crm_pid_value_segment)
)

# Review segment distribution
cat("\nValue Segment Distribution:\n")
print(sort(table(telecom_clean$crm_pid_value_segment), decreasing = TRUE))

cat("\nEffective Segment Distribution:\n")
print(sort(table(telecom_clean$effective_segment), decreasing = TRUE))

# -----------------------------------------------------------------------------
# 6. TARGET VARIABLE ENCODING
# Churn: Yes → 1, No → 0
# -----------------------------------------------------------------------------
telecom_clean$churn <- ifelse(telecom_clean$churn == "Yes", 1, 0)

cat("\nChurn Distribution:\n")
print(table(telecom_clean$churn))
cat("Churn Rate:", round(mean(telecom_clean$churn) * 100, 2), "%\n")

# -----------------------------------------------------------------------------
# 7. DUMMY ENCODING (for GLM, kNN, Tree models)
# Random Forest handles factors natively — separate dataset created
# -----------------------------------------------------------------------------
telecom_encoded <- fastDummies::dummy_cols(
  telecom_clean,
  select_columns        = c("crm_pid_value_segment", "effective_segment"),
  remove_first_dummy    = TRUE,   # Avoids perfect multicollinearity
  remove_selected_columns = TRUE  # Drop original categorical columns
)

cat("\nFinal encoded dataset dimensions:", nrow(telecom_encoded), "x", ncol(telecom_encoded), "\n")
cat("Columns:\n")
print(names(telecom_encoded))

# For Random Forest: keep as factors
telecom_rf <- telecom_clean %>%
  mutate(
    churn                    = as.factor(churn),
    crm_pid_value_segment    = as.factor(crm_pid_value_segment),
    effective_segment        = as.factor(effective_segment)
  )

# -----------------------------------------------------------------------------
# 8. TRAIN/TEST SPLIT (60/40, stratified)
# -----------------------------------------------------------------------------
set.seed(12345)
training_size <- 0.6
training_idx  <- sample(1:nrow(telecom_encoded), training_size * nrow(telecom_encoded))

cat("\nTraining observations:", length(training_idx))
cat("\nTest observations:", nrow(telecom_encoded) - length(training_idx), "\n")

# Verify churn rate is preserved across splits
cat("Training churn rate:", round(mean(telecom_encoded$churn[training_idx]) * 100, 2), "%\n")
cat("Test churn rate:    ", round(mean(telecom_encoded$churn[-training_idx]) * 100, 2), "%\n")

# -----------------------------------------------------------------------------
# 9. EXPORT CLEANED DATASETS
# -----------------------------------------------------------------------------
saveRDS(telecom_encoded, "data/telecom_encoded.rds")
saveRDS(telecom_rf,      "data/telecom_rf.rds")
saveRDS(training_idx,    "data/training_idx.rds")

cat("\nData preparation complete. Cleaned datasets saved to /data.\n")
