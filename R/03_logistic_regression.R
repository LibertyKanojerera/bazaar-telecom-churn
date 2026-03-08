# =============================================================================
# Script: 03_logistic_regression.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Logistic regression model — train, evaluate, interpret
# =============================================================================

library(caret)

telecom      <- readRDS("data/telecom_encoded.rds")
training_idx <- readRDS("data/training_idx.rds")

train_data <- telecom[training_idx, ]
test_data  <- telecom[-training_idx, ]

# -----------------------------------------------------------------------------
# 1. FIT MODEL
# -----------------------------------------------------------------------------
logit_model <- glm(
  churn ~ .,
  data   = train_data,
  family = binomial(link = "logit")
)

cat("Logistic Regression Summary:\n")
print(summary(logit_model))

# -----------------------------------------------------------------------------
# 2. PREDICT ON TEST SET
# -----------------------------------------------------------------------------
# Default threshold: 0.5
logit_probs  <- predict(logit_model, newdata = test_data, type = "response")
logit_preds  <- ifelse(logit_probs >= 0.5, 1, 0)

# -----------------------------------------------------------------------------
# 3. CONFUSION MATRIX & METRICS
# -----------------------------------------------------------------------------
conf_mat <- table(Predicted = logit_preds, Actual = test_data$churn)
cat("\nLogistic Regression Confusion Matrix:\n")
print(conf_mat)

accuracy    <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- conf_mat[2, 2] / sum(conf_mat[, 2])   # TP / (TP + FN)
specificity <- conf_mat[1, 1] / sum(conf_mat[, 1])   # TN / (TN + FP)

cat(sprintf("\nAccuracy:    %.1f%%\n", accuracy * 100))
cat(sprintf("Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("Specificity: %.1f%%\n", specificity * 100))
cat(sprintf("True Positives (churners caught): %d\n", conf_mat[2, 2]))

# -----------------------------------------------------------------------------
# 4. THRESHOLD SENSITIVITY ANALYSIS
# At what threshold do we start catching churners?
# -----------------------------------------------------------------------------
cat("\n--- Threshold Sensitivity Analysis ---\n")
thresholds <- c(0.5, 0.3, 0.2, 0.1, 0.07)

for (thresh in thresholds) {
  preds <- ifelse(logit_probs >= thresh, 1, 0)
  cm    <- table(Predicted = preds, Actual = test_data$churn)
  tp    <- if (nrow(cm) == 2) cm[2, 2] else 0
  sens  <- tp / sum(test_data$churn == 1)
  cat(sprintf("Threshold = %.2f | True Positives = %d | Sensitivity = %.1f%%\n",
              thresh, tp, sens * 100))
}

# -----------------------------------------------------------------------------
# 5. SIGNIFICANT PREDICTORS
# -----------------------------------------------------------------------------
coef_df <- as.data.frame(summary(logit_model)$coefficients)
coef_df <- coef_df[order(coef_df[, 4]), ]
cat("\nTop predictors by p-value:\n")
print(head(coef_df, 10))

cat("\nLogistic Regression modeling complete.\n")
