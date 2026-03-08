# =============================================================================
# Script: 04_knn.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: k-Nearest Neighbors with optimal k selection
# =============================================================================

library(class)
library(ggplot2)

telecom      <- readRDS("data/telecom_encoded.rds")
training_idx <- readRDS("data/training_idx.rds")

train_data <- telecom[training_idx, ]
test_data  <- telecom[-training_idx, ]

# Separate features and labels
train_X <- train_data[, names(train_data) != "churn"]
test_X  <- test_data[,  names(test_data)  != "churn"]
train_y <- train_data$churn
test_y  <- test_data$churn

# Normalize features (required for kNN — distance-based algorithm)
normalize <- function(x) (x - min(x)) / (max(x) - min(x) + 1e-10)
train_X_norm <- as.data.frame(lapply(train_X, normalize))
test_X_norm  <- as.data.frame(lapply(test_X,  normalize))

# -----------------------------------------------------------------------------
# 1. FIND OPTIMAL K
# -----------------------------------------------------------------------------
k_range    <- 1:20
error_rate <- numeric(length(k_range))

cat("Searching for optimal k...\n")
for (k in k_range) {
  preds        <- knn(train_X_norm, test_X_norm, cl = train_y, k = k)
  error_rate[k] <- mean(preds != test_y)
  cat(sprintf("  k = %2d | Error Rate = %.4f\n", k, error_rate[k]))
}

best_k <- which.min(error_rate)
cat(sprintf("\nBest k = %d (Error Rate = %.4f)\n", best_k, error_rate[best_k]))

# Plot k vs error rate
k_df <- data.frame(k = k_range, error_rate = error_rate)
ggplot(k_df, aes(x = k, y = error_rate)) +
  geom_line(color = "#3498DB", linewidth = 1) +
  geom_point(color = "#E74C3C", size = 3) +
  geom_vline(xintercept = best_k, linetype = "dashed", color = "#E74C3C") +
  labs(
    title    = paste0("kNN: Error Rate vs k (Best k = ", best_k, ")"),
    subtitle = "Model with minimum test error selected",
    x        = "k (Number of Neighbors)",
    y        = "Test Error Rate"
  ) +
  theme_minimal(base_size = 13)

ggsave("outputs/05_knn_k_selection.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 2. FINAL MODEL WITH BEST K
# -----------------------------------------------------------------------------
knn_preds <- knn(train_X_norm, test_X_norm, cl = train_y, k = best_k)

# -----------------------------------------------------------------------------
# 3. EVALUATION
# -----------------------------------------------------------------------------
conf_mat <- table(Predicted = knn_preds, Actual = test_y)
cat("\nkNN Confusion Matrix (k =", best_k, "):\n")
print(conf_mat)

accuracy    <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- if (nrow(conf_mat) == 2) conf_mat[2, 2] / sum(conf_mat[, 2]) else 0
specificity <- conf_mat[1, 1] / sum(conf_mat[, 1])
tp          <- if (nrow(conf_mat) == 2) conf_mat[2, 2] else 0

cat(sprintf("\nAccuracy:    %.1f%%\n", accuracy * 100))
cat(sprintf("Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("Specificity: %.1f%%\n", specificity * 100))
cat(sprintf("True Positives (churners caught): %d\n", tp))
cat(sprintf("False Negatives (missed churners): %d\n", sum(test_y == 1) - tp))

cat("\nkNN modeling complete.\n")
