# =============================================================================
# Script: 05_classification_tree.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Classification tree with mindev hyperparameter tuning
# =============================================================================

library(tree)
library(ggplot2)

telecom      <- readRDS("data/telecom_encoded.rds")
training_idx <- readRDS("data/training_idx.rds")

train_data <- telecom[training_idx, ]
test_data  <- telecom[-training_idx, ]

# tree() requires factor target
train_data$churn <- as.factor(train_data$churn)
test_data$churn  <- as.factor(test_data$churn)

# -----------------------------------------------------------------------------
# 1. HYPERPARAMETER SEARCH: mindev
# mindev controls minimum deviance gain for a split — lower = more complex tree
# -----------------------------------------------------------------------------
mindev_range <- seq(0.0005, 0.05, by = 0.0005)
results      <- data.frame(mindev = mindev_range, error_rate = NA_real_,
                            n_leaves = NA_integer_)

cat("Searching over mindev values...\n")
for (i in seq_along(mindev_range)) {
  md <- mindev_range[i]
  tryCatch({
    fit <- tree(churn ~ ., data = train_data, control = tree.control(
      nobs    = nrow(train_data),
      mindev  = md
    ))
    preds                  <- predict(fit, newdata = test_data, type = "class")
    results$error_rate[i]  <- mean(preds != test_data$churn)
    results$n_leaves[i]    <- sum(fit$frame$var == "<leaf>")
  }, error = function(e) {
    results$error_rate[i] <<- NA
  })
}

best_idx    <- which.min(results$error_rate)
best_mindev <- results$mindev[best_idx]
cat(sprintf("Best mindev = %.4f (Error Rate = %.4f, Leaves = %d)\n",
            best_mindev, results$error_rate[best_idx], results$n_leaves[best_idx]))

# Plot mindev tuning curve
ggplot(results, aes(x = mindev, y = error_rate)) +
  geom_line(color = "#3498DB", linewidth = 1) +
  geom_vline(xintercept = best_mindev, linetype = "dashed", color = "#E74C3C") +
  labs(
    title    = "Classification Tree: Error Rate vs mindev",
    subtitle = paste0("Best mindev = ", best_mindev),
    x        = "mindev (minimum deviance for split)",
    y        = "Test Error Rate"
  ) +
  theme_minimal(base_size = 13)

ggsave("outputs/06_tree_mindev_tuning.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 2. FINAL MODEL
# -----------------------------------------------------------------------------
best_tree <- tree(churn ~ ., data = train_data, control = tree.control(
  nobs   = nrow(train_data),
  mindev = best_mindev
))

cat("\nTree Structure Summary:\n")
print(summary(best_tree))

# Plot tree
png("outputs/07_classification_tree.png", width = 1000, height = 700)
plot(best_tree)
text(best_tree, pretty = 0, cex = 0.7)
title(main = "Bazaar Telecom — Classification Tree")
dev.off()

# -----------------------------------------------------------------------------
# 3. EVALUATION
# -----------------------------------------------------------------------------
tree_preds <- predict(best_tree, newdata = test_data, type = "class")

conf_mat <- table(Predicted = tree_preds, Actual = test_data$churn)
cat("\nClassification Tree Confusion Matrix:\n")
print(conf_mat)

accuracy    <- sum(diag(conf_mat)) / sum(conf_mat)
sensitivity <- if (nrow(conf_mat) == 2 && ncol(conf_mat) == 2)
                 conf_mat[2, 2] / sum(conf_mat[, 2]) else 0
specificity <- conf_mat[1, 1] / sum(conf_mat[, 1])
tp          <- if (nrow(conf_mat) == 2) conf_mat[2, 2] else 0

cat(sprintf("\nAccuracy:    %.1f%%\n", accuracy * 100))
cat(sprintf("Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("Specificity: %.1f%%\n", specificity * 100))
cat(sprintf("True Positives: %d\n", tp))

cat("\nClassification Tree modeling complete.\n")
