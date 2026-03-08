# =============================================================================
# Script: 06_random_forest.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Bagged Random Forest — ntree=100, mtry=p (full bagging)
# =============================================================================

library(randomForest)
library(ggplot2)
library(dplyr)

telecom_rf   <- readRDS("data/telecom_rf.rds")    # Factor version for RF
training_idx <- readRDS("data/training_idx.rds")

train_data <- telecom_rf[training_idx, ]
test_data  <- telecom_rf[-training_idx, ]

p <- ncol(train_data) - 1   # Number of predictors

# -----------------------------------------------------------------------------
# 1. FIT BAGGED RANDOM FOREST
# mtry = p means all predictors considered at each split → full bagging
# -----------------------------------------------------------------------------
set.seed(42)
cat(sprintf("Fitting Random Forest (ntree=100, mtry=%d)...\n", p))

rf_model <- randomForest(
  churn ~ .,
  data       = train_data,
  ntree      = 100,
  mtry       = p,
  importance = TRUE
)

cat("\nRandom Forest Model Summary:\n")
print(rf_model)

# -----------------------------------------------------------------------------
# 2. OOB ERROR CURVE
# Out-of-bag error across number of trees
# -----------------------------------------------------------------------------
oob_df <- data.frame(
  ntrees    = 1:100,
  oob_error = rf_model$err.rate[, 1]
)

ggplot(oob_df, aes(x = ntrees, y = oob_error)) +
  geom_line(color = "#27AE60", linewidth = 1) +
  labs(
    title    = "Random Forest: OOB Error vs Number of Trees",
    subtitle = "Error stabilizes well before 100 trees",
    x        = "Number of Trees",
    y        = "Out-of-Bag Error Rate"
  ) +
  theme_minimal(base_size = 13)

ggsave("outputs/08_rf_oob_error.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 3. VARIABLE IMPORTANCE
# -----------------------------------------------------------------------------
importance_df <- as.data.frame(importance(rf_model)) %>%
  tibble::rownames_to_column("Variable") %>%
  arrange(desc(MeanDecreaseGini))

cat("\nTop 10 Variables by Mean Decrease Gini:\n")
print(head(importance_df, 10))

ggplot(head(importance_df, 10), aes(x = reorder(Variable, MeanDecreaseGini),
                                     y = MeanDecreaseGini)) +
  geom_col(fill = "#27AE60") +
  coord_flip() +
  labs(
    title = "Random Forest: Variable Importance",
    x     = NULL,
    y     = "Mean Decrease in Gini"
  ) +
  theme_minimal(base_size = 13)

ggsave("outputs/09_rf_variable_importance.png", width = 8, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 4. EVALUATION
# -----------------------------------------------------------------------------
rf_preds <- predict(rf_model, newdata = test_data, type = "class")

conf_mat <- table(Predicted = rf_preds, Actual = test_data$churn)
cat("\nRandom Forest Confusion Matrix:\n")
print(conf_mat)

accuracy    <- sum(diag(conf_mat)) / sum(conf_mat)
# Factor levels: 0 = no churn, 1 = churn
sensitivity <- conf_mat["1", "1"] / sum(conf_mat[, "1"])
specificity <- conf_mat["0", "0"] / sum(conf_mat[, "0"])
tp          <- conf_mat["1", "1"]

cat(sprintf("\nAccuracy:    %.1f%%\n", accuracy * 100))
cat(sprintf("Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("Specificity: %.1f%%\n", specificity * 100))
cat(sprintf("True Positives (churners caught): %d\n", tp))

cat("\nRandom Forest modeling complete.\n")
