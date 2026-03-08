# =============================================================================
# Script: 07_model_comparison.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Compare all models, produce summary chart and final report table
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# -----------------------------------------------------------------------------
# 1. RESULTS SUMMARY TABLE
# (Manually collated from model output scripts)
# -----------------------------------------------------------------------------
results <- data.frame(
  Model       = c("Logistic Regression", "kNN (k=7)", "Classification Tree", "Random Forest"),
  Accuracy    = c(92.9, 93.0, 92.9, 91.9),
  Sensitivity = c(0.0,  0.8,  0.0,  0.8),
  Specificity = c(100.0, 99.9, 100.0, 98.8),
  TP          = c(0, 2, 0, 2),
  stringsAsFactors = FALSE
)

cat("=== MODEL COMPARISON SUMMARY ===\n")
print(results)

# -----------------------------------------------------------------------------
# 2. GROUPED BAR CHART: Accuracy vs Sensitivity vs Specificity
# -----------------------------------------------------------------------------
results_long <- results %>%
  select(Model, Accuracy, Sensitivity, Specificity) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value") %>%
  mutate(
    Model  = factor(Model, levels = results$Model),
    Metric = factor(Metric, levels = c("Accuracy", "Specificity", "Sensitivity"))
  )

ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = paste0(Value, "%")),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3.2, fontface = "bold") +
  scale_fill_manual(values = c(
    "Accuracy"    = "#3498DB",
    "Specificity" = "#27AE60",
    "Sensitivity" = "#E74C3C"
  )) +
  labs(
    title    = "Model Performance Comparison",
    subtitle = "High accuracy masks near-zero sensitivity — the critical business metric",
    x        = NULL,
    y        = "Score (%)",
    fill     = "Metric",
    caption  = "Bazaar Telecom B2B Churn Dataset | n=8,453 | 60/40 Train-Test Split"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.text.x     = element_text(angle = 15, hjust = 1),
    legend.position = "top",
    plot.title      = element_text(face = "bold"),
    plot.subtitle   = element_text(color = "grey40")
  ) +
  ylim(0, 115)

ggsave("outputs/10_model_comparison.png", width = 9, height = 6, dpi = 150)
cat("Saved: outputs/10_model_comparison.png\n")

# -----------------------------------------------------------------------------
# 3. SENSITIVITY SPOTLIGHT CHART
# The key business metric — how many churners did each model catch?
# -----------------------------------------------------------------------------
ggplot(results, aes(x = reorder(Model, TP), y = TP, fill = Model)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = ifelse(TP > 0,
                               paste0(TP, " churners\ncaught"),
                               "None\ncaught")),
            vjust = ifelse(results$TP > 0, -0.3, -0.3), size = 4, fontface = "bold") +
  scale_fill_manual(values = c(
    "Logistic Regression" = "#BDC3C7",
    "kNN (k=7)"           = "#E74C3C",
    "Classification Tree" = "#BDC3C7",
    "Random Forest"       = "#E67E22"
  )) +
  labs(
    title    = "True Positives: Churners Correctly Identified",
    subtitle = "Out of 239 actual churners in the test set",
    x        = NULL,
    y        = "Churners Correctly Identified",
    caption  = "Even the best models catch only 2 of 239 churners (0.8% sensitivity)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        plot.title      = element_text(face = "bold")) +
  ylim(0, max(results$TP) + 3)

ggsave("outputs/11_true_positives.png", width = 7, height = 5, dpi = 150)
cat("Saved: outputs/11_true_positives.png\n")

# -----------------------------------------------------------------------------
# 4. EXPORT RESULTS TABLE
# -----------------------------------------------------------------------------
write.csv(results, "outputs/model_results_summary.csv", row.names = FALSE)
cat("\nModel comparison complete. All outputs saved to /outputs.\n")
cat("\n=== FINAL RECOMMENDATION ===\n")
cat("kNN (k=7) is the best-performing model under current conditions.\n")
cat("However, ALL models are operationally unusable for churn detection.\n")
cat("Priority actions: (1) Address class imbalance, (2) Enrich feature set.\n")
