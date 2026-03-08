# =============================================================================
# Script: 02_eda.R
# Project: Bazaar Telecom B2B Customer Churn Prediction
# Purpose: Exploratory Data Analysis — class imbalance, distributions, correlations
# =============================================================================

library(ggplot2)
library(dplyr)
library(corrplot)

telecom   <- readRDS("data/telecom_encoded.rds")
telecom_c <- readRDS("data/telecom_rf.rds")   # Categorical version

# -----------------------------------------------------------------------------
# 1. CLASS IMBALANCE
# -----------------------------------------------------------------------------
churn_counts <- as.data.frame(table(telecom$churn)) %>%
  rename(Churn = Var1) %>%
  mutate(
    Churn = ifelse(Churn == 1, "Churned", "Retained"),
    Pct   = round(Freq / sum(Freq) * 100, 1)
  )

ggplot(churn_counts, aes(x = Churn, y = Freq, fill = Churn)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(Pct, "%\n(n=", Freq, ")")),
            vjust = -0.3, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Churned" = "#E74C3C", "Retained" = "#2ECC71")) +
  labs(
    title    = "Target Variable Distribution: Severe Class Imbalance",
    subtitle = "Only 6.5% of business customers churned",
    x        = NULL,
    y        = "Count",
    caption  = "Bazaar Telecom B2B Churn Dataset (n=8,453)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  ylim(0, max(churn_counts$Freq) * 1.15)

ggsave("outputs/01_class_imbalance.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 2. REVENUE DISTRIBUTION BY CHURN STATUS
# -----------------------------------------------------------------------------
telecom_c_num <- telecom_c %>%
  mutate(churn_label = ifelse(churn == 1, "Churned", "Retained"))

ggplot(telecom_c_num, aes(x = churn_label, y = avg_mobile_revenue, fill = churn_label)) +
  geom_boxplot(outlier.alpha = 0.2) +
  scale_fill_manual(values = c("Churned" = "#E74C3C", "Retained" = "#3498DB")) +
  labs(
    title    = "Average Mobile Revenue by Churn Status",
    subtitle = "Higher revenue customers show marginally higher churn signal",
    x        = NULL,
    y        = "Avg Mobile Revenue",
    caption  = "Note: Outliers retained — likely represent legitimate high-value enterprise accounts"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("outputs/02_revenue_by_churn.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 3. CHURN RATE BY VALUE SEGMENT
# -----------------------------------------------------------------------------
segment_churn <- telecom_c %>%
  group_by(crm_pid_value_segment) %>%
  summarise(
    n          = n(),
    churned    = sum(as.numeric(as.character(churn))),
    churn_rate = churned / n * 100
  ) %>%
  arrange(desc(churn_rate))

ggplot(segment_churn, aes(x = reorder(crm_pid_value_segment, churn_rate),
                           y = churn_rate, fill = churn_rate)) +
  geom_col() +
  geom_text(aes(label = paste0(round(churn_rate, 1), "%")), hjust = -0.1, size = 4) +
  coord_flip() +
  scale_fill_gradient(low = "#F9E79F", high = "#E74C3C") +
  labs(
    title = "Churn Rate by Customer Value Segment",
    x     = "Value Segment",
    y     = "Churn Rate (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  xlim(0, max(segment_churn$churn_rate) * 1.2)

ggsave("outputs/03_churn_by_segment.png", width = 7, height = 5, dpi = 150)

# -----------------------------------------------------------------------------
# 4. CORRELATION MATRIX (numeric features)
# -----------------------------------------------------------------------------
numeric_cols <- telecom %>%
  select(active_subscribers, not_active_subscribers, suspended_subscribers,
         avg_mobile_revenue, avg_fix_revenue, churn)

cor_matrix <- cor(numeric_cols, use = "complete.obs")

png("outputs/04_correlation_matrix.png", width = 700, height = 700)
corrplot(cor_matrix,
         method  = "color",
         type    = "upper",
         addCoef.col = "black",
         tl.col  = "black",
         tl.srt  = 45,
         title   = "Correlation Matrix — Numeric Features",
         mar     = c(0, 0, 2, 0))
dev.off()

cat("EDA complete. Charts saved to /outputs.\n")
