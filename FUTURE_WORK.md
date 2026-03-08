# 🔬 Future Work & Extensions

This document outlines a structured roadmap for extending this project beyond its current scope. These recommendations reflect lessons learned from Phase 1 and are organized by priority and estimated impact.

---

## Priority 1 — Address Class Imbalance

The most impactful single improvement. Until class imbalance is addressed, no algorithm will reliably detect churners.

### Recommended approaches

**SMOTE (Synthetic Minority Oversampling)**
```r
# Using the themis package (tidymodels ecosystem)
library(themis)
library(recipes)

churn_recipe <- recipe(churn ~ ., data = train_data) %>%
  step_smote(churn, over_ratio = 0.5)  # Bring minority class to 50% of majority
```

**Cost-Sensitive Learning**
```r
# Penalize missed churners 10x more than false alarms
class_weights <- ifelse(train_data$churn == 1, 10, 1)
rf_weighted <- randomForest(churn ~ ., data = train_data,
                             classwt = c("0" = 1, "1" = 10))
```

**Threshold Tuning**
```r
# Instead of default 0.5, optimize threshold using Youden's J
library(pROC)
roc_obj   <- roc(test_y, predicted_probs)
best_thresh <- coords(roc_obj, "best", ret = "threshold",
                       best.method = "youden")
```

---

## Priority 2 — Advanced Models

### Gradient Boosting (XGBoost)
```r
library(xgboost)

# scale_pos_weight handles imbalance internally
xgb_model <- xgboost(
  data            = xgb.DMatrix(as.matrix(train_X), label = train_y),
  nrounds         = 200,
  max_depth       = 6,
  eta             = 0.05,
  scale_pos_weight = sum(train_y == 0) / sum(train_y == 1),  # ~14.4
  objective       = "binary:logistic",
  eval_metric     = "auc"
)
```

### LASSO Logistic Regression (feature selection + regularization)
```r
library(glmnet)

lasso_cv  <- cv.glmnet(as.matrix(train_X), train_y, family = "binomial",
                        alpha = 1, type.measure = "auc")
lasso_preds <- predict(lasso_cv, newx = as.matrix(test_X),
                        s = "lambda.min", type = "response")
```

---

## Priority 3 — Rigorous Evaluation Framework

Replace the single 60/40 split with proper cross-validation:

```r
library(caret)

ctrl <- trainControl(
  method          = "cv",
  number          = 10,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,  # Reports ROC-AUC, Sensitivity, Specificity
  savePredictions = TRUE
)

# Compare models on ROC-AUC instead of accuracy
model_glm <- train(churn ~ ., data = telecom, method = "glm",
                   family = "binomial", trControl = ctrl, metric = "ROC")
model_rf  <- train(churn ~ ., data = telecom, method = "rf",
                   trControl = ctrl, metric = "ROC")
```

**Better metrics to adopt:**

| Metric | Why it matters |
|--------|---------------|
| ROC-AUC | Threshold-independent, handles imbalance better than accuracy |
| PR-AUC (Precision-Recall) | Most informative for severely imbalanced classes |
| F1-Score | Harmonic mean of precision and recall |
| Matthews Correlation Coefficient | Works well for imbalanced binary classification |

---

## Priority 4 — Feature Engineering

Critical features absent from the current dataset:

| Feature | Proxy / How to generate |
|---------|------------------------|
| Tenure | `first_contract_date - today()` |
| Contract renewal proximity | `days_until_renewal` — high risk window |
| Revenue trend | `Δ avg_revenue over 3 months` — declining = risk signal |
| Subscriber churn within account | `pct_subscribers_deactivated` |
| Support interactions | Count of CRM tickets in last 90 days |
| Payment behavior | `days_past_due`, `payment_failures` |

---

## Priority 5 — Business Integration

A deployed churn model should produce an **actionable risk score**, not just a binary prediction:

```r
# Generate churn probability for every customer
churn_scores <- predict(best_model, newdata = all_customers, type = "response")

# Segment into risk tiers for targeted intervention
intervention_plan <- data.frame(
  customer_id   = all_customers$pid,
  churn_prob    = churn_scores,
  risk_tier     = cut(churn_scores,
                       breaks = c(0, 0.2, 0.5, 0.8, 1.0),
                       labels = c("Low", "Medium", "High", "Critical"))
)
```

**Suggested intervention mapping:**

| Risk Tier | Probability | Action |
|-----------|------------|--------|
| Critical | > 80% | Immediate account manager outreach + custom retention offer |
| High | 50–80% | Proactive check-in call + service review |
| Medium | 20–50% | Email campaign + loyalty incentive |
| Low | < 20% | Standard account management cadence |

---

## Stretch Goal — Survival Analysis

For a truly sophisticated churn model, consider **time-to-churn** rather than binary churn:

```r
library(survival)
library(survminer)

# Model: how long until a customer churns?
surv_model <- survfit(Surv(tenure_months, churn) ~ effective_segment,
                       data = customer_data)
ggsurvplot(surv_model, pval = TRUE, conf.int = TRUE)
```

This enables "churn risk in next 30/60/90 days" scoring — far more actionable for operations.
