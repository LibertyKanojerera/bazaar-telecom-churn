# 📡 Bazaar Telecom — B2B Customer Churn Prediction

![R](https://img.shields.io/badge/Language-R-276DC3?logo=r&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Domain](https://img.shields.io/badge/Domain-Telecom%20%7C%20B2B-orange)
![Models](https://img.shields.io/badge/Models-Logistic%20Regression%20%7C%20kNN%20%7C%20Tree%20%7C%20Random%20Forest-9cf)

> *Can we build a predictive model that flags business customers at high risk of churn, so that the operator can proactively intervene with targeted retention actions?*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Context](#-business-context)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Recommendations](#-recommendations)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Authors](#-authors)

---

## 🔍 Project Overview

This project applies supervised machine learning classification techniques to predict **B2B customer churn** for a major Bulgarian telecommunications operator. Using a dataset of **8,453 business accounts**, we built, tuned, and evaluated four predictive models: Logistic Regression, k-Nearest Neighbors (kNN), Classification Trees, and Bagged Random Forests.

The central challenge — and finding — is that **class imbalance** (only 6.5% of customers churned) severely limits the practical utility of models optimizing for accuracy. This project documents that limitation rigorously and proposes a roadmap for operationally deployable churn detection.

---

## 💼 Business Context

Customer churn in B2B telecoms is a high-stakes problem. Each lost account represents:

- Significant recurring revenue loss
- Risk of churn contagion across related business networks
- High client acquisition costs to replace departed accounts

**The business objective** is to deploy targeted, data-driven retention interventions *before* a customer disengages — not to react after the fact.

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Bulgarian telecom operator (business segment) |
| **Records** | 8,453 business customer accounts |
| **Churn Rate** | 6.5% (549 churned / 7,904 retained) |
| **Features** | Revenue metrics, subscriber activity, customer segmentation |

### Variables

| Variable | Description |
|----------|-------------|
| `crm_pid_value_segment` | Customer value tier (Bronze, Silver, Gold, Platinum, etc.) |
| `effective_segment` | Business size segment (SOHO, VSE, SME, SE, LE) |
| `active_subscribers` | Number of active SIMs/subscriptions |
| `not_active_subscribers` | Count of inactive subscribers |
| `suspended_subscribers` | Subscribers suspended in last 6 months |
| `avg_mobile_revenue` | Average mobile revenue per customer |
| `avg_fix_revenue` | Average fixed-line/internet revenue |
| `churn` | Target variable — churned (1) or retained (0) |

### Data Quality & Preprocessing

| Variable | Missing | Treatment |
|----------|---------|-----------|
| `suspended_subscribers` | 95.8% | Imputed → 0 (no record = no suspended subscribers) |
| `not_active_subscribers` | 49.1% | Imputed → 0 (same operational logic) |
| `billing_zip` | 0.02% | Removed (no predictive value; overfitting risk) |
| `pid`, `ka_name` | — | Dropped (identifiers) |
| `total_subs`, `total_revenue`, `arpu` | — | Dropped (multicollinear with other revenue/subscriber fields) |

Categorical variables (`crm_pid_value_segment`, `effective_segment`) were **dummy-encoded** using `fastDummies::dummy_cols()` with `remove_first_dummy = TRUE` to avoid perfect multicollinearity. The final modeling dataset contained **18 columns**.

---

## 🧪 Methodology

### Train/Test Split

```r
set.seed(12345)
training_size <- 0.6
training <- sample(1:nrow(telecom), training_size * nrow(telecom))
# Training: ~5,071 observations | Test: ~3,382 observations
```

### Models Applied

| Model | Configuration |
|-------|--------------|
| **Logistic Regression** | `glm(..., family = binomial(link = "logit"))` on all encoded features |
| **k-Nearest Neighbors** | `knn()` from `class` package; k=1–20 loop, best k by error rate |
| **Classification Tree** | `tree()` with `mindev` search (0.0005–0.05); best by test error |
| **Random Forest (Bagged)** | `ntree = 100`, `mtry = p` (full bagging) |

### Evaluation Metrics

Given the severe class imbalance, **accuracy alone is misleading**. Models were assessed on:

- **Sensitivity (Recall for churners)** = TP / (TP + FN) — *primary business metric*
- **Specificity** = TN / (TN + FP)
- **Confusion matrix** analysis
- **Overall Accuracy**

> ⚠️ **From a business standpoint:** Missing a churner (low sensitivity) is far more costly than incorrectly flagging a retained customer.

---

## 📈 Results

### Model Performance Summary

| Model | Accuracy | Sensitivity | Specificity | True Positives | Notes |
|-------|----------|-------------|-------------|----------------|-------|
| Logistic Regression | 92.9% | 0.0% | 100.0% | 0 | Predicts all as non-churn |
| **kNN (k=7)** | **93.0%** | **0.8%** | **99.9%** | **2** | **Best overall** |
| Classification Tree | 92.9% | 0.0% | 100.0% | 0 | No churners detected |
| Random Forest | 91.9% | 0.8% | 98.8% | 2 | Second-best sensitivity |

### Confusion Matrices

<details>
<summary><b>kNN (k=7) — Best Model</b></summary>

|  | Predicted: No Churn | Predicted: Churn |
|--|--|--|
| **Actual: No Churn** | 3,142 | 1 |
| **Actual: Churn** | 237 | 2 |

</details>

<details>
<summary><b>Random Forest</b></summary>

|  | Predicted: No Churn | Predicted: Churn |
|--|--|--|
| **Actual: No Churn** | 3,105 | 38 |
| **Actual: Churn** | 237 | 2 |

</details>

<details>
<summary><b>Classification Tree</b></summary>

|  | Predicted: No Churn | Predicted: Churn |
|--|--|--|
| **Actual: No Churn** | 3,143 | 0 |
| **Actual: Churn** | 239 | 0 |

</details>

### Key Insight

Despite 92–93% accuracy, **kNN's 0.8% sensitivity means only 1 in 125 actual churners is identified**. This is operationally useless for a proactive retention program. High accuracy in imbalanced classification is a known pitfall — this project quantifies that limitation empirically.

---

## 🔑 Key Findings

1. **Class imbalance is the dominant constraint.** A 93.5% majority class causes models to default to "No churn" predictions, making accuracy a misleading success metric.

2. **`avg_mobile_revenue` is the only statistically significant predictor** in the logistic regression (p = 0.004), suggesting revenue patterns carry some signal but are insufficient alone.

3. **The modeling pipeline is technically sound.** Cleaning, encoding, hyperparameter tuning, and ensemble methods were implemented correctly — the bottleneck is data quality, not methodology.

4. **Better features > better algorithms.** Tenure, contract renewal dates, payment behavior, service complaints, and competitive offers are the variables that typically drive churn — none are present in this dataset.

---

## 💡 Recommendations

### 1. Address Class Imbalance
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) or **undersampling**
- Use **cost-sensitive learning** that penalizes missed churners more heavily
- **Lower the classification threshold** (e.g., from 0.5 → 0.3) to improve sensitivity at the cost of some specificity

### 2. Upgrade Model Architecture
- **Gradient Boosting** (XGBoost, LightGBM) with class-weight tuning
- **Regularized logistic regression** (LASSO/Ridge) with imbalance-aware settings
- **Neural networks** for feature interaction capture

### 3. Enrich the Dataset
Critical features currently missing:

| Feature Category | Examples |
|-----------------|---------|
| Lifecycle | Tenure, contract start/end dates, renewal history |
| Behavioral | Usage trends, data consumption, call volume changes |
| Service Quality | Network complaints, ticket history, resolution time |
| Financial | Payment delays, discount history, pricing tier changes |
| Competitive | Competitor pricing exposure, market region |

### 4. Improve Evaluation Rigor
- Replace single 60/40 split with **k-fold cross-validation** (k=5 or 10)
- Add **ROC-AUC** and **PR-AUC** as primary metrics for imbalanced classification
- Implement **cost-benefit matrix** to quantify the financial value of correct churn predictions

---

## 📁 Project Structure

```
bazaar-telecom-churn/
│
├── README.md                        # This file
│
├── data/
│   └── README.md                    # Data dictionary and sourcing notes
│
├── R/
│   ├── 01_data_preparation.R        # Cleaning, imputation, dummy encoding
│   ├── 02_eda.R                     # Exploratory data analysis & class imbalance
│   ├── 03_logistic_regression.R     # GLM model, coefficients, confusion matrix
│   ├── 04_knn.R                     # kNN with k-optimization loop
│   ├── 05_classification_tree.R     # Tree with mindev tuning
│   ├── 06_random_forest.R           # Bagged random forest
│   └── 07_model_comparison.R        # Summary table & visualization
│
├── outputs/
│   ├── model_comparison.png         # Performance chart
│   └── confusion_matrices.txt       # All confusion matrix outputs
│
└── docs/
    └── Bazaar_Telecom_Churn_Report.docx   # Full project report
```

---

## 🚀 Getting Started

### Prerequisites

```r
install.packages(c(
  "dplyr",
  "ggplot2",
  "fastDummies",
  "class",       # kNN
  "tree",        # Classification trees
  "randomForest",
  "caret",       # Confusion matrices
  "pROC"         # ROC curves
))
```

### Run the Analysis

```r
# Step 1: Data preparation
source("R/01_data_preparation.R")

# Step 2: Exploratory analysis
source("R/02_eda.R")

# Step 3–6: Run each model
source("R/03_logistic_regression.R")
source("R/04_knn.R")
source("R/05_classification_tree.R")
source("R/06_random_forest.R")

# Step 7: Compare all models
source("R/07_model_comparison.R")
```

> **Note:** The dataset is proprietary. Scripts are structured to accept a `telecom.csv` file placed in the `/data` directory.

---

## 👥 Authors

| Name | Contribution |
|------|-------------|
| Pamella Nyahuma | Data preparation, logistic regression |
| Shingirai Machaka | kNN modeling, evaluation metrics |
| Lillian Mkandla | Classification tree, discussion |
| Libert Kanojerera | Data Preparation, Random forest, recommendations |

---

## 📚 References & Further Reading

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
- Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE TKDE*, 21(9), 1263–1284.

---

*This project was completed as part of a graduate-level Business Analytics program.*
