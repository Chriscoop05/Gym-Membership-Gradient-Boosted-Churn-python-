# Gym Membership Churn Prediction

A machine learning project that uses Gradient Boosting Classification to identify gym members at high risk of churning. 

## Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Pipeline & Data Preprocessing](#-pipeline--data-preprocessing)
- [Model Tuning & Training](#-model-tuning--training)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Insights & Feature Importance](#-key-insights--feature-importance)
- [Installation & Usage](#-installation--usage)

---

##  Project Overview
Predicting customer churn is a vital objective for subscription-based business models like fitness centers. In this project, a **Gradient Boosting Classifier** was optimized using structural hyperparameter tuning to capture complex interactions within gym member behavior—such as declining visit frequencies and auxiliary spending patterns—yielding a highly robust predictive framework.

## Dataset Summary
The project utilizes the `gym_churn_us.csv` dataset, which captures a profile of 4,000 gym members across 14 attributes:
* **Demographics/Profile:** `gender`, `Near_Location` (living/working near the gym), `Partner` (employee of a partner company), `Promo_friends` (joined via a "bring a friend" offer), `Phone` (provided contact phone), `Age`.
* **Behavioral/Engagement:** `Contract_period` (1, 6, or 12 months), `Group_visits` (participation in group classes), `Lifetime` (months since initial sign-up), `Avg_additional_charges_total` (spending on ancillary services like cafe, trainers, gear).
* **Temporal Class Frequency:**
    * `Avg_class_frequency_total`: Average weekly visits over the entire lifetime.
    * `Avg_class_frequency_current_month`: Average weekly visits specifically during the most recent month.
* **Target Indicator:** `Churn` (1 if the user canceled, 0 if retained).

##  Pipeline & Data Preprocessing
To scale continuous features uniformly without distorting sparse, binary categorical attributes (0 or 1), a multi-stage workflow was developed using `scikit-learn` components:
1.  **Feature Selection:** Divided variables into binary flags and continuous numerical vectors.
2.  **Column Scaling:** Applied `StandardScaler` explicitly to continuous features to center their distributions around a mean ($\mu=0$) and variance ($\sigma=1$).
3.  **Transformer Isolation:** Integrated via a robust `ColumnTransformer` layout to avoid data leakage during cross-validation splits.

##  Model Tuning & Training
The core predictive engine is built on `GradientBoostingClassifier`. Given the asymmetric real-world business penalty of failing to catch a churning customer, the hyperparameter tuning phase focused heavily on optimizing **Recall**. 

A grid search cross-validation space (`GridSearchCV`) with 5 folds was systematically evaluated over 27 candidate configurations:
```python
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
