# Diabetes Risk Prediction

An end-to-end machine learning pipeline for predicting diabetes risk using structured health survey data. The project focuses on handling class imbalance, reliable evaluation, calibration, and decision-oriented modeling.

---

## Problem

Diabetes prediction is a highly imbalanced classification problem where:

- The minority class (diabetes cases) is the most important  
- Accuracy and ROC-AUC can be misleading  
- Models must prioritize recall of positive cases without excessive false alarms  

---

## Dataset

- Source: BRFSS-style health survey dataset  
- ~250k samples, ~20+ features  
- Mix of binary, ordinal, and continuous variables  
- Target: `Diabetes_binary` (0 = No diabetes, 1 = Diabetes)

---

## Pipeline

Data → Cleaning → EDA → Feature Engineering → Modeling  
     → Hyperparameter Tuning → Evaluation → Calibration  
     → Threshold Optimization → Interpretation → Robustness

---

## Data Analysis & Preprocessing

- Removed duplicates and verified no missing values  
- Identified class imbalance (~6:1)  
- Used stratified train-test split  
- Standardized continuous features for linear models  
- Retained ordinal structure without unnecessary one-hot encoding  

---

## Feature Engineering

- BMI categories (captures non-linear health risk)
- Age × BMI interaction feature  
- Total health burden (physical + mental health days)  
- Log transformations for skewed variables  

Used Mutual Information to identify relevant features beyond linear correlation.

---

## Models

- Logistic Regression (baseline, interpretable)  
- Random Forest (non-linear, robust)  
- XGBoost (best performance on tabular data)  

Handled imbalance using:
- `class_weight`  
- `scale_pos_weight` (XGBoost)  

---

## Evaluation Strategy

Metrics used:

- ROC-AUC (ranking ability)  
- PR-AUC (primary metric for imbalance)  
- Log Loss (probability quality)  

Baseline comparisons:
- Dummy classifier  
- Simple rule-based model  

---

## Hyperparameter Tuning

- Used Optuna with 3-fold cross-validation  
- Optimized XGBoost for PR-AUC  
- Tuned parameters: depth, learning rate, estimators, subsampling  

---

## Calibration

Predicted probabilities were calibrated using:

- Platt scaling (sigmoid)  
- Isotonic regression  

Evaluated using:
- Reliability curves  
- Expected Calibration Error (ECE)  

---

## Threshold Optimization

Default threshold (0.5) is not suitable.

Objective:
- Maximize recall  
- Subject to precision ≥ 0.30  

Threshold selected on validation set and evaluated on test set.

---

## Model Interpretation

- Logistic Regression → Odds ratios  
- XGBoost → SHAP values  

Key factors:
- High BP, High Cholesterol, BMI, Age  

---

## Stability & Uncertainty

- Bootstrap used for coefficient stability  
- Cross-validation for performance variability  
- SHAP stability analyzed across folds  

---

## Fairness Analysis

Evaluated across:
- Income levels  
- Age groups  
- Gender  

Measured:
- ROC-AUC per group  
- Equal opportunity gap (recall differences)  

---

## Robustness Testing

- 5% random noise in binary features  
- Missing BMI at inference  

Findings:
- Small drop under noise  
- Larger drop with missing data  

---

## Results

- Best model: XGBoost  
- ROC-AUC ≈ 0.81  
- PR-AUC ≈ 0.44  
- Calibration improved probability reliability  
- Threshold tuning improved recall of diabetic cases  

---

## Limitations

- Cross-sectional dataset (no temporal modeling)  
- No causal inference  
- Depends on survey-quality inputs  
- Missing-value handling not fully integrated  

---

## Tech Stack

- Python  
- NumPy, pandas  
- scikit-learn  
- XGBoost  
- Optuna  
- SHAP  
- Matplotlib, Seaborn  

---

## Future Work

- Add missing-value handling pipeline  
- Deploy using FastAPI  
- Monitor calibration and drift  
- Extend to survival analysis  
