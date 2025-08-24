# Liver Disease Prediction Using XGBoost

This project predicts liver disease based on patient clinical and demographic data. The model uses a **balanced ensemble of XGBoost classifiers** with feature engineering, threshold optimization, and SHAP explainability.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Project Overview
Liver disease can be life-threatening if undetected. This project uses machine learning to predict the presence of liver disease from blood test results and demographic data, providing interpretable insights into feature importance.

---

## Dataset
- **Source:** Indian Liver Patient Dataset (CSV format)
- **Target:** `Dataset`  
  - `1` = liver disease (positive)  
  - `0` = healthy (negative)  
- **Features include:** Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alanine Aminotransferase, Aspartate Aminotransferase, Total Proteins, Albumin, Albumin/Globulin Ratio.

---

## Features
- Raw lab values (`Age`, `AST`, `ALT`, etc.)
- Engineered features:
  - `AST_ALT_ratio` = Aspartate/Alanine Aminotransferase ratio  
  - `Albumin_TotalProtein_ratio` = Albumin / Total Proteins  
  - `Direct_TotalBilirubin_ratio` = Direct / Total Bilirubin  
- These ratios capture clinically meaningful patterns to improve model performance.

---

## Methodology

1. **Data Preprocessing**
   - Encode `Gender` as numeric.  
   - Map `Dataset` to binary labels.  
   - Fill missing values with median.  

2. **Balanced Ensemble**
   - Dataset is imbalanced: more patients with liver disease than healthy.  
   - Multiple XGBoost models are trained on **balanced 50-50 subsets** (equal healthy and diseased samples).  
   - Predictions are averaged to reduce variance and improve recall for minority class.

3. **Threshold Optimization**
   - Default probability threshold (0.5) is tuned to improve **F1-score for healthy patients**.  

4. **Evaluation**
   - Metrics: Accuracy, F1-score, Precision, Recall, Confusion Matrix, ROC AUC.  
   - Visualizations: Confusion matrix and ROC curve.  

5. **Explainability**
   - **SHAP** is used to compute feature importance.  
   - Bar plots show overall importance; dot plots show direction and magnitude of influence per feature.

---

## Model Training

- **Algorithm:** XGBoost Classifier
- **Parameters:**
  - `n_estimators=300`
  - `max_depth=6`
  - `learning_rate=0.05`
  - `subsample=0.7`
  - `colsample_bytree=0.8`
- **Balanced ensemble:** 5 iterations, averaging predictions.

---

## Evaluation

- **Optimal threshold:** ~0.53 for healthy class.  
- **Metrics on test set:**
  - Accuracy: 0.744
  - ROC AUC: 0.798
  - Confusion Matrix:

| Actual \ Predicted | 0 (Healthy) | 1 (Liver Disease) |
|------------------|-------------|------------------|
| 0 (Healthy)      | 28          | 6                |
| 1 (Liver Disease)| 24          | 59               |

- Achieves **better recall for minority class** while maintaining overall accuracy.

---

## Feature Importance

Top features according to SHAP:

| Feature                        | Importance |
|--------------------------------|-----------|
| Age                             | 666       |
| AST_ALT_ratio                   | 627       |
| Alkaline_Phosphotase            | 590       |
| Alamine_Aminotransferase        | 539       |
| Aspartate_Aminotransferase      | 525       |
| Albumin_TotalProtein_ratio      | 462       |
| Albumin                         | 430       |
| Total_Protiens                  | 400       |
| Direct_TotalBilirubin_ratio     | 376       |
| Total_Bilirubin                 | 272       |
| Albumin_and_Globulin_Ratio      | 224       |
| Direct_Bilirubin                | 119       |
| Gender                          | 61        |

---

## Results

- Predicted probabilities and classes are saved in `model_predictions.csv`.  
- Balanced ensemble ensures **healthy patients are not overlooked**.  
- SHAP plots provide **interpretable insights** for clinical validation.

---

## Usage

1. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap
```
2. Place indian_liver_patient.csv in the same directory.
3. Run the Python script:
      `python app.py
4. View predictions, metrics, and feature importance plots.

## Dependencies
 * Python 3.x
 * pandas        
 * numpy
 * matplotlib
 * scikit-learn
 * xgboost
 * shap
