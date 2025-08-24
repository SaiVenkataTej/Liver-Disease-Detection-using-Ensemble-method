import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

# Load dataset
ds = pd.read_csv("indian_liver_patient.csv")
print("Dataset head:\n", ds.head())
print("\nDataset info:\n")
print(ds.info())

# Encode categorical variables
ds['Dataset'] = ds['Dataset'].map({1: 1, 2: 0})
ds['Gender'] = ds['Gender'].map({'Male': 1, 'Female': 0})

# Drop rows with missing target
ds.dropna(subset=['Dataset'], inplace=True)

# Fill missing values with median
X = ds.drop(columns=['Dataset'])
y = ds['Dataset']
X = X.fillna(X.median())

# Print class distribution
count_Has = y.sum()
count_no = len(y) - count_Has
print(f"\nClass distribution:\nPeople with liver disease: {count_Has}\nPeople with no liver disease: {count_no}")

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=410,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=count_no/count_Has
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix plot
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# ROC curve plot
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Feature importance plot
xgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()
