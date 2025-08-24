import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    f1_score, ConfusionMatrixDisplay, RocCurveDisplay
)
import xgboost as xgb
import matplotlib.pyplot as plt
import shap

# -------------------------
# 1. Load dataset
# -------------------------
ds = pd.read_csv("indian_liver_patient.csv")

ds['Dataset'] = ds['Dataset'].map({1:1, 2:0})
ds['Gender'] = ds['Gender'].map({'Male': 1, 'Female': 0})

ds.dropna(subset=['Dataset'], inplace=True)
X = ds.drop(columns=['Dataset'])
y = ds['Dataset']
X = X.fillna(X.median())

# Feature engineering
X['AST_ALT_ratio'] = X['Aspartate_Aminotransferase'] / (X['Alamine_Aminotransferase'] + 1e-5)
X['Albumin_TotalProtein_ratio'] = X['Albumin'] / (X['Total_Protiens'] + 1e-5)
X['Direct_TotalBilirubin_ratio'] = X['Direct_Bilirubin'] / (X['Total_Bilirubin'] + 1e-5)

# Train-test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# 2. Balanced ensemble training
# -------------------------
n_iterations = 5
y_proba_ensemble = np.zeros(len(y_test))

for i in range(n_iterations):
    # Sample majority class to match minority
    df_train = pd.concat([X_train_full, y_train_full], axis=1)
    df_minority = df_train[df_train['Dataset']==0]
    df_majority = df_train[df_train['Dataset']==1].sample(len(df_minority), random_state=i)
    df_balanced = pd.concat([df_minority, df_majority]).sample(frac=1, random_state=i)  # shuffle

    X_train = df_balanced.drop(columns=['Dataset'])
    y_train = df_balanced['Dataset']

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=i
    )
    model.fit(X_train, y_train)
    
    # Predict probabilities on test set
    y_proba_ensemble += model.predict_proba(X_test)[:,1]

# Average ensemble predictions
y_proba_ensemble /= n_iterations

# -------------------------
# 3. Threshold optimization
# -------------------------
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_proba_ensemble>t).astype(int), pos_label=0) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print("Optimal threshold for class 0:", best_threshold)

y_pred = (y_proba_ensemble > best_threshold).astype(int)

# -------------------------
# 4. Evaluation
# -------------------------
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_ensemble))
print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_proba_ensemble)
plt.title("ROC Curve")
plt.show()

# -------------------------
# 5. SHAP for interpretability
# -------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values.values, X_test, plot_type="bar", feature_names=X_test.columns)
shap.summary_plot(shap_values.values, X_test, feature_names=X_test.columns)
