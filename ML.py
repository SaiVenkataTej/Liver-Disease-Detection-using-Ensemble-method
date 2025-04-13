import pandas as pd 
import numpy as np
import xgboost as xgb
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score,ConfusionMatrixDisplay, RocCurveDisplay

ds=pd.read_csv("C:/Users/SAI VENKAT TEJA S/Desktop/sem 4/introduction to machine learning/ML PAPER/indian_liver_patient.csv")
print(ds.head())
print(ds.info())

#Considerations overfitting- perform max_depth or mis_samples_split or pruning parameters, feature importance 
count_Has=(ds['Dataset']==1).sum()
count_no=(ds['Dataset']==2).sum()
print(f"Dataset contains \nPeople with liver disease: {count_Has}\nPeople with no liver disease: {count_no}")

ds['Dataset']= ds['Dataset'].map({1:1,2:0})
ds['Gender'] = ds['Gender'].map({'Male': 1, 'Female': 0})


ds.dropna(subset=['Dataset'], inplace=True)

X=ds.drop(columns=['Dataset'])
y=ds['Dataset']

X=X.fillna(X.median())

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model= xgb.XGBClassifier(
    max_depth= 5,
    n_estimators=410,
    use_label_encoder= False,
    eval_metric='logloss',
    scale_pos_weight= count_no/count_Has
)

model.fit(X_train,y_train)

y_pred= model.predict(X_test)
y_proba= model.predict_proba(X_test)[:,1]

print("Classification report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
print("ROC AUC Score: ",roc_auc_score(y_test,y_proba))
print("Accuracy: ",accuracy_score(y_test,y_pred))

ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_estimator(model,X_test,y_test)
plt.title("ROC curve")
plt.show()
