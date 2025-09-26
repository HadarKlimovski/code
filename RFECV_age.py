
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from proteins_lists import PROTEINS

# Global variables
cv = 10
OUTPUTS_DIR = pd.to_csv('your_file_path.csv')

# Load Data
data = pd.read_csv('your_file_path.csv')
#load your data for classification : x= proteins abundance y = age of onset labels(above 60 and below).
X = data[PROTEINS]
y = data['1 = Age Onset ≥ 60']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define model parameters
rfecv_params = {
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'n_estimators': 120,
    'max_depth': 4,
    'subsample': 0.8,
    'min_child_weight': 12,
    'reg_alpha': 0.9530114696324256, 
    'n_jobs': -1,
    'random_state': 42,
    'scale_pos_weight': 1.669, #)165\90)
}

# Perform RFECV for feature selection
def perform_rfecv(model, X_train, y_train, cv):
    cv_rfecv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=cv_rfecv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    rfecv.fit(X_train, y_train)

    print(f"Number of optimal features: {rfecv.n_features_}")

    # Extract selected features
    selected_features = X_train.columns[rfecv.support_]
    
    # Save selected features
    selected_features_df = pd.DataFrame(selected_features, columns=["Selected Features"])
    selected_features_df.to_csv(os.path.join(OUTPUTS_DIR, 'selected_features_rfecv.csv'), index=False)

    return rfecv.support_, selected_features

# Build XGBoost model
def build_xgboost_model(custom_params=None):
    default_params = {
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'learning_rate': 0.05,
        'n_estimators': 100,
        'max_depth': 4,
        'subsample': 0.8,
        'min_child_weight': 12,
        'n_jobs': -1,
        'random_state': 42,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBClassifier(**default_params)




# Perform RFECV feature selection
rfecv_model = build_xgboost_model(rfecv_params)
selected_features_mask, selected_features = perform_rfecv(rfecv_model, X_train, y_train, cv=cv)

