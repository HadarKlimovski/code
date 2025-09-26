import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV,cross_val_score
from sklearn.metrics import roc_curve, auc,confusion_matrix, roc_auc_score
from sklearn.calibration import calibration_curve
from proteins_lists import SELECTED_PROTEINS_FROM_RFECV
from loaders.proteomics import load_discovery_data_for_binary_classification
from age_scores_survival.classification_utils import (
    build_xgboost_model,
    evaluate_confusion_matrix,
    extract_feature_importance,
    plot_cv_and_test_roc,
    plot_sigmoid_calibration_comparison,
    run_shap_explainer,
    run_cv_evaluation_and_collect_risks
)

# Global config
cv = 10
seed = 42
#TODO:
OUTPUTS_DIR = pd.to_csv('your_file_path.csv')
FIGURES_DIR = pd.to_csv('your_file_path.csv')

#model param
model_params = {
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'learning_rate': 0.2,
    'n_estimators': 200,
    'max_depth': 4,
    'subsample': 0.7,
    'min_child_weight': 9,
    'n_jobs': -1,
    'scale_pos_weight': 1.22,
    'random_state': seed
}

# === Load Data ===
#TODO:
data = pd.read_csv('your_path.cvs')
print(data)
X = data[SELECTED_PROTEINS_FROM_RFECV]
y = data['1 = Age Onset ≥ 60']


# === Run Stratified CV and Save CV Risk Scores ===
cv_risk_scores_df = run_cv_evaluation_and_collect_risks(
    model_builder=build_xgboost_model,
    X=X,
    y=y,
    class_names=['younger', 'older'],
    cv=cv
)
cv_risk_scores_df.to_csv(os.path.join(OUTPUTS_DIR, "cv_age_scores_proteins.csv"), index=False)

# === 2. Final Model (Train/Test) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
final_model = build_xgboost_model()
final_model.fit(X_train, y_train)

# === Evaluate and Plot ===
evaluate_confusion_matrix(final_model, X_test, y_test, class_names=['younger', 'Older'])
extract_feature_importance(final_model, X.columns, top_n=30)
plot_cv_and_test_roc(final_model, X, y, X_test, y_test, cv=cv, filename="roc_curve_cv_test_age_proteins.png")

# === Save Raw Test Risk Scores ===
age_score_test = final_model.predict_proba(X_test)[:, 1]
age_score_test_df = pd.DataFrame({
    "Sample": X_test.index,
    "Proteomic_Age_Score": age_score_test,
    "Label": y_test
})
age_score_test_df.to_csv(os.path.join(OUTPUTS_DIR, 'age_scores_proteins_test.csv'), index=False)

# === Calibration ===
calibrated_model = plot_sigmoid_calibration_comparison(
    base_model=final_model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_bins=20,
    filename="calibration_curve_sigmoid_vs_raw.png",
    csv_filename="age_scores_test_calibrated.csv"
)

# === SHAP Explanation ===
shap_values, shap_importance_df = run_shap_explainer(
    final_model, X_train, X_test, feature_names=X.columns,
    top_n=30, max_dependence=6
)

