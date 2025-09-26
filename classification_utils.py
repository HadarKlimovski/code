import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc,confusion_matrix, roc_auc_score



# Global config
cv = 10
seed = 42
OUTPUTS_DIR = pd.to_csv('your_file_path.csv')
FIGURES_DIR = pd.to_csv('your_file_path.csv')

#model param
model_params = {
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'learning_rate': 0.2,
    'n_estimators': 100,
    'max_depth': 4,
    'subsample': 0.7,
    'min_child_weight': 9,
    'n_jobs': -1,
    'scale_pos_weight': 1.22,
    'random_state': seed
}


def cv_model(model, X_train, y_train, X_test, y_test, cv=5):
    # 1. Evaluate CV AUC on TRAINING data (only)
    cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    # 2. Fit on full training data
    model.fit(X_train, y_train)

    # 3. Evaluate on TEST set
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    return model, cv_auc_scores, test_auc


# model
def build_xgboost_model(custom_params=None):
    params = model_params.copy()
    if custom_params:
        params.update(custom_params)
    return xgb.XGBClassifier(**params)

# Train model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    early_stopping_rounds=50,
    verbose=True
    )
    print(f"Best iteration: {model.best_iteration}")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Final Test ROC-AUC: {roc_auc:.4f}")
    return model

# confusion matrix
def evaluate_confusion_matrix(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix_age_proteins.png'), bbox_inches='tight', dpi=300)
    plt.close()

#plot feature importance 
def extract_feature_importance(model, feature_names, top_n=40):
    importance = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    feature_df.to_csv(os.path.join(OUTPUTS_DIR, 'feature_importance_age_proteins.csv'), index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(top_n), palette='coolwarm')
    plt.title('Top Important Proteins for Classification')
    plt.xlabel('Feature Importance')
    plt.ylabel('Protein (Feature)')
    plt.yticks(size = 10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance_binary_age_proteins.png'), bbox_inches='tight', dpi=300)
    plt.close()
     

# ROC curve with CV and Test 
def plot_cv_and_test_roc(model, X, y, X_test, y_test, cv=5, filename="roc_curve_cv_test.png"):
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
    plt.figure(figsize=(8, 6))
    
    # --- CV curves ---
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        model_cv = build_xgboost_model()
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_proba_cv = model_cv.predict_proba(X_val_cv)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_val_cv, y_pred_proba_cv)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate for mean curve
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        
        # plt.plot(fpr, tpr, lw=1, alpha=0.4,
        #          label=f"CV fold {i+1} (AUC = {roc_auc:.2f})")
    
    # Mean CV curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color="#2c9602",
             label=f"Mean CV ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
             lw=2, alpha=0.7)
    
    # --- Test curve ---
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, color="#6C0296", lw=2,
             label=f"Test ROC (AUC = {roc_auc_test:.2f})")
    
    # --- Plot settings ---
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=300)
    plt.close()

# SHAP explanations 
def run_shap_explainer(model, X_train, X_test, feature_names, 
                       top_n=30, max_dependence=6,
                       figures_dir=FIGURES_DIR, outputs_dir=OUTPUTS_DIR):
    """
    Computes SHAP values for an XGBoost model and saves:
      - beeswarm summary (probability scale)
      - bar summary of mean |SHAP|
      - dependence plots for top features
      - CSV of SHAP importances
    """
    # 1) Background (reference) set for interventional SHAP (small, fast, stable)
    bg_n = min(200, X_train.shape[0])
    background = shap.sample(X_train, bg_n, random_state=seed)

    # 2) Explainer on probability scale (not raw logits)
    explainer = shap.TreeExplainer(
        model,
        data=background,
        model_output="probability",          # probabilities instead of margins
        feature_perturbation="interventional"
    )

    # 3) SHAP values on X_test
    shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

    # 4) Mean |SHAP| importances + CSV
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "mean_abs_SHAP": mean_abs
    }).sort_values("mean_abs_SHAP", ascending=False)
    imp_df.to_csv(os.path.join(outputs_dir, "shap_importance_age_proteins.csv"), index=False)

    # 5) Beeswarm summary (top_n)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names,
        max_display=top_n, show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "shap_age_proteins.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
    
    # 6) SHAP heatmap for top 30 features
    top_features = imp_df["Feature"].values[:30]
    top_feature_indices = [list(feature_names).index(f) for f in top_features]

    # Subset SHAP and X for top features
    shap_top = shap_values[:, top_feature_indices]
    X_top = X_test[top_features]

    # 8) Aggregate interaction matrix (mean abs values)
    top_features = imp_df["Feature"].head(top_n).tolist()
    top_indices = [list(feature_names).index(f) for f in top_features]


    return shap_values, imp_df
    
    
    
def plot_sigmoid_calibration_comparison(base_model, X_train, y_train, X_test, y_test,
                                        n_bins=20,
                                        filename="calibration_curve_sigmoid_vs_raw.png",
                                        csv_filename="proto_age_Collection_socre_test.csv"):
    """
    Fits a sigmoid-calibrated classifier, plots both calibrated and uncalibrated calibration curves,
    and saves the calibrated probabilities to CSV.
    """
    # 1. Fit sigmoid-calibrated model on training data
    calibrated_clf = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train, y_train)

    # 2. Get predicted probabilities (test set)
    y_prob_raw = base_model.predict_proba(X_test)[:, 1]             # Uncalibrated
    y_prob_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]  # Calibrated

    # 3. Compute calibration curves
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=n_bins, strategy='quantile')
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_prob_calibrated, n_bins=n_bins, strategy='quantile')

    # 4. Plot both curves
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred_raw, prob_true_raw, marker='o', label='Uncalibrated', color='blue')
    plt.plot(prob_pred_cal, prob_true_cal, marker='o', label='Sigmoid-Calibrated', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Uncalibrated vs Sigmoid-Calibrated")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), bbox_inches="tight", dpi=300)
    plt.close()

    # 5. Save calibrated scores to CSV
    age_score_test_df = pd.DataFrame({
        "Sample": X_test.index,
        "Proteomic_Age_Score_SigmoidCalibrated": y_prob_calibrated,
        "Label": y_test
    })
    age_score_test_df.to_csv(os.path.join(OUTPUTS_DIR, csv_filename), index=False)

    return calibrated_clf



def run_cv_evaluation_and_collect_risks(model_builder, X, y, class_names, cv=5):
    """
    Perform stratified CV, plot per-fold confusion matrix, and collect risk scores for test folds.
    Returns a DataFrame of all test fold predictions (proteomic age score).
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    all_risk_scores = []
    cv_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        model_cv = model_builder()
        model_cv.fit(X_train_cv, y_train_cv)
        
        # Predict
        y_pred_cv = model_cv.predict(X_test_cv)
        y_proba_cv = model_cv.predict_proba(X_test_cv)[:, 1]
        
        # AUC
        auc_cv = roc_auc_score(y_test_cv, y_proba_cv)
        cv_aucs.append(auc_cv)
        print(f"[Fold {fold}] ROC-AUC: {auc_cv:.4f}")
        
        # Save risk scores
        fold_df = pd.DataFrame({
            "Sample": X_test_cv.index,
            "Proteomic_Age_Score": y_proba_cv,
            "True_Label": y_test_cv.values,
            "Fold": fold
        })
        all_risk_scores.append(fold_df)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_cv, y_pred_cv)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - Fold {fold}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_age_proteins.png_fold_{fold}.png'), dpi=300)
        plt.close()
    
    # Report CV AUC
    print(f"\nMean CV ROC-AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    
    # Combine all fold risk scores
    risk_scores_df = pd.concat(all_risk_scores, axis=0).sort_values("Sample").reset_index(drop=True)
    return risk_scores_df




