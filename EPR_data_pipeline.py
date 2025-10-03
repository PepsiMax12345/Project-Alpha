
"""


@author: ethan
"""

"""
Code for running the models for patient data
"""
from sklearn.feature_selection import mutual_info_classif 
# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

# Survival analysis imports
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


# Machine Learning imports
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import (cross_val_score, train_test_split, 
                                     GridSearchCV)
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support, silhouette_score,
                           brier_score_loss, confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, average_precision_score)

from sklearn.inspection import permutation_importance


# UMAP for dimensionality reduction
import umap.umap_ as umap

# Network analysis
import networkx as nx
import community.community_louvain as community_louvain





class ModelSelector:
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boost': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
    def systematic_model_comparison(self, X, y, cv_folds=5):
        """Compare multiple algorithms systematically"""
        results = {}
        
        for name, model in self.models.items():
            # Cross-validation with multiple metrics
            scoring = ['roc_auc', 'precision', 'recall', 'f1']
            scores = {}
            
            for metric in scoring:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                scores[metric] = {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'ci_lower': cv_scores.mean() - 1.96 * cv_scores.std(),
                    'ci_upper': cv_scores.mean() + 1.96 * cv_scores.std()
                }
            
            results[name] = scores
            
        return self.select_best_model(results)
    
    def select_best_model(self, results):
        """Select best model based on clinical priorities"""
        # Weight metrics by clinical importance
        weights = {'roc_auc': 0.4, 'precision': 0.3, 'recall': 0.2, 'f1': 0.1}
        
        best_score = 0
        best_model = None
        
        for model_name, scores in results.items():
            weighted_score = sum(
                scores[metric]['mean'] * weight 
                for metric, weight in weights.items()
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model = model_name
                
        return best_model, results


class SimpleValidation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
    def train_val_test_split(self, X, y, model, param_grid):
        """
        Simple 70-20-10 train-val-test split with automatic plotting
        """
        # First split: 90% train+val, 10% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=self.random_state, stratify=y
        )
        
        # Second split: 20% of the 90% (i.e. 0.222) for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=2/9, random_state=self.random_state, stratify=y_temp
        )
        
        print("\nData split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Hyperparameter tuning on validation set
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=3,
            scoring='roc_auc', 
            n_jobs=-1
        )
        
        # Fit on training data
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, y_val_pred)
        
        print(f"\nValidation AUC: {val_score:.3f}")
        print(f"Best params: {best_params}")
        
        # Retrain on combined train+val for final model
        X_train_final = pd.concat([X_train, X_val])
        y_train_final = pd.concat([y_train, y_val])
        
        final_model = model.__class__(**best_params)
        final_model.fit(X_train_final, y_train_final)
        
        # Final test evaluation
        y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        test_score = roc_auc_score(y_test, y_test_pred_proba)
        
        print(f"Test AUC: {test_score:.3f}")
        
        # Get feature importance using permutation importance
        feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        
        print("\nCalculating permutation importance...")
        perm_importance = permutation_importance(
            final_model, X_test, y_test, 
            n_repeats=10, 
            random_state=self.random_state, 
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        # Get the sorted indices and sort the feature names
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        sorted_feature_names = [feature_names[i] for i in sorted_idx]
        sorted_perm_importance = perm_importance.importances_mean[sorted_idx]
        
        # Assign permutation importance for plotting
        feature_importance = sorted_perm_importance
        feature_names = sorted_feature_names
        
        # Print the top 10 most important features
        print("Top Permutation Important Features (ROC AUC decrease):")
        for i in range(min(10, len(feature_names))):
            original_idx = perm_importance.importances_mean.argsort()[::-1][i]
            print(f"  {feature_names[i]}: {feature_importance[i]:.4f} +/- {perm_importance.importances_std[original_idx]:.4f}")
    
        # Plot the permutation importance
        plt.figure(figsize=(12, 8))
        plt.title("Permutation Importance on Test Set")
        plt.barh(feature_names, feature_importance)
        plt.xlabel("Mean decrease in AUC")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
        # Create all plots
        self.create_all_plots(
            y_test, y_test_pred, y_test_pred_proba,
            final_model, feature_importance, feature_names,
            val_score, test_score
        )   
        
        return {
            'model': final_model,
            'best_params': best_params,
            'val_score': val_score,
            'test_score': test_score,
            'mean_score': test_score,
            'std_score': 0.0,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_test_pred_proba
        }
    
    def create_all_plots(self, y_test, y_pred, y_proba, model,
                     feature_importance, feature_names, val_score, test_score):
        """
        Create all essential plots for the dissertation
        """
        # Create main figure with 6 subplots
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=1.02)
        
        # ROC CURVE
        ax1 = plt.subplot(2, 3, 1)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC
        ax1.plot(fpr, tpr, color='#2E86AB', lw=3,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Random classifier')
        
        # Mark optimal point
        optimal_idx = np.argmax(tpr - fpr)
        ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=10,
                 label=f'Optimal threshold = {thresholds[optimal_idx]:.3f}')
        
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-0.01, 1.01])
        ax1.set_ylim([-0.01, 1.01])
        
        # CONFUSION MATRIX
        ax2 = plt.subplot(2, 3, 2)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Create annotated heatmap
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = [f'{value}' for value in cm.flatten()]
        group_percentages = [f'{value/np.sum(cm):.1%}' for value in cm.flatten()]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                  zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True,
                    square=True, ax=ax2, vmin=0, vmax=cm.max())
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('Actual', fontsize=11)
        ax2.set_title(f'Confusion Matrix\n(Sens={sensitivity:.2f}, Spec={specificity:.2f})',
                      fontsize=13, fontweight='bold')
        
        # CALIBRATION PLOT
        ax3 = plt.subplot(2, 3, 3)
        
        # Calculate calibration
        fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        
        # Perfect calibration line
        ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
        
        # Model calibration
        ax3.plot(mean_pred, fraction_pos, 's-', color='#2E86AB',
                 markersize=8, lw=2, label='Model calibration')
        
        # Calculate ECE
        ece = np.mean(np.abs(fraction_pos - mean_pred))
        ax3.text(0.05, 0.95, f'ECE = {ece:.3f}', transform=ax3.transAxes,
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='white'))
        
        # Histogram
        ax3_hist = ax3.twinx()
        ax3_hist.hist(y_proba, bins=10, alpha=0.3, color='gray', edgecolor='black')
        ax3_hist.set_ylabel('Count', fontsize=10)
        ax3_hist.set_ylim([0, len(y_proba)*0.5])
        
        ax3.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax3.set_ylabel('Fraction of Positives', fontsize=11)
        ax3.set_title('Calibration Plot', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([-0.01, 1.01])
        ax3.set_ylim([-0.01, 1.01])
        
        # FEATURE IMPORTANCE (Top 15)
        ax4 = plt.subplot(2, 3, 4)
        
        if feature_importance is not None and feature_names is not None and len(feature_importance) > 0:
            num_features_to_plot = min(15, len(feature_importance))
            indices = np.argsort(feature_importance)[::-1][:num_features_to_plot]
            
            # Plot top features
            bars = ax4.barh(range(num_features_to_plot), feature_importance[indices], 
                           color=plt.cm.viridis(np.linspace(0.3, 0.9, num_features_to_plot)))
            ax4.set_yticks(range(num_features_to_plot))
            ax4.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
            ax4.set_xlabel('Importance', fontsize=11)
            ax4.set_title('Top Features', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add values
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                         f'{feature_importance[indices[i]]:.3f}',
                         ha='left', va='center', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance', fontsize=13, fontweight='bold')
        
        # PRECISION-RECALL CURVE
        ax5 = plt.subplot(2, 3, 5)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        ax5.plot(recall, precision, color='#2E86AB', lw=3,
                 label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline
        baseline = np.mean(y_test)
        ax5.axhline(y=baseline, color='r', linestyle='--', lw=2,
                    label=f'Baseline (prevalence = {baseline:.2f})')
        
        ax5.set_xlabel('Recall (Sensitivity)', fontsize=11)
        ax5.set_ylabel('Precision (PPV)', fontsize=11)
        ax5.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([-0.01, 1.01])
        ax5.set_ylim([-0.01, 1.01])
        
        # SCORE DISTRIBUTION
        ax6 = plt.subplot(2, 3, 6)
        
        # Split scores by outcome
        deceased_scores = y_proba[y_test == 1]
        survived_scores = y_proba[y_test == 0]
        
        # Plot distributions
        ax6.hist(survived_scores, bins=30, alpha=0.6, label=f'Survived (n={len(survived_scores)})',
                 color='#2E86AB', edgecolor='black', density=True)
        ax6.hist(deceased_scores, bins=30, alpha=0.6, label=f'Deceased (n={len(deceased_scores)})',
                 color='#A23B72', edgecolor='black', density=True)
        
        # Add threshold line
        ax6.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
        
        # Add mean lines
        ax6.axvline(x=np.mean(survived_scores), color='#2E86AB', linestyle=':', lw=2)
        ax6.axvline(x=np.mean(deceased_scores), color='#A23B72', linestyle=':', lw=2)
        
        ax6.set_xlabel('Predicted Probability', fontsize=11)
        ax6.set_ylabel('Density', fontsize=11)
        ax6.set_title('Score Distribution by Class', fontsize=13, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # Add text with validation and test scores
        fig.text(0.5, 0.01, f'Validation AUC: {val_score:.3f} | Test AUC: {test_score:.3f}',
                 ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Save figure
        fig.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved as 'model_performance_dashboard.png'")
    
    def analyze_feature_stability(self, feature_importance):
        """
        Simplified version - just returns the feature importance
        """
        if feature_importance is None:
            return None
            
        return {
            'mean_importance': feature_importance,
            'cv_importance': np.zeros_like(feature_importance)
        }


# Conservative parameter grids
CONSERVATIVE_PARAM_GRIDS = {
    'random_forest': {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [10, 20],
        'class_weight': ['balanced']
    },
    'gradient_boost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5]
    },
    'logistic': {
        'C': [0.1, 1.0, 10.0],
        'class_weight': ['balanced']
    }
}


def calculate_optimal_diagnosis_threshold(diag_df, target, min_patients=5):
    """
    Find optimal diagnosis prevalence threshold based on predictive value
    """
    print(f"Input validation - Diagnosis matrix: {diag_df.shape}")
    print(f"Input validation - Target: {len(target)} patients")
    
    # Ensure target aligns with diagnosis matrix
    if len(diag_df) != len(target):
        print(f"ERROR: Diagnosis matrix ({len(diag_df)}) and target ({len(target)}) size mismatch!")
        return 0.005  # fallback
    
    # Convert target to numeric if needed
    target = pd.Series(target).map({'Yes': 1, 'No': 0}) if target.dtype == 'object' else target
    
    prevalence_thresholds = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02]
    
    best_score = 0
    best_threshold = 0.005
    best_n_features = 0
    
    print("Testing diagnosis prevalence thresholds:")
    print("Threshold | Min Patients | Features | Quick Score")
    print("-" * 50)
    
    for threshold in prevalence_thresholds:
        min_prevalence = max(min_patients, int(threshold * len(diag_df)))
        
        # Filter diagnoses by threshold
        diag_cols_to_keep = diag_df.columns[diag_df.sum() >= min_prevalence]
        filtered_diag_df = diag_df[diag_cols_to_keep]
        
        if len(diag_cols_to_keep) < 10:
            print(f"{threshold:>8.3f} | {min_prevalence:>11} | {len(diag_cols_to_keep):>8} | TOO_FEW")
            continue
        
        try:
            # Check for valid data
            if filtered_diag_df.sum().sum() == 0:
                print(f"{threshold:>8.3f} | {min_prevalence:>11} | {len(diag_cols_to_keep):>8} | NO_DATA")
                continue
                
            # Quick mutual information score
            mi_scores = mutual_info_classif(filtered_diag_df, target, random_state=42)
            avg_mi_score = np.mean(mi_scores) if len(mi_scores) > 0 else 0
            
            # Quick RF test
            if target.nunique() < 2:
                print(f"{threshold:>8.3f} | {min_prevalence:>11} | {len(diag_cols_to_keep):>8} | NO_VARIATION")
                continue
                
            rf_quick = RandomForestClassifier(n_estimators=10, random_state=42)
            quick_score = cross_val_score(rf_quick, filtered_diag_df, target, cv=3, scoring='roc_auc').mean()
            
            # Combined score
            combined_score = (avg_mi_score * 0.3) + (quick_score * 0.7)
            
            print(f"{threshold:>8.3f} | {min_prevalence:>11} | {len(diag_cols_to_keep):>8} | {combined_score:>10.3f}")
            
            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold
                best_n_features = len(diag_cols_to_keep)
                
        except Exception as e:
            print(f"{threshold:>8.3f} | {min_prevalence:>11} | {len(diag_cols_to_keep):>8} | ERROR: {str(e)[:20]}")
            continue
    
    print(f"\nOptimal threshold: {best_threshold:.3f} ({best_n_features} features)")
    return best_threshold


def validate_final_dataset(final_df):
    """
    Validate the final dataset for consistency
    """
    print("\nFINAL DATASET VALIDATION")
    print("-" * 30)
    
    # Check for duplicates
    duplicates = final_df.duplicated('PatientKey').sum()
    print(f"Duplicate patients in final dataset: {duplicates}")
    
    # Check key statistics
    print(f"Total unique patients: {final_df['PatientKey'].nunique()}")
    print(f"Total records: {len(final_df)}")
    print(f"Mortality rate: {final_df['IsDeceased'].mean():.1%}")
    print(f"Age range: {final_df['Age'].min():.1f} - {final_df['Age'].max():.1f}")
    
    # Check for impossible values
    impossible_ages = (final_df['Age'] < 0) | (final_df['Age'] > 120)
    if impossible_ages.any():
        print(f"Warning: {impossible_ages.sum()} patients with impossible ages")
    
    negative_survival = final_df['survival_days'] < 0
    if negative_survival.any():
        print(f"Warning: {negative_survival.sum()} patients with negative survival time")
    
    return final_df


def preprocess_with_fixes_corrected(history_path, details_path):
    """
    Processing ONLY patients with admission history
    """
    print("\nProcessing ONLY patients with admission history")
    print("-" * 50)
    
    # Load data
    history = pd.read_csv(history_path)
    details = pd.read_csv(details_path)
    
    print(f"Original - History: {len(history)} records from {history['PatientKey'].nunique()} patients")
    print(f"Original - Details: {len(details)} records for {details['PatientKey'].nunique()} patients")
    
    # Only keep patients who have admission history
    patients_with_history = set(history['PatientKey'].astype(str))
    details_filtered = details[details['PatientKey'].astype(str).isin(patients_with_history)]
    
    print(f"After filtering - Details: {len(details_filtered)} patients with admission history")
    print(f"Excluded: {len(details) - len(details_filtered)} patients with no admissions")
    
    # Now work with matched data
    details = details_filtered.copy()
    
    # Clean diagnosis codes
    history["DiagnosisCodesOnAdmission"] = history["DiagnosisCodesOnAdmission"].fillna("")
    history["DiagnosisCodesOnDischarge"] = history["DiagnosisCodesOnDischarge"].fillna("")
    history["ProcedureCodes"] = history["ProcedureCodes"].fillna("")
    
    # Convert dates
    history["AdmissionDateTime"] = pd.to_datetime(history["AdmissionDateTime"], dayfirst=True)
    history["DischargeDateTime"] = pd.to_datetime(history["DischargeDateTime"], dayfirst=True)
    
    # Ensure consistent PatientKey types
    history["PatientKey"] = history["PatientKey"].astype(str)
    details["PatientKey"] = details["PatientKey"].astype(str)
    
    print(f"Processing {len(history)} admission records for {len(details)} patients")
    
    
    def extract_icd3_fixed(code_list):
        """Better ICD code extraction with validation"""
        codes = []
        for code in str(code_list).split(","):
            code = code.strip().upper()
            if code and not code.startswith("NOT") and len(code) >= 3:
                # Validate ICD format (letter + numbers)
                if code[0].isalpha() and len(code) >= 3:
                    codes.append(code[:3])
        return list(set(codes))
    
    history["DiagnosisAdmission"] = history["DiagnosisCodesOnAdmission"].apply(extract_icd3_fixed)
    history["DiagnosisDischarge"] = history["DiagnosisCodesOnDischarge"].apply(extract_icd3_fixed)
    history["ProcedureCodesClean"] = history["ProcedureCodes"].apply(extract_icd3_fixed)
    
    # TEMPORAL FEATURES
    print("Calculating temporal features...")
    
    # 1. Length of stay
    history["LOS_days"] = (history["DischargeDateTime"] - history["AdmissionDateTime"]).dt.days
    history["LOS_days"] = history["LOS_days"].clip(lower=0)  # Fix negative LOS
    
    # 2. Admission counts and patterns
    admission_counts = history.groupby("PatientKey").size().reset_index(name='n_admissions')
    
    los_stats = history.groupby("PatientKey")["LOS_days"].agg([
        'mean', 'sum', 'std', 'min', 'max'
    ]).reset_index()
    los_stats.columns = ['PatientKey', 'avg_los_days', 'total_los_days', 
                         'std_los_days', 'min_los_days', 'max_los_days']
    los_stats['std_los_days'] = los_stats['std_los_days'].fillna(0)
    
    # 3. Readmission patterns
    history_sorted = history.sort_values(['PatientKey', 'AdmissionDateTime'])
    history_sorted['days_to_next'] = (
        history_sorted.groupby('PatientKey')['AdmissionDateTime']
        .shift(-1) - history_sorted['AdmissionDateTime']
    ).dt.days
    
    readmission_stats = history_sorted.groupby('PatientKey').agg({
        'days_to_next': [
            lambda x: (x < 30).sum(),  # 30-day readmissions
            lambda x: (x < 90).sum(),  # 90-day readmissions
            'mean'
        ]
    }).reset_index()
    readmission_stats.columns = ['PatientKey', 'readmit_30d', 'readmit_90d', 'avg_days_between']
    readmission_stats = readmission_stats.fillna(0)
    
    # 4. Time span and observation period
    time_span = history.groupby("PatientKey")["AdmissionDateTime"].agg(['min', 'max']).reset_index()
    time_span['observation_days'] = (time_span['max'] - time_span['min']).dt.days
    time_span['observation_months'] = time_span['observation_days'] / 30.44
    time_span = time_span[['PatientKey', 'observation_days', 'observation_months']]
    
    # DIAGNOSIS FEATURES
    print("Processing diagnosis features...")
    
    # Combine all diagnoses per patient
    all_diagnoses = history.groupby("PatientKey")[["DiagnosisAdmission", "DiagnosisDischarge"]].sum()
    all_diagnoses = all_diagnoses.apply(
        lambda row: list(set(row["DiagnosisAdmission"] + row["DiagnosisDischarge"])), axis=1
    )
    
    # Create binary features
    mlb_diag = MultiLabelBinarizer()
    diag_df = pd.DataFrame(
        mlb_diag.fit_transform(all_diagnoses),
        columns=[f"DIAG_{c}" for c in mlb_diag.classes_],
        index=all_diagnoses.index
    )
    
    # Remove ultra-rare codes
    optimal_threshold = calculate_optimal_diagnosis_threshold(diag_df, details['IsDeceased'])
    min_prevalence = max(5, int(optimal_threshold * len(diag_df)))
    diag_cols_to_keep = diag_df.columns[diag_df.sum() >= min_prevalence]
    diag_df = diag_df[diag_cols_to_keep]

    print(f"Using prevalence threshold: {optimal_threshold:.3f}")
    print(f"Minimum patients per diagnosis: {min_prevalence}")
    print(f"Diagnoses retained: {len(diag_cols_to_keep)}")
    
    # CHARLSON CALCULATION
    print("Calculating Charlson Comorbidity Index...")
    
    # Define Charlson weights
    charlson_weights = {
        'MI': 1, 'CHF': 1, 'PVD': 1, 'CVD': 1, 'Dementia': 1, 'COPD': 1,
        'Rheumatic': 1, 'PUD': 1, 'Liver_mild': 1, 'Diabetes': 1,
        'Diabetes_complications': 2, 'Hemiplegia': 2, 'Renal': 2, 'Cancer': 2,
        'Liver_severe': 3, 'Metastatic': 6, 'AIDS': 6
    }
    
    # Complete Charlson condition definitions
    charlson_conditions = {
        'MI': ['I21', 'I22', 'I25'],
        'CHF': ['I50'],
        'PVD': ['I71', 'I73', 'I74'],
        'CVD': ['G45', 'G46', 'I60', 'I61', 'I62', 'I63', 'I64'],
        'Dementia': ['F00', 'F01', 'F02', 'F03', 'G30'],
        'COPD': ['J40', 'J41', 'J42', 'J43', 'J44'],
        'Rheumatic': ['M05', 'M06', 'M31', 'M32', 'M33'],
        'PUD': ['K25', 'K26', 'K27', 'K28'],
        'Liver_mild': ['K70', 'K71', 'K73', 'K74'],
        'Diabetes': ['E10', 'E11', 'E13', 'E14'],
        'Diabetes_complications': ['E10', 'E11', 'E13', 'E14'],
        'Hemiplegia': ['G81', 'G82'],
        'Renal': ['N18', 'N19'],
        'Cancer': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'Liver_severe': ['K72', 'K76'],
        'Metastatic': ['C77', 'C78', 'C79', 'C80'],
        'AIDS': ['B20', 'B21', 'B22', 'B23', 'B24']
    }
    
    # Calculate Charlson score
    charlson_df = pd.DataFrame(index=all_diagnoses.index)
    charlson_df['charlson_score'] = 0
    charlson_df['n_charlson_conditions'] = 0
    
    for condition, codes in charlson_conditions.items():
        has_condition = all_diagnoses.apply(
            lambda x: any(any(diag.startswith(code) for diag in x) for code in codes)
        )
        charlson_df[f'charlson_{condition}'] = has_condition.astype(int)
        charlson_df['charlson_score'] += has_condition * charlson_weights[condition]
        charlson_df['n_charlson_conditions'] += has_condition.astype(int)
    
    print(f"Average Charlson score: {charlson_df['charlson_score'].mean():.2f}")
    
    # AGE AND SURVIVAL CALCULATIONS
    print("Processing patient demographics and survival data...")
    
    # Process dates
    details['DateOfBirth'] = pd.to_datetime(details['DateOfBirth'], dayfirst=True, errors='coerce')
    details['DateOfDeath'] = pd.to_datetime(details['DateOfDeath'], dayfirst=True, errors='coerce')
    
    # Get first admission and last discharge separately with clear naming
    first_admission = history.groupby('PatientKey')['AdmissionDateTime'].min().reset_index()
    first_admission.rename(columns={'AdmissionDateTime': 'FirstAdmissionDate'}, inplace=True)
    
    last_discharge = history.groupby('PatientKey')['DischargeDateTime'].max().reset_index()
    last_discharge.rename(columns={'DischargeDateTime': 'LastDischargeDate'}, inplace=True)
    
    # Merge dates with details
    details = details.merge(first_admission, on='PatientKey', how='left')
    details = details.merge(last_discharge, on='PatientKey', how='left')
    
    # Calculate age at first admission
    details['Age'] = (details['FirstAdmissionDate'] - details['DateOfBirth']).dt.days / 365.25
    
    # Calculate survival times properly
    details['survival_days'] = np.where(
        details['DateOfDeath'].notna(),
        (details['DateOfDeath'] - details['FirstAdmissionDate']).dt.days,
        (details['LastDischargeDate'] - details['FirstAdmissionDate']).dt.days
    )
    details['survival_days'] = details['survival_days'].clip(lower=1)  # Minimum 1 day
    
    # Death indicator
    details['IsDeceased'] = details['IsDeceased'].map({'Yes': 1, 'No': 0})
    details['event_observed'] = details['IsDeceased']
    
    print(f"Age at first admission: {details['Age'].mean():.1f} ± {details['Age'].std():.1f} years")
    print(f"Mortality rate: {details['IsDeceased'].mean():.1%}")
    print(f"Median survival time: {details['survival_days'].median():.0f} days")
    
    # MERGE ALL FEATURES
    print("Merging all features...")
    
    # Start with core patient info
    final_df = details[['PatientKey', 'Age', 'GenderCode', 'IsDeceased', 'survival_days', 'event_observed']].copy()
    
    # Merge all feature dataframes
    dataframes_to_merge = [
        admission_counts,
        los_stats, 
        readmission_stats,
        time_span,
        diag_df.reset_index(),
        charlson_df.reset_index()
    ]
    
    for df_to_merge in dataframes_to_merge:
        df_to_merge['PatientKey'] = df_to_merge['PatientKey'].astype(str)
        final_df = final_df.merge(df_to_merge, on='PatientKey', how='left')
    
    # Fill missing values
    final_df = final_df.fillna(0)
    
    # Final validation and statistics
    print(f"\nCORRECTED DATASET STATISTICS:")
    print(f"Total patients: {len(final_df):,}")
    print(f"Mortality rate: {final_df['IsDeceased'].mean():.1%}")
    print(f"Average age: {final_df['Age'].mean():.1f} years")
    print(f"Average Charlson score: {final_df['charlson_score'].mean():.2f}")
    print(f"Features: {len(final_df.columns)}")
    
    # Check for any remaining issues
    if final_df['charlson_score'].mean() == 0:
        print("WARNING: Charlson scores still 0 - check diagnosis code matching")
    
    if final_df['IsDeceased'].mean() > 0.25:
        print("WARNING: Mortality rate still high - check patient selection")
    
    return final_df


def perform_corrected_clustering(df, max_clusters=8):
    """
    Disease-focused clustering that uses only diagnosis features
    and adds prevalence heatmap + cluster profiling.
    """
    print("\nCorrected Disease Clustering")
    print("-" * 40)
    
    # Step 1: Select diagnosis features
    diag_features = [col for col in df.columns if col.startswith('DIAG_')]
    diag_data = df[diag_features]
    
    # Remove ultra-rare diagnoses (<10 patients)
    diag_prevalence = diag_data.sum()
    common_diag = diag_prevalence[diag_prevalence >= 10].index.tolist()
    print(f"Using {len(common_diag)} common diagnoses for clustering")
    
    X_clustering = df[common_diag].values
    
    # Step 2: Dimensionality reduction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, len(common_diag)//2), random_state=42)
    X_pca = pca.fit_transform(X_clustering)
    
    # UMAP with Hamming distance for binary data
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=30,
        min_dist=0.1,
        metric='hamming',
        random_state=42
    )
    X_umap = reducer.fit_transform(X_clustering)
    
    # Step 3: Find optimal number of clusters
    print("\nEvaluating optimal number of clusters...")
    cluster_range = range(2, min(max_clusters + 1, 11))
    silhouette_scores = []
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
        labels = kmeans.fit_predict(X_umap)
        sil_score = silhouette_score(X_umap, labels)
        silhouette_scores.append(sil_score)
        print(f"k={k}: Silhouette={sil_score:.3f}")
    
    optimal_k = cluster_range[0] + np.argmax(silhouette_scores)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Final clustering
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=50, random_state=42)
    df['Cluster'] = kmeans_final.fit_predict(X_umap)
    
    # Step 4: Heatmap of disease prevalence
    cluster_df = pd.DataFrame(X_clustering, columns=common_diag)
    cluster_df['Cluster'] = df['Cluster']
    heatmap_data = cluster_df.groupby('Cluster')[common_diag].mean().T
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="Reds", annot=False)
    plt.title("Disease Prevalence by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Disease")
    plt.tight_layout()
    plt.show()
    
    # Step 5: Louvain community detection
    print("\nPerforming Louvain community detection...")
    co_occurrence = diag_data.T.dot(diag_data)
    np.fill_diagonal(co_occurrence.values, 0)
    
    G = nx.from_pandas_adjacency(co_occurrence)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 10]
    G.remove_edges_from(edges_to_remove)
    
    partition = community_louvain.best_partition(G, weight='weight')
    
    # Assign patient to dominant disease community
    patient_communities = pd.DataFrame(index=df.index)
    for comm in set(partition.values()):
        diseases_in_comm = [d for d, c in partition.items() if c == comm]
        diseases_in_comm = [d for d in diseases_in_comm if d in diag_data.columns]
        if diseases_in_comm:
            patient_communities[f'DiseaseComm_{comm}'] = diag_data[diseases_in_comm].sum(axis=1)
    
    df['Disease_Community'] = patient_communities.idxmax(axis=1).str.replace('DiseaseComm_', '').astype(int)
        
    # Step 6: Profile clusters
    cluster_profiles = []
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster]
        
        profile = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Mortality_Rate': cluster_data['IsDeceased'].mean(),
            'Avg_Age': cluster_data['Age'].mean(),
            'Avg_Admissions': cluster_data['n_admissions'].mean(),
            'Avg_LOS': cluster_data['avg_los_days'].mean(),
            'Readmit_30d_Rate': (cluster_data['readmit_30d'] > 0).mean()
        }
        cluster_profiles.append(profile)
        
        print(f"\nCluster {cluster}:")
        print(f"  Size: {profile['Size']} patients")
        print(f"  Mortality: {profile['Mortality_Rate']:.1%}")
        print(f"  Avg age: {profile['Avg_Age']:.1f} years")
        print(f"  Avg admissions: {profile['Avg_Admissions']:.1f}")
        print(f"  Avg LOS: {profile['Avg_LOS']:.1f} days")
    
    # Step 7: Visualise clusters in UMAP space
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hls", optimal_k)
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=df['Cluster'],
        palette=palette,
        legend='full',
        s=60, alpha=0.8, edgecolor='k'
    )
    
    plt.title("Patient Clusters (UMAP projection)", fontsize=14)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return df, pd.DataFrame(cluster_profiles)


def create_feature_sets(df_final):
    """Create all feature set combinations for evaluation"""
    
    print("\nCREATING FEATURE SETS")
    print("-" * 40)
    
    # Transform age to categories (if not already done)
    if 'Age_Group' not in df_final.columns:
        df_final['Age_Group'] = pd.cut(df_final['Age'], 
                                bins=[0, 50, 65, 75, 85, 100], 
                                labels=['<50', '50-65', '65-75', '75-85', '85+'])
    
    age_dummies = pd.get_dummies(df_final['Age_Group'], prefix='AgeGroup')
    all_training_age_groups = age_dummies.columns.tolist()
    
    # Define different feature categories
    demographic_features = ['GenderCode']
    age_features = age_dummies.columns.tolist()
    
    temporal_features = [col for col in ['n_admissions', 'avg_los_days', 'total_los_days', 
                        'readmit_30d', 'readmit_90d', 'observation_months'] if col in df_final.columns]
    
    # Get top diagnosis features (most common)
    diag_features = [col for col in df_final.columns if col.startswith('DIAG_')]
    if diag_features:
        diag_prevalence = df_final[diag_features].sum().sort_values(ascending=False)
        top_diag_features = diag_prevalence.head(30).index.tolist()
    else:
        top_diag_features = []
    
    # Feature lists
    feature_sets = {
        'Demographics only': demographic_features + age_features,
        'Demographics + Temporal': demographic_features + age_features + temporal_features,
        'Demographics + Diagnoses': demographic_features + age_features + top_diag_features,
        'Traditional Model': demographic_features + age_features + temporal_features + top_diag_features,
    }
    
    # Only add cluster/community features if they exist
    if 'Cluster' in df_final.columns:
        cluster_features = ['Cluster_' + str(i) for i in df_final['Cluster'].unique()]
        feature_sets['Demographics + Clusters'] = demographic_features + age_features + cluster_features
        feature_sets['Diagnoses + Clusters'] = demographic_features + age_features + top_diag_features + cluster_features
    
    if 'Disease_Community' in df_final.columns:
        community_features = ['DiseaseComm_' + str(i) for i in df_final['Disease_Community'].unique()]
        feature_sets['Full Model'] = demographic_features + age_features + temporal_features + top_diag_features + cluster_features + community_features
    
    print(f"Created {len(feature_sets)} feature sets")
    for name, features in feature_sets.items():
        print(f"  {name}: {len(features)} features")
    
    return feature_sets, age_dummies


def evaluate_feature_sets_robust(feature_sets, df, y, age_dummies=None):
    import pandas as pd
    import numpy as np
    
    # Initialize model selector and validator
    model_selector = ModelSelector()
    validator = SimpleValidation(random_state=42)
    
    results = []
    
    for feature_set_name, features in feature_sets.items():
        print(f"\nEvaluating: {feature_set_name}")
        print("-" * 40)
        
        # Build feature matrix
        X_parts = []
        
        # Process different feature types
        if any('AgeGroup' in f for f in features):
            if age_dummies is not None:
                X_parts.append(age_dummies)
            features = [f for f in features if 'AgeGroup' not in f]
        
        regular_features = [f for f in features if f in df.columns]
        if regular_features:
            X_parts.append(df[regular_features])
        
        if any('Cluster_' in f for f in features):
            if 'Cluster' in df.columns:
                X_parts.append(pd.get_dummies(df['Cluster'], prefix='Cluster'))
        
        if any('DiseaseComm_' in f for f in features):
            if 'Disease_Community' in df.columns:
                X_parts.append(pd.get_dummies(df['Disease_Community'], prefix='DiseaseComm'))
        
        # Check if X_parts is empty
        if not X_parts:
            print(f"WARNING: No valid features found for {feature_set_name}, skipping...")
            continue
            
        X = pd.concat(X_parts, axis=1)
        X = X.loc[:, ~X.columns.duplicated()]  # Remove duplicates
        feature_names = X.columns.tolist()
        
        # Additional validation
        if X.empty or len(X.columns) == 0:
            print(f"WARNING: Empty feature matrix for {feature_set_name}, skipping...")
            continue
        
        print(f"Features shape: {X.shape}")
        
        # Run model comparison
        try:
            best_model_name, model_comparison = model_selector.systematic_model_comparison(X, y)
            
            # Get best model and parameters
            best_model = model_selector.models[best_model_name]
            
            if best_model_name in CONSERVATIVE_PARAM_GRIDS:
                param_grid = CONSERVATIVE_PARAM_GRIDS[best_model_name]
            else:
                param_grid = {'random_state': [42]}
            
            nested_results = validator.train_val_test_split(X, y, best_model, param_grid)
            
            result = {
                'feature_set': feature_set_name,
                'n_features': X.shape[1],
                'best_model': best_model_name,
                'auc_mean': nested_results['mean_score'],
                'auc_std': nested_results['std_score']
            }
            results.append(result)
            
            print(f"AUC: {nested_results['mean_score']:.3f} ± {nested_results['std_score']:.3f}")    
            
            # SHAP Analysis
            print("\nComputing SHAP values for the best model...")
            import shap
            try:
                # Fit the model on the full dataset
                best_model.fit(X, y)
                
                explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X)
                
                # Handle binary/multi-class
                if isinstance(shap_values, list):
                    shap_values_to_use = np.mean(np.abs(shap_values[1]), axis=0)
                else:
                    shap_values_to_use = np.mean(np.abs(shap_values), axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': X.columns.tolist(),
                    'mean_abs_shap': shap_values_to_use
                }).sort_values(by='mean_abs_shap', ascending=False)
                
                print("\nTop 10 important features by SHAP:")
                print(importance_df.head(10))
                
                # Summary plots
                shap.summary_plot(shap_values, X, plot_type="bar")
                shap.summary_plot(shap_values, X)
                
            except Exception as e:
                print(f"WARNING: SHAP analysis failed: {e}")
            
        except Exception as e:
            print(f"ERROR evaluating {feature_set_name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    if results:
        results_df = pd.DataFrame(results)
        print("\nRESULTS SUMMARY:")
        print(results_df)
        
        return results_df, best_model, feature_names
    else:
        print("WARNING: No models were successfully evaluated!")
        return pd.DataFrame()


def perform_survival_analysis(df):
    """
    Comprehensive Kaplan-Meier survival analysis
    """
    print("\nKaplan-Meier Survival Analysis")
    print("-" * 40)
    
    # Prepare data for survival analysis
    survival_data = df[['survival_days', 'event_observed', 'Cluster', 'Age', 
                       'GenderCode']].copy()
    
    # Convert survival days to months for better interpretation
    survival_data['survival_months'] = survival_data['survival_days'] / 30.44
    
    # 1. OVERALL SURVIVAL CURVE
    print("1. Overall survival analysis...")
    
    kmf_overall = KaplanMeierFitter()
    kmf_overall.fit(survival_data['survival_months'], 
                   survival_data['event_observed'], 
                   label='All Patients')
    
    # Calculate key survival statistics
    median_survival = kmf_overall.median_survival_time_
    survival_6m = kmf_overall.survival_function_at_times(6).iloc[0]
    survival_12m = kmf_overall.survival_function_at_times(12).iloc[0]
    
    print(f"Overall median survival: {median_survival:.1f} months")
    print(f"6-month survival rate: {survival_6m:.1%}")
    print(f"12-month survival rate: {survival_12m:.1%}")
    
    # 2. SURVIVAL BY CLUSTER
    print("\n2. Survival analysis by cluster...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall survival
    plt.subplot(2, 3, 1)
    kmf_overall.plot_survival_function(ax=plt.gca())
    plt.title('Overall Survival Curve')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Survival by cluster
    plt.subplot(2, 3, 2)
    
    cluster_survival_stats = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, cluster in enumerate(sorted(df['Cluster'].unique())):
        cluster_data = survival_data[survival_data['Cluster'] == cluster]
        
        if len(cluster_data) > 10:  # Only plot if sufficient data
            kmf_cluster = KaplanMeierFitter()
            kmf_cluster.fit(cluster_data['survival_months'], 
                           cluster_data['event_observed'], 
                           label=f'Cluster {cluster}')
            
            kmf_cluster.plot_survival_function(ax=plt.gca(), color=colors[i % len(colors)])
            
            # Calculate cluster statistics
            median_surv = kmf_cluster.median_survival_time_
            surv_6m = kmf_cluster.survival_function_at_times(6).iloc[0] if median_surv > 6 else 0
            
            cluster_survival_stats.append({
                'cluster': cluster,
                'n_patients': len(cluster_data),
                'events': cluster_data['event_observed'].sum(),
                'median_survival': median_surv,
                'survival_6m': surv_6m,
                'mortality_rate': cluster_data['event_observed'].mean()
            })
    
    plt.title('Survival by Disease Cluster')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. LOG-RANK TESTS
    print("\n3. Statistical significance testing...")
    
    # Overall log-rank test across all clusters
    cluster_groups = []
    survival_times = []
    events = []
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = survival_data[survival_data['Cluster'] == cluster]
        if len(cluster_data) > 10:
            cluster_groups.extend([cluster] * len(cluster_data))
            survival_times.extend(cluster_data['survival_months'].tolist())
            events.extend(cluster_data['event_observed'].tolist())
    
    if len(set(cluster_groups)) > 1:
        test_result = multivariate_logrank_test(
            survival_times, cluster_groups, events
        )
        print(f"Log-rank test across clusters: p-value = {test_result.p_value:.6f}")
        
        if test_result.p_value < 0.001:
            print("*** Highly significant differences in survival between clusters ***")
        elif test_result.p_value < 0.05:
            print("** Significant differences in survival between clusters **")
    
    # 4. SURVIVAL BY RISK FACTORS
    
    # Plot 3: Survival by age groups
    plt.subplot(2, 3, 3)
    
    df['age_group'] = pd.cut(df['Age'], 
                            bins=[0, 65, 75, 85, 100], 
                            labels=['<65', '65-75', '75-85', '85+'])
    
    for age_group in df['age_group'].cat.categories:
        age_data = survival_data[df['age_group'] == age_group]
        if len(age_data) > 10:
            kmf_age = KaplanMeierFitter()
            kmf_age.fit(age_data['survival_months'], 
                       age_data['event_observed'], 
                       label=f'Age {age_group}')
            kmf_age.plot_survival_function(ax=plt.gca())
    
    plt.title('Survival by Age Group')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Survival by Charlson score
    plt.subplot(2, 3, 4)
    
    df['charlson_group'] = pd.cut(df['charlson_score'], 
                                 bins=[-1, 0, 2, 4, 20], 
                                 labels=['0', '1-2', '3-4', '5+'])
    
    for charlson_group in df['charlson_group'].cat.categories:
        charlson_data = survival_data[df['charlson_group'] == charlson_group]
        if len(charlson_data) > 10:
            kmf_charlson = KaplanMeierFitter()
            kmf_charlson.fit(charlson_data['survival_months'], 
                           charlson_data['event_observed'], 
                           label=f'Charlson {charlson_group}')
            kmf_charlson.plot_survival_function(ax=plt.gca())
    
    plt.title('Survival by Charlson Score')
    plt.xlabel('Time (months)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Use Nelson-Aalen for cumulative hazard
    plt.subplot(2, 3, 5)
    from lifelines import NelsonAalenFitter
    naf = NelsonAalenFitter()
    naf.fit(survival_data['survival_months'], survival_data['event_observed'])
    naf.plot_cumulative_hazard(ax=plt.gca())
    plt.title('Cumulative Hazard Function')
    plt.xlabel('Time (months)')
    plt.ylabel('Cumulative Hazard')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Risk table
    plt.subplot(2, 3, 6)
    
    # Create risk table data
    time_points = [0, 6, 12, 18, 24, 36]
    risk_table_data = []
    
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = survival_data[survival_data['Cluster'] == cluster]
        if len(cluster_data) > 10:
            kmf_cluster = KaplanMeierFitter()
            kmf_cluster.fit(cluster_data['survival_months'], 
                           cluster_data['event_observed'])
            
            at_risk = []
            for tp in time_points:
                try:
                    n_at_risk = kmf_cluster.durations[kmf_cluster.durations >= tp].shape[0]
                    at_risk.append(n_at_risk)
                except:
                    at_risk.append(0)
            
            risk_table_data.append([f'Cluster {cluster}'] + at_risk)
    
    # Display as table
    plt.axis('off')
    if risk_table_data:
        table = plt.table(cellText=risk_table_data,
                         colLabels=['Cluster'] + [f'{tp}m' for tp in time_points],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.title('Number at Risk by Time Point')
    
    plt.tight_layout()
    plt.show()
    
    # 5. COX PROPORTIONAL HAZARDS MODEL
    print("\n4. Cox Proportional Hazards Analysis...")
    
    # Prepare data for Cox regression
    cox_data = df[['survival_days', 'event_observed', 'Age', 
                   'GenderCode', 'n_admissions', 'avg_los_days']].copy()
    
    # Add cluster dummies
    cluster_dummies = pd.get_dummies(df['Cluster'], prefix='Cluster')
    cox_data = pd.concat([cox_data, cluster_dummies], axis=1)
    
    # Remove reference cluster (largest one)
    largest_cluster = df['Cluster'].value_counts().index[0]
    cox_data = cox_data.drop(f'Cluster_{largest_cluster}', axis=1)
    
    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data, duration_col='survival_days', event_col='event_observed')
        
        print("Cox Proportional Hazards Results:")
        print(cph.summary[['coef', 'exp(coef)', 'p']])
        
        # Plot hazard ratios
        plt.figure(figsize=(10, 6))
        cph.plot(hazard_ratios=True)
        plt.title('Hazard Ratios from Cox Regression')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Cox regression failed: {e}")
        print("This may be due to insufficient follow-up time or data issues")
    
    # Print summary statistics
    print("\nSURVIVAL ANALYSIS SUMMARY:")
    print("=" * 50)
    
    survival_summary = pd.DataFrame(cluster_survival_stats)
    if len(survival_summary) > 0:
        print(survival_summary.to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"• Median survival: {median_survival:.1f} months")
    print(f"• 6-month survival: {survival_6m:.1%}")
    print(f"• 12-month survival: {survival_12m:.1%}")
    print(f"• Total deaths: {df['event_observed'].sum()} of {len(df)} patients")
    
    return survival_summary


def create_best_model_dashboard(df, cluster_df, results_df):
    """
    Simple dashboard for the best model
    """
    import matplotlib.pyplot as plt
    
    
    # Get best model info
    best_idx = results_df['auc_mean'].idxmax()
    best_model_name = results_df.loc[best_idx, 'feature_set']
    best_auc = results_df.loc[best_idx, 'auc_mean']
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model Performance Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(results_df)), results_df['auc_mean'], 
                   yerr=results_df['auc_std'], capsize=5)
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['feature_set'], rotation=45, ha='right')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_ylim(0.5, 1.0)
    
    # Highlight best model
    bars[best_idx].set_color('darkgreen')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good Performance')
    ax1.legend()
    
    # 2. Cluster Mortality Analysis
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Mortality_Rate' in cluster_df.columns:
        colors = ['green' if x < 0.15 else 'orange' if x < 0.25 else 'red' 
                 for x in cluster_df['Mortality_Rate']]
        bars2 = ax2.bar(cluster_df['Cluster'], cluster_df['Mortality_Rate'], color=colors)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Mortality Rate')
        ax2.set_title('Mortality Rate by Cluster', fontweight='bold')
        ax2.axhline(y=df['IsDeceased'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall')
        ax2.legend()
        
        # Add cluster sizes as labels
        for i, (cluster, size) in enumerate(zip(cluster_df['Cluster'], cluster_df['Size'])):
            ax2.text(i, cluster_df.loc[i, 'Mortality_Rate'] + 0.01, f'n={size}', 
                    ha='center', va='bottom', fontsize=9)
    
    # 3. Age Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'Age' in df.columns:
        ax3.hist(df['Age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
        ax3.set_xlabel('Age (years)')
        ax3.set_ylabel('Count')
        ax3.set_title('Age Distribution', fontweight='bold')
        ax3.legend()
    
    # 4. Cluster Size Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'Size' in cluster_df.columns:
        ax4.pie(cluster_df['Size'], labels=[f'Cluster {c}' for c in cluster_df['Cluster']], 
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Cluster Size Distribution', fontweight='bold')
    
    # 5. Admissions Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    if 'n_admissions' in df.columns and 'Cluster' in df.columns:
        df.boxplot(column='n_admissions', by='Cluster', ax=ax5)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Number of Admissions')
        ax5.set_title('Admissions by Cluster', fontweight='bold')
        plt.sca(ax5)
        plt.xticks(rotation=0)
    
    # 6. Length of Stay Analysis
    ax6 = fig.add_subplot(gs[1, 2])
    if 'avg_los_days' in df.columns and 'Cluster' in df.columns:
        df.boxplot(column='avg_los_days', by='Cluster', ax=ax6)
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Average LOS (days)')
        ax6.set_title('Length of Stay by Cluster', fontweight='bold')
        plt.sca(ax6)
        plt.xticks(rotation=0)
    
    # 7. Feature Importance 
    ax7 = fig.add_subplot(gs[2, 0])
    # Simple feature importance based on what's available
    feature_importance = []
    if 'n_admissions' in df.columns:
        corr = df[['n_admissions', 'IsDeceased']].corr().iloc[0, 1]
        feature_importance.append(('Admissions', abs(corr)))
    if 'avg_los_days' in df.columns:
        corr = df[['avg_los_days', 'IsDeceased']].corr().iloc[0, 1]
        feature_importance.append(('Avg LOS', abs(corr)))
    if 'Age' in df.columns:
        corr = df[['Age', 'IsDeceased']].corr().iloc[0, 1]
        feature_importance.append(('Age', abs(corr)))
    
    if feature_importance:
        features, importances = zip(*sorted(feature_importance, key=lambda x: x[1], reverse=True))
        ax7.barh(features, importances)
        ax7.set_xlabel('Correlation with Mortality')
        ax7.set_title('Feature Correlations', fontweight='bold')
    
    # 8. Risk Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    if 'Risk_Level' in cluster_df.columns:
        risk_counts = cluster_df.groupby('Risk_Level')['Size'].sum()
        ax8.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        ax8.set_title('Risk Level Distribution', fontweight='bold')
    else:
        # Create risk levels based on mortality
        risk_levels = []
        for _, row in cluster_df.iterrows():
            if 'Mortality_Rate' in row:
                if row['Mortality_Rate'] >= 0.25:
                    risk_levels.append('High')
                elif row['Mortality_Rate'] >= 0.15:
                    risk_levels.append('Medium')
                else:
                    risk_levels.append('Low')
        
        if risk_levels:
            cluster_df_temp = cluster_df.copy()
            cluster_df_temp['Risk_Level'] = risk_levels
            risk_counts = cluster_df_temp.groupby('Risk_Level')['Size'].sum()
            ax8.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
            ax8.set_title('Risk Level Distribution', fontweight='bold')
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
BEST MODEL SUMMARY
{'='*25}

Model: {best_model_name}
AUC Score: {best_auc:.3f}

Dataset Statistics:
• Patients: {len(df):,}
• Mortality: {df['IsDeceased'].mean():.1%}
• Avg Age: {df['Age'].mean():.1f} years
• Clusters: {len(cluster_df)}

Cluster Mortality Rates:
"""
    
    for _, row in cluster_df.iterrows():
        if 'Mortality_Rate' in row:
            summary_text += f"• Cluster {row['Cluster']}: {row['Mortality_Rate']:.1%} (n={row['Size']})\n"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('CONDENSED PIPELINE: Best Model Dashboard\n' + 
                'Disease Clusters for Personalised Risk Prediction',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('condensed_best_model_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'best_model_name': best_model_name,
        'best_auc': best_auc
    }


def run_condensed_pipeline(history_path,details_path):
    """
    Main function that runs the complete analysis pipeline
    """
    try:
        print("\nStep 1: Loading and cleaning data...")
        df = preprocess_with_fixes_corrected(history_path,details_path
            
        )
        
        # Validate final dataset
        df = validate_final_dataset(df)
        
        print(f"\nCleaned data: {len(df)} patients")
        print(f"Mortality rate: {df['IsDeceased'].mean():.1%}")
        
        print("\nStep 2: Performing clustering analysis...")
        df, cluster_df = perform_corrected_clustering(df)
        
        print("\nStep 3: Creating feature sets...")
        feature_sets, age_dummies = create_feature_sets(df)
        
        print("\nStep 4: Evaluating models...")
        df_combined = pd.concat([df, age_dummies], axis=1)
        y = df_combined['IsDeceased']
        results_df, best_model, feature_names = evaluate_feature_sets_robust(feature_sets, df_combined, y, age_dummies)
        
        print("\nStep 5: Performing survival analysis...")
        survival_summary = perform_survival_analysis(df)
        
        print("\nStep 6: Creating dashboard...")
        dashboard_results = create_best_model_dashboard(df, cluster_df, results_df)
        
        # Final summary
        print(f"\nFINAL RESULTS:")
        print(f"Dataset: {len(df)} patients")
        print(f"Mortality rate: {df['IsDeceased'].mean():.1%}")
        print(f"Number of clusters: {len(cluster_df)}")
        print(f"Best model: {dashboard_results['best_model_name']}")
        print(f"Best AUC: {dashboard_results['best_auc']:.3f}")
        print(f"Dashboard saved as 'condensed_best_model_dashboard.png'")
        
        return df, cluster_df, results_df, survival_summary
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting patient data analysis pipeline...")
    results = run_condensed_pipeline()
    
    if results is not None:
        df, cluster_df, results_df, survival_summary = results
        print("\nPipeline completed successfully!")
        print("Available outputs:")
        print("- df: Main dataset with clusters and features")
        print("- cluster_df: Cluster profiles and statistics") 
        print("- results_df: Model performance comparison")
        print("- survival_summary: Survival analysis results")
    else:
        print("\nPipeline failed. Check error messages above.")
