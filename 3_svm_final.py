import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.patheffects as PathEffects

# --- CONFIGURATION (Keep your existing paths) ---
MASTER_FILE = r"D:\Inference\nxm\grid_master_16x16.csv"
GRID_SIZE = 16
SESSION_CSV = r"D:\Inference\nxm\Session Details - Sheet1.csv"
MASK_IMAGE = r"D:\Inference\nxm\1758306529.51_mask.png"
NETD_THRESHOLD = 0.0016 

def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('variance_thresh', VarianceThreshold(threshold=NETD_THRESHOLD)),
        ('kbest', SelectKBest(score_func=f_classif, k='all')), 
        # MUST be linear kernel to extract spatial feature importances
        ('svm', SVC(kernel='linear', class_weight='balanced', random_state=42))
    ])

# [Keep your exact plot_visualizations and analyze_environment functions here]
# ...

def main():
    print(f"Loading master dataset from {MASTER_FILE}...")
    df = pd.read_csv(MASTER_FILE)
    
    if os.path.exists(SESSION_CSV):
        session_df = pd.read_csv(SESSION_CSV)
        session_df.columns = session_df.columns.str.strip() 
        session_df['Session_ID'] = session_df['Session_ID'].astype(str)
        df['session_id'] = df['session_id'].astype(str)
        df = df.merge(session_df, left_on='session_id', right_on='Session_ID', how='left')
        df['target_label'] = df['App_Name'].fillna(df['app_id']).astype(str)
    else:
        return

    df['target_label'] = df['target_label'].str.strip()
    rogue_classes = ['7_old', '9_old', '53_old']
    df = df[~df['target_label'].isin(rogue_classes)]

    clean_idx = df.dropna(subset=['session_id', 'target_label']).index
    y = df.loc[clean_idx, 'target_label'].astype(str)
    groups = df.loc[clean_idx, 'session_id'].astype(str) 

    meta_cols = ['timestamp', 'app_id', 'session_id', 'target_label', 'App_Name', 'Session_ID', 'App_ID', 'VR_Refresh_Rate', 'battery_at_start', 'avg_dist_from_camera (cm)', 'ambient_temp (F)']
    cols_to_drop = [c for c in meta_cols if c in df.columns]
    
    features_df = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
    features_df = features_df.loc[clean_idx].ffill().bfill().dropna(axis=1, how='all')
    
    final_clean_idx = features_df.dropna().index
    X = features_df.loc[final_clean_idx]
    y = y.loc[final_clean_idx]
    groups = groups.loc[final_clean_idx]

    logo = LeaveOneGroupOut()
    y_true_sessions = []
    y_pred_sessions = []
    global_importances = np.zeros(X.shape[1])
    fold = 1
    
    print("\nStarting Linear SVM Leave-One-Session-Out Cross Validation...")
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_session = groups.iloc[test_idx].iloc[0]
        actual_app = y_test.iloc[0]

        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        frame_preds = pipeline.predict(X_test)
        majority_vote = Counter(frame_preds).most_common(1)[0][0]
        
        y_true_sessions.append(actual_app)
        y_pred_sessions.append(majority_vote)
        
        is_correct = int(majority_vote == actual_app)
        frame_acc = accuracy_score(y_test, frame_preds)
        vote_correct = "✓" if is_correct else "✗"
        
        print(f"Fold {fold:02d} | Session: {test_session:^4} | True App: {actual_app:<15} | Pred: {majority_vote:<15} [{vote_correct}] | Frame Acc: {frame_acc:.2f}")
        
        # --- FEATURE IMPORTANCE EXTRACTION (Linear SVM) ---
        svm_model = pipeline.named_steps['svm']
        kbest_selector = pipeline.named_steps['kbest']
        var_thresh = pipeline.named_steps['variance_thresh']
        
        # Average the absolute weights across all One-vs-One hyperplanes to find which cells govern the most decisions
        fold_importances = np.mean(np.abs(svm_model.coef_), axis=0)
        
        kbest_mask = kbest_selector.get_support()
        var_survivors_importances = np.zeros(len(kbest_mask))
        var_survivors_importances[kbest_mask] = fold_importances
        
        var_mask = var_thresh.get_support()
        full_fold_importances = np.zeros(len(var_mask))
        full_fold_importances[var_mask] = var_survivors_importances
        
        global_importances += full_fold_importances
        fold += 1

    print("\n" + "="*50)
    print("--- OVERALL METRICS (LINEAR SVM) ---")
    print("="*50)
    print(f"\nGlobal Session Accuracy: {accuracy_score(y_true_sessions, y_pred_sessions):.4f}")
    print("\nClassification Report (Session Level):")
    print(classification_report(y_true_sessions, y_pred_sessions, zero_division=0))

    avg_importances = global_importances / groups.nunique()
    classes = np.unique(y_true_sessions)
    
    # Needs your plot_visualizations function from the original script
    # plot_visualizations(y_true_sessions, y_pred_sessions, avg_importances, classes, X.columns)

if __name__ == "__main__":
    main()