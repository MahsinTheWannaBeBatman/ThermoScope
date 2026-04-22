import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.patheffects as PathEffects

# --- CONFIGURATION ---
MASTER_FILE = r"D:\Inference\nxm\grid_master_4x4.csv"
GRID_SIZE = 4

# MASTER_FILE = r"D:\Inference\nxm\grid_master_8x8.csv"
# GRID_SIZE = 8

# MASTER_FILE = r"D:\Inference\nxm\grid_master_12x12.csv"
# GRID_SIZE = 12

# MASTER_FILE = r"D:\Inference\nxm\grid_master_14x14.csv"
# GRID_SIZE = 14

# MASTER_FILE = r"D:\Inference\nxm\grid_master_16x16.csv"
# GRID_SIZE = 16

# MASTER_FILE = r"D:\Inference\nxm\grid_master_20x20.csv"
# GRID_SIZE = 20

# MASTER_FILE = r"D:\Inference\nxm\grid_master_24x24.csv"
# GRID_SIZE = 24


SESSION_CSV = r"D:\Inference\nxm\Session Details - Sheet1.csv"
MASK_IMAGE = r"D:\Inference\nxm\1758306529.51_mask.png"
NETD_THRESHOLD = 0.0016 # 40mK squared

def create_pipeline():
    """Creates a standardized pipeline to prevent data leakage."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('variance_thresh', VarianceThreshold(threshold=NETD_THRESHOLD)),
        ('kbest', SelectKBest(score_func=f_classif, k='all')), 
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
    ])

# def plot_visualizations(y_true, y_pred, feature_importances, classes):
#     """Generates confusion matrix, top features, and grid overlay."""
#     print("\nGenerating Visualizations...")
    
#     # 1. Confusion Matrix
#     plt.figure(figsize=(10, 8))
#     cm = confusion_matrix(y_true, y_pred, labels=classes)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
#     disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
#     plt.title('Session-Level Confusion Matrix')
#     plt.tight_layout()
#     plt.show()

#     # 2. Top Feature Importances (Bar Chart)
#     plt.figure(figsize=(12, 6))
#     top_indices = np.argsort(feature_importances)[-20:]
#     plt.barh(range(20), feature_importances[top_indices], align='center')
#     plt.yticks(range(20), [f"Grid Cell {i}" for i in top_indices])
#     plt.xlabel('Importance')
#     plt.title('Top 20 Spatial Feature Importances')
#     plt.tight_layout()
#     plt.show()

#     # 3. Image Overlay
#     if os.path.exists(MASK_IMAGE):
#         importance_grid = feature_importances.reshape((GRID_SIZE, GRID_SIZE))
#         if importance_grid.max() > 0:
#             importance_grid = (importance_grid - importance_grid.min()) / (importance_grid.max() - importance_grid.min())
        
#         img = cv2.imread(MASK_IMAGE)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         heatmap = cv2.resize(importance_grid, (img.shape[1], img.shape[0]))
#         heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
#         overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        
#         plt.figure(figsize=(10, 10))
#         plt.imshow(overlay)
#         plt.title('Thermal Feature Importance Overlay')
#         plt.axis('off')
#         plt.show()
#     else:
#         print(f"Warning: Image {MASK_IMAGE} not found. Skipping overlay.")

def plot_visualizations(y_true, y_pred, feature_importances, classes, feature_names):
    """Generates confusion matrix, top features, and a fully annotated, cropped grid overlay."""
    print("\nGenerating Visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Session-Level Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # 2. Top Feature Importances (Bar Chart)
    # plt.figure(figsize=(12, 6))
    # top_indices = np.argsort(feature_importances)[-20:]
    # top_feature_names = [feature_names[i] for i in top_indices]
    # plt.barh(range(20), feature_importances[top_indices], align='center')
    # plt.yticks(range(20), top_feature_names)
    # plt.xlabel('Raw Importance Score')
    # plt.title('Top 20 Feature Importances')
    # plt.tight_layout()
    # plt.show()

    # 3. Annotated Grid Overlay (FIXED: Cropped to Mask)
    if os.path.exists(MASK_IMAGE):
        expected_pixels = GRID_SIZE * GRID_SIZE
        spatial_importances = feature_importances[:expected_pixels]
        
        if len(spatial_importances) == expected_pixels:
            importance_grid = spatial_importances.reshape((GRID_SIZE, GRID_SIZE))
            
            # Normalize scores to 0-100 for readability
            max_imp = importance_grid.max()
            if max_imp > 0:
                scores_grid = (importance_grid / max_imp) * 100
            else:
                scores_grid = importance_grid
                
            img = cv2.imread(MASK_IMAGE)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # --- THE FIX: ISOLATE AND CROP TO THE MASK ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Threshold to isolate the mask from the background
            _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Grab the largest contour (the headset mask itself)
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                # Crop image to the exact bounding box of the headset
                display_img = img_rgb[y:y+h, x:x+w]
            else:
                # Fallback just in case
                display_img = img_rgb
                h, w = display_img.shape[:2]
            
            cell_h = h / GRID_SIZE
            cell_w = w / GRID_SIZE
            
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(display_img)
            
            # Determine thresholds for highlighting
            flat_scores = scores_grid.flatten()
            top_threshold = np.percentile(flat_scores, 80)
            bottom_threshold = np.percentile(flat_scores, 20)
            
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    score = scores_grid[i, j]
                    
                    x_start = j * cell_w
                    y_start = i * cell_h
                    
                    # Default cell styling
                    edge_color = 'white'
                    linewidth = 0.5
                    face_color = 'none'
                    alpha = 0.0
                    
                    # Highlight Top 5% (Important - Red)
                    if score >= top_threshold and score > 0:
                        edge_color = 'green'
                        linewidth = 2.5
                        face_color = 'green'
                        alpha = 0.3
                    # Highlight Bottom 5% (Least Important - Blue)
                    elif score <= bottom_threshold:
                        edge_color = 'red'
                        linewidth = 1.0
                        face_color = 'red'
                        alpha = 0.2
                        
                    # Draw background tint
                    if face_color != 'none':
                        rect_bg = plt.Rectangle((x_start, y_start), cell_w, cell_h, 
                                                linewidth=0, edgecolor='none', 
                                                facecolor=face_color, alpha=alpha)
                        ax.add_patch(rect_bg)
                    
                    # Draw cell border
                    rect_border = plt.Rectangle((x_start, y_start), cell_w, cell_h, 
                                                linewidth=linewidth, edgecolor=edge_color, 
                                                facecolor='none', alpha=0.7)
                    ax.add_patch(rect_border)
                    
                    # Stamp the score text (Only for cells with > 1% relative importance)
                    txt = ax.text(x_start + cell_w/2, y_start + cell_h/2, f"{score:.0f}", 
                                    color='white', ha='center', va='center', 
                                    fontsize=7, fontweight='bold')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

            plt.title('Thermal Leakage Grid (Cropped to Mask)\n(Green: Top 20% | Red: Bottom 20% | Scores: Normalized 0-100)')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
             print(f"Error: Could not reshape spatial features. Expected {expected_pixels}, got {len(spatial_importances)}")
    else:
        print(f"Warning: Image {MASK_IMAGE} not found. Skipping overlay.")
        
        
def analyze_environment(session_results_df):
    """Calculates statistics for optimal distance and temperature."""
    print("\n" + "="*50)
    print("--- OPTIMAL ENVIRONMENT PARAMETERS ---")
    print("="*50)
    
    env_df = session_results_df.dropna(subset=['Temp', 'Distance'])
    
    try:
        env_df['Temp_Range'] = pd.qcut(env_df['Temp'], q=3, precision=1, duplicates='drop')
        env_df['Dist_Range'] = pd.qcut(env_df['Distance'], q=3, precision=1, duplicates='drop')
        
        print("\nAccuracy by Ambient Temperature Range (°F):")
        temp_stats = env_df.groupby('Temp_Range', observed=False)['Correct'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Sessions'})
        temp_stats['Accuracy'] = (temp_stats['Accuracy'] * 100).round(2).astype(str) + '%'
        print(temp_stats)
        
        print("\nAccuracy by Distance from Camera (cm):")
        dist_stats = env_df.groupby('Dist_Range', observed=False)['Correct'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Sessions'})
        dist_stats['Accuracy'] = (dist_stats['Accuracy'] * 100).round(2).astype(str) + '%'
        print(dist_stats)
        
    except ValueError as e:
        print(f"Not enough variation in environment data to create bins: {e}")

def main():
    print(f"Loading master dataset from {MASTER_FILE}...")
    df = pd.read_csv(MASTER_FILE)
    
    if os.path.exists(SESSION_CSV):
        print(f"Loading and merging session details from {SESSION_CSV}...")
        session_df = pd.read_csv(SESSION_CSV)
        session_df.columns = session_df.columns.str.strip() 
        session_df['Session_ID'] = session_df['Session_ID'].astype(str)
        df['session_id'] = df['session_id'].astype(str)
        
        # Merge on correctly cased keys
        df = df.merge(session_df, left_on='session_id', right_on='Session_ID', how='left')
        df['target_label'] = df['App_Name'].fillna(df['app_id']).astype(str)
    else:
        print(f"Error: {SESSION_CSV} not found.")
        return

    df['target_label'] = df['target_label'].str.strip()

    # --- REQUIREMENT 2: Drop the Home App ---
    # initial_rows = len(df)
    # df = df[~df['target_label'].str.lower().str.contains('home')]
    # print(f"Dropped {initial_rows - len(df)} rows belonging to 'Home' classes.")

    # Filter out rogue classes
    rogue_classes = ['7_old', '9_old', '53_old']
    initial_rows = len(df)
    df = df[~df['target_label'].isin(rogue_classes)]
    if len(df) < initial_rows:
        print(f"Dropped {initial_rows - len(df)} rows belonging to invalid classes.")

    print("\nInterpolating missing temporal data...")
    clean_idx = df.dropna(subset=['session_id', 'target_label']).index
    y_final = df.loc[clean_idx, 'target_label'].astype(str)
    groups = df.loc[clean_idx, 'session_id'].astype(str) 

    # Isolate Features and FORCE STRICT NUMERIC TYPES
    meta_cols = ['timestamp', 'app_id', 'session_id', 'target_label', 'App_Name', 'Session_ID', 'App_ID', 'VR_Refresh_Rate', 'battery_at_start', 'avg_dist_from_camera (cm)', 'ambient_temp (F)']
    cols_to_drop = [c for c in meta_cols if c in df.columns]
    
    features_df = df.drop(columns=cols_to_drop)
    features_df = features_df.select_dtypes(include=[np.number])
    
    # Clean the isolated numeric features
    features_df = features_df.loc[clean_idx]
    features_df = features_df.ffill().bfill()
    features_df = features_df.dropna(axis=1, how='all')
    
    final_clean_idx = features_df.dropna().index
    X = features_df.loc[final_clean_idx]
    y = y_final.loc[final_clean_idx]
    groups = groups.loc[final_clean_idx]

    print(f"Feature count (strictly numeric): {X.shape[1]}")
    print(f"Total sessions for LOSO Cross-Validation: {groups.nunique()}")

    logo = LeaveOneGroupOut()
    
    y_true_sessions = []
    y_pred_sessions = []
    session_results = []
    
    global_importances = np.zeros(X.shape[1])
    fold = 1
    
    print("\nStarting Leave-One-Session-Out Cross Validation...")
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
        
        # Collect Environment Data
        session_info = df[df['session_id'] == test_session].iloc[0]
        session_results.append({
            'Session_ID': test_session,
            'Actual': actual_app,
            'Predicted': majority_vote,
            'Correct': is_correct,
            'Distance': session_info.get('avg_dist_from_camera (cm)', np.nan),
            'Temp': session_info.get('ambient_temp (F)', np.nan)
        })

        # Map Feature Importances
        rf_model = pipeline.named_steps['rf']
        kbest_selector = pipeline.named_steps['kbest']
        var_thresh = pipeline.named_steps['variance_thresh']
        
        fold_importances = rf_model.feature_importances_
        
        kbest_mask = kbest_selector.get_support()
        var_survivors_importances = np.zeros(len(kbest_mask))
        var_survivors_importances[kbest_mask] = fold_importances
        
        var_mask = var_thresh.get_support()
        full_fold_importances = np.zeros(len(var_mask))
        full_fold_importances[var_mask] = var_survivors_importances
        
        global_importances += full_fold_importances
        fold += 1

    print("\n" + "="*50)
    print("--- OVERALL METRICS (FLAT CLASSIFICATION) ---")
    print("="*50)
    
    session_acc = accuracy_score(y_true_sessions, y_pred_sessions)
    print(f"\nGlobal Session Accuracy: {session_acc:.4f}")
    print("\nClassification Report (Session Level):")
    print(classification_report(y_true_sessions, y_pred_sessions, zero_division=0))

    # avg_importances = global_importances / logo.get_n_splits(groups)
    avg_importances = global_importances / groups.nunique()
    classes = np.unique(y_true_sessions)

    # plot_visualizations(y_true_sessions, y_pred_sessions, avg_importances, classes)
    
    plot_visualizations(y_true_sessions, y_pred_sessions, avg_importances, classes, X.columns)
    
    results_df = pd.DataFrame(session_results)
    analyze_environment(results_df)

if __name__ == "__main__":
    main()