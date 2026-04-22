import os
import glob
import pandas as pd
import numpy as np

# INPUT_DIR = r"D:\Inference\exp1_nxm\Grid_Output_1x1"
# MASTER_FILE = r"D:\Inference\exp1_nxm\grid_master_1x1.csv"
# GRID_ROWS, GRID_COLS = 1, 1

INPUT_DIR = r"D:\Inference\exp1_nxm\Grid_Output_2x2"
MASTER_FILE = r"D:\Inference\exp1_nxm\grid_master_2x2.csv"
GRID_ROWS, GRID_COLS = 2, 2

# INPUT_DIR = r"D:\Inference\exp1_nxm\Grid_Output_4x4"
# MASTER_FILE = r"D:\Inference\exp1_nxm\grid_master_4x4.csv"
# GRID_ROWS, GRID_COLS = 4, 4

# INPUT_DIR = r"D:\Inference\exp1_nxm\Grid_Output_16x16"
# MASTER_FILE = r"D:\Inference\exp1_nxm\grid_master_16x16.csv"
# GRID_ROWS, GRID_COLS = 16, 16

def calculate_spatial_gradients(df):
    """Calculates difference between cell mean and neighbor means."""
    new_cols = {}
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            cell_mean_col = f"cell_{i}_{j}_mean"
            if cell_mean_col not in df.columns: continue
            
            # Find valid neighbors (up, down, left, right)
            neighbors = []
            if i > 0: neighbors.append(f"cell_{i-1}_{j}_mean")
            if i < GRID_ROWS - 1: neighbors.append(f"cell_{i+1}_{j}_mean")
            if j > 0: neighbors.append(f"cell_{i}_{j-1}_mean")
            if j < GRID_COLS - 1: neighbors.append(f"cell_{i}_{j+1}_mean")
                
            # Compute average of valid neighbors for each row
            neighbor_means = df[neighbors].mean(axis=1)
            new_cols[f"cell_{i}_{j}_spatial_grad"] = df[cell_mean_col] - neighbor_means
            
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

def calculate_temporal_deltas(df):
    """Calculates 5s and 30s heating/cooling rates."""
    new_cols = {}
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            cell_mean_col = f"cell_{i}_{j}_mean"
            if cell_mean_col not in df.columns: continue
            
            # Since data is 1 Hz, periods=5 means 5 seconds ago
            new_cols[f"cell_{i}_{j}_delta_5s"] = df[cell_mean_col].diff(periods=5)
            new_cols[f"cell_{i}_{j}_delta_30s"] = df[cell_mean_col].diff(periods=30)
            
    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

def main():
    all_sessions = []
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*_base.csv"))
    
    for file in csv_files:
        df = pd.read_csv(file)
        
        # 1. Spatiotemporal Engineering
        df = calculate_spatial_gradients(df)
        df = calculate_temporal_deltas(df)
        
        all_sessions.append(df)
        print(f"Engineered features for: {os.path.basename(file)}")
        
    # Combine into Master
    master_df = pd.concat(all_sessions, ignore_index=True)
    master_df.to_csv(MASTER_FILE, index=False)
    print(f"\nMasterfile created: {MASTER_FILE} with shape {master_df.shape}")

if __name__ == "__main__":
    main()