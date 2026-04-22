import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# Import your UNet exactly as done in infer_all_quadrant.py
try:
    from unet_best_model import UNet_2d
except ImportError:
    sys.exit("Error: Could not import UNet_2d from unet_best_model.py.")

# ================= CONFIGURATION =================
MODEL_PATH = r"D:\Inference\exp1_nxm\unet_best_model.pth"

INPUT_ROOT = r'C:\Users\mahsi\OneDrive - University of Texas at San Antonio\Personal\Code\snapshots'

# OUTPUT_ROOT = r'D:\Inference\exp1_nxm\Grid_Output_1x1'
# GRID_ROWS, GRID_COLS = 1, 1

OUTPUT_ROOT = r'D:\Inference\exp1_nxm\Grid_Output_2x2'
GRID_ROWS, GRID_COLS = 2, 2

# OUTPUT_ROOT = r'D:\Inference\exp1_nxm\Grid_Output_4x4'
# GRID_ROWS, GRID_COLS = 4, 4

# OUTPUT_ROOT = r'D:\Inference\exp1_nxm\Grid_Output_16x16'
# GRID_ROWS, GRID_COLS = 16, 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IN_CHANNELS = 1 
CLASS_NUM = 2 


# Geometric Filtering Thresholds
MIN_ASPECT_RATIO = 1.3  # Headset is wider than it is tall
MAX_ASPECT_RATIO = 2.8  # Drop if > 2.8 (heavily rotated/distorted)
MIN_SOLIDITY = 0.80     # Drop if < 0.80 (mask grabbed straps or face)

transform = transforms.Compose([transforms.ToTensor()])

def load_model(path):
    model = UNet_2d(in_chns=IN_CHANNELS, class_num=CLASS_NUM)
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    except FileNotFoundError:
        sys.exit(f"Error: Model file not found at {path}")
    model.to(DEVICE)
    model.eval()
    return model

def process_data():
    print(f"Loading model on {DEVICE}...")
    model = load_model(MODEL_PATH)
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Walk through directories
    for root, dirs, files in os.walk(INPUT_ROOT):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # If no images in this folder, skip (e.g., root folder or app folder)
        if not image_files:
            continue
            
        # VERY IMPORTANT: Sort files to maintain the 1Hz time-series integrity
        image_files.sort()
        
        rel_path = os.path.relpath(root, INPUT_ROOT)
        # Assuming folder structure is Apps/SessionID, extract names for labels
        parts = rel_path.replace('\\', '/').split('/')
        app_id = parts[0] if len(parts) > 0 else "unknown_app"
        session_id = parts[-1] if len(parts) > 0 else "unknown_session"
        
        print(f"\nProcessing Session: {app_id} -> {session_id} ({len(image_files)} frames)")
        
        session_data = []
        
        for filename in image_files:
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(root, filename)
            csv_path = os.path.join(root, base_name + '.csv')
            
            # Base dictionary filled with NaNs (fallback if frame is dropped)
            frame_features = {f"cell_{i}_{j}_{stat}": np.nan 
                              for i in range(GRID_ROWS) for j in range(GRID_COLS) 
                              for stat in ['min', 'max', 'mean', 'std']}
            frame_features['timestamp'] = filename
            frame_features['app_id'] = app_id
            frame_features['session_id'] = session_id
            
            if not os.path.exists(csv_path):
                session_data.append(frame_features)
                continue
                
            try:
                # 1. PREPARE IMAGE & INFERENCE
                original_img = Image.open(img_path).convert("L")
                input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    if output.shape[1] == 1:
                        mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                    else:
                        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # 2. MATCH THERMAL DATA RESOLUTION
                thermal_df = pd.read_csv(csv_path, header=None)
                thermal_data = thermal_df.values
                if mask.shape != thermal_data.shape:
                    mask = cv2.resize(mask, (thermal_data.shape[1], thermal_data.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 3. GEOMETRIC FILTERING (The "Trash" Filter)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    session_data.append(frame_features) # No headset found
                    continue
                    
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                area = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area == 0 or h == 0:
                    session_data.append(frame_features)
                    continue
                    
                solidity = float(area) / hull_area
                aspect_ratio = float(w) / h
                
                # Reject bad frames (rotation > 45 deg or messy mask)
                if not (MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO) or solidity < MIN_SOLIDITY:
                    # Append NaNs to maintain temporal sequence
                    session_data.append(frame_features)
                    continue

                # 4. EXTRACT 8x8 GRID FEATURES
                cell_w, cell_h = w / GRID_COLS, h / GRID_ROWS
                
                for i in range(GRID_ROWS):
                    for j in range(GRID_COLS):
                        cell_name = f"cell_{i}_{j}"
                        
                        # Dynamic boundaries
                        x1, y1 = int(x + j * cell_w), int(y + i * cell_h)
                        x2, y2 = int(x + (j + 1) * cell_w), int(y + (i + 1) * cell_h)
                        
                        cell_mask = mask[y1:y2, x1:x2]
                        cell_temp = thermal_data[y1:y2, x1:x2]
                        
                        valid_pixels = cell_temp[(cell_mask == 1) & (cell_temp != 0) & (~np.isnan(cell_temp))]
                        total_area = (x2 - x1) * (y2 - y1)
                        
                        # Coverage Check: Is cell mostly on the headset? (> 50%)
                        if len(valid_pixels) > (0.5 * total_area) and len(valid_pixels) > 0:
                            frame_features[f"{cell_name}_min"] = np.min(valid_pixels)
                            frame_features[f"{cell_name}_max"] = np.max(valid_pixels)
                            frame_features[f"{cell_name}_mean"] = np.mean(valid_pixels)
                            frame_features[f"{cell_name}_std"] = np.std(valid_pixels)
                        # Else, it naturally stays as NaN from the base dictionary
                        
                session_data.append(frame_features)
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                session_data.append(frame_features) # Ensure row is added on error to keep 1Hz timing
                
        # 5. SAVE SESSION CSV
        if session_data:
            df_session = pd.DataFrame(session_data)
            output_csv_name = os.path.join(OUTPUT_ROOT, f"{app_id}_{session_id}_base.csv")
            df_session.to_csv(output_csv_name, index=False)
            print(f"  -> Saved {output_csv_name}")

    print("Extraction complete.")

if __name__ == "__main__":
    process_data()