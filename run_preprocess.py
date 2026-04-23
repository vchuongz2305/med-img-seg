import os
import torch

# Set environment variables for nnU-Net
base_dir = "f:\\Workspace\\med-img-seg"
os.environ['nnUNet_raw'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_preprocessed")
os.environ['nnUNet_results'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_results")

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess

if __name__ == "__main__":
    try:
        dataset_id = 4
        print(f"--- Starting Preprocessing for Dataset {dataset_id} ---")
        
        print(f"Step 1: Extracting fingerprints...")
        extract_fingerprints([dataset_id], num_processes=1)
        print(f"Step 1 completed.")
        
        print(f"Step 2: Planning experiments...")
        plan_experiments([dataset_id])
        print(f"Step 2 completed.")
        
        print(f"Step 3: Preprocessing for 2d configuration...")
        preprocess([dataset_id], configurations=['2d'], num_processes=(1,))
        print(f"Step 3 completed.")
        
        print("--- Preprocessing finished SUCCESSFULLY ---")
    except Exception as e:
        print(f"--- Error during preprocessing: {e} ---")
        import traceback
        traceback.print_exc()
