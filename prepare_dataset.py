import os
import requests
import tarfile
import shutil

# Set environment variables for nnU-Net
base_dir = "f:\\Workspace\\med-img-seg"
os.environ['nnUNet_raw'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_preprocessed")
os.environ['nnUNet_results'] = os.path.join(base_dir, "nnUNet_data", "nnUNet_results")

from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset

def download_and_extract_msd(task_name, target_dir):
    url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{task_name}.tar"
    print(f"Downloading {task_name}...")
    r = requests.get(url, stream=True)
    tar_path = os.path.join(target_dir, f"{task_name}.tar")
    with open(tar_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
    print(f"Extracting {task_name}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    os.remove(tar_path)

if __name__ == "__main__":
    base_dir = "f:\\Workspace\\med-img-seg"
    raw_dir = os.path.join(base_dir, "nnUNet_data", "nnUNet_raw")
    
    # Download Task04_Hippocampus (Smallest MSD dataset)
    task_name = "Task04_Hippocampus"
    download_and_extract_msd(task_name, raw_dir)
    
    # Convert to nnU-Net v2 format
    # nnU-Net v2 uses DatasetXXX_Name format
    print("Converting dataset to nnU-Net v2 format...")
    convert_msd_dataset(os.path.join(raw_dir, task_name), 4)
    
    print("Dataset prepared successfully in nnUNet_raw/Dataset004_Hippocampus")
