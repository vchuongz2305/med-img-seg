import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# Đường dẫn đến ảnh gốc và ảnh dự đoán
raw_img_path = r"f:\Workspace\med-img-seg\nnUNet_data\test_images\hippocampus_002_0000.nii.gz"
pred_img_path = r"f:\Workspace\med-img-seg\nnUNet_data\test_predictions\hippocampus_002.nii.gz"

if not os.path.exists(raw_img_path) or not os.path.exists(pred_img_path):
    print("Không tìm thấy file! Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Tải file NIfTI
raw_img = nib.load(raw_img_path).get_fdata()
pred_img = nib.load(pred_img_path).get_fdata()

# Hippocampus là ảnh 3D, ta lấy một lát cắt ở giữa (ví dụ lát cắt số 18 theo trục z)
z_slice = raw_img.shape[2] // 2

plt.figure(figsize=(10, 5))

# Vẽ ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(raw_img[:, :, z_slice], cmap='gray')
plt.title(f'Ảnh MRI gốc (Lát cắt {z_slice})')
plt.axis('off')

# Vẽ ảnh dự đoán (Mask)
plt.subplot(1, 2, 2)
plt.imshow(raw_img[:, :, z_slice], cmap='gray')
plt.imshow(pred_img[:, :, z_slice], cmap='jet', alpha=0.5) # Vẽ đè lớp dự đoán màu đỏ/xanh lên
plt.title(f'WNet2D Dự đoán (20 Epochs)')
plt.axis('off')

plt.tight_layout()
plt.show()
