
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_consistent(img_path, vanilla_path, improved_path, output_path):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(vanilla_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(improved_path))

        z_slice = np.argmax(np.sum(improved, axis=(1, 2)))
        coords = np.array(np.where(improved[z_slice] > 0))
        if coords.size == 0: return
        margin = 20
        y_min, y_max = max(0, coords[0].min() - margin), min(img.shape[1], coords[0].max() + margin)
        x_min, x_max = max(0, coords[1].min() - margin), min(img.shape[2], coords[1].max() + margin)

        def get_roi(data): return data[z_slice, y_min:y_max, x_min:x_max]
        roi_img = get_roi(img)
        roi_v = get_roi(vanilla)
        roi_imp = get_roi(improved)

        # Định nghĩa bộ màu chuẩn: 0: transparent, 1: Orange, 2: Purple
        cmap_custom = ListedColormap(['none', '#ff7f0e', '#9467bd'])

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 1. MRI
        axes[0].imshow(roi_img, cmap='gray'); axes[0].set_title("1. Original MRI", fontsize=15)
        
        # 2. Vanilla (Dùng chung bộ màu Cam/Tím)
        axes[1].imshow(roi_img, cmap='gray')
        axes[1].imshow(roi_v, cmap=cmap_custom, alpha=0.7)
        axes[1].set_title("2. Vanilla Result", fontsize=15)

        # 3. Improved (Dùng chung bộ màu Cam/Tím)
        axes[2].imshow(roi_img, cmap='gray')
        axes[2].imshow(roi_imp, cmap=cmap_custom, alpha=0.7)
        axes[2].set_title("3. Improved Result", fontsize=15)

        # 4. Highlight vùng Cứu (Màu Vàng Chanh)
        axes[3].imshow(roi_img, cmap='gray')
        axes[3].imshow(roi_imp, cmap=cmap_custom, alpha=0.3) # Làm mờ bản mới
        rescue = ((roi_imp > 0).astype(float) - (roi_v > 0).astype(float))
        rescue[rescue < 0] = 0
        mask_rescue = np.ma.masked_where(rescue == 0, rescue)
        axes[3].imshow(mask_rescue, cmap='Wistia', alpha=1.0) # Vàng chanh rực rỡ
        axes[3].set_title("4. Rescue Zones (Bright)", fontsize=15, fontweight='bold')

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Success: {output_path}")
    except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    base_res = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
    raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
    out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
    case_id = "017"
    visualize_consistent(
        f"{raw_dir}/imagesTr/hippocampus_{case_id}_0000.nii.gz",
        f"{base_res}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz",
        f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz",
        f"{out_dir}/consistent_comparison.png"
    )
