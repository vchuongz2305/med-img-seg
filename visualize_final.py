
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def visualize_4_cols(img_path, vanilla_path, improved_path, output_path):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(vanilla_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(improved_path))

        # Tìm lát cắt có vùng segmentation lớn nhất
        z_slice = np.argmax(np.sum(improved, axis=(1, 2)))
        
        # Zoom cực cận
        coords = np.array(np.where(improved[z_slice] > 0))
        if coords.size == 0: return
        margin = 15
        y_min, y_max = max(0, coords[0].min() - margin), min(img.shape[1], coords[0].max() + margin)
        x_min, x_max = max(0, coords[1].min() - margin), min(img.shape[2], coords[1].max() + margin)

        def get_roi(data): return data[z_slice, y_min:y_max, x_min:x_max]

        roi_img = get_roi(img)
        roi_v = get_roi(vanilla)
        roi_imp = get_roi(improved)

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 1. Ảnh gốc
        axes[0].imshow(roi_img, cmap='gray')
        axes[0].set_title("1. Original MRI", fontsize=15, fontweight='bold')
        
        # 2. Bản cũ (Xanh dương)
        axes[1].imshow(roi_img, cmap='gray')
        mask_v = np.ma.masked_where(roi_v == 0, roi_v)
        axes[1].imshow(mask_v, cmap='winter', alpha=0.8) # Xanh dương/lục
        axes[2].set_title("2. Vanilla WNet", fontsize=15, fontweight='bold', color='blue')

        # 3. Bản mới (Đỏ)
        axes[2].imshow(roi_img, cmap='gray')
        mask_imp = np.ma.masked_where(roi_imp == 0, roi_imp)
        axes[2].imshow(mask_imp, cmap='autumn', alpha=0.8) # Đỏ/vàng
        axes[2].set_title("3. Improved WNet", fontsize=15, fontweight='bold', color='red')

        # 4. Bản mới + Vùng Cứu (Highlight Vàng)
        axes[3].imshow(roi_img, cmap='gray')
        # Vẽ bản mới mờ mờ phía dưới
        axes[3].imshow(mask_imp, cmap='autumn', alpha=0.4)
        # Highlight vùng cứu (Improved - Vanilla)
        rescue = ((roi_imp > 0).astype(float) - (roi_v > 0).astype(float))
        rescue[rescue < 0] = 0
        mask_rescue = np.ma.masked_where(rescue == 0, rescue)
        axes[3].imshow(mask_rescue, cmap='Wistia', alpha=0.9) # Màu vàng sáng rực
        axes[3].set_title("4. Improved + Rescue Zone", fontsize=15, fontweight='bold', color='orange')

        for ax in axes: ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Success: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_res = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
    raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
    out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
    
    case_id = "017" # Lấy mẫu tiêu biểu nhất
    visualize_4_cols(
        f"{raw_dir}/imagesTr/hippocampus_{case_id}_0000.nii.gz",
        f"{base_res}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz",
        f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz",
        f"{out_dir}/final_4_col_comparison.png"
    )
