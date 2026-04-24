
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def visualize_super(img_path, label_path, vanilla_path, improved_path, output_path):
    try:
        img_itk = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img_itk)
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(vanilla_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(improved_path))

        # Lấy lát cắt trung tâm của vùng label
        z_slice = np.argmax(np.sum(label, axis=(1, 2)))
        
        # Zoom cực cận vào vùng ROI
        coords = np.array(np.where(label[z_slice] > 0))
        if coords.size == 0: return
        margin = 15
        y_min, y_max = max(0, coords[0].min() - margin), min(img.shape[1], coords[0].max() + margin)
        x_min, x_max = max(0, coords[1].min() - margin), min(img.shape[2], coords[1].max() + margin)

        def get_roi(data): return data[z_slice, y_min:y_max, x_min:x_max]

        roi_img = get_roi(img)
        roi_gt = get_roi(label)
        roi_v = get_roi(vanilla)
        roi_imp = get_roi(improved)

        fig, axes = plt.subplots(1, 5, figsize=(25, 6))
        
        # Cấu hình màu sắc
        # 1. MRI Gốc
        axes[0].imshow(roi_img, cmap='gray')
        axes[0].set_title("1. Original MRI (ROI)", fontsize=14, fontweight='bold')
        
        # 2. Ground Truth (Green)
        axes[1].imshow(roi_img, cmap='gray')
        mask_gt = np.ma.masked_where(roi_gt == 0, roi_gt)
        axes[1].imshow(mask_gt, cmap='brg', alpha=0.7) # Màu xanh lá rực rỡ
        axes[1].set_title("2. Ground Truth", fontsize=14, fontweight='bold', color='green')

        # 3. Vanilla (Blue)
        axes[2].imshow(roi_img, cmap='gray')
        mask_v = np.ma.masked_where(roi_v == 0, roi_v)
        axes[2].imshow(mask_v, cmap='cool', alpha=0.7) # Màu xanh dương
        axes[2].set_title("3. Vanilla WNet", fontsize=14, fontweight='bold', color='blue')

        # 4. Improved (Red)
        axes[3].imshow(roi_img, cmap='gray')
        mask_imp = np.ma.masked_where(roi_imp == 0, roi_imp)
        axes[3].imshow(mask_imp, cmap='autumn', alpha=0.7) # Màu đỏ/vàng rực rỡ
        axes[3].set_title("4. Improved WNet", fontsize=14, fontweight='bold', color='red')

        # 5. Error Highlight (Chỉ ra sự cải tiến)
        # Màu vàng: Chỗ bản Improved đúng nhưng bản Vanilla sai/thiếu
        axes[4].imshow(roi_img, cmap='gray')
        diff = (roi_imp > 0).astype(float) - (roi_v > 0).astype(float)
        diff[diff < 0] = 0 # Chỉ lấy phần Improved thêm vào
        mask_diff = np.ma.masked_where(diff == 0, diff)
        axes[4].imshow(mask_diff, cmap='YlOrRd', alpha=0.8)
        axes[4].set_title("5. Improvement Area", fontsize=14, fontweight='bold', color='orange')

        for ax in axes: ax.axis('off')

        plt.suptitle(f"COMPARISON ANALYSIS - Case {os.path.basename(img_path).split('_')[1]}", fontsize=20, y=1.05)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Super visualization saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_res = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
    raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
    out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
    
    cases = ["017", "033", "019"]
    for c in cases:
        visualize_super(
            f"{raw_dir}/imagesTr/hippocampus_{c}_0000.nii.gz",
            f"{raw_dir}/labelsTr/hippocampus_{c}.nii.gz",
            f"{base_res}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation/hippocampus_{c}.nii.gz",
            f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{c}.nii.gz",
            f"{out_dir}/super_comparison_{c}.png"
        )
