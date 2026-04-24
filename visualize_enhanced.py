
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def visualize_enhanced(img_path, label_path, vanilla_path, improved_path, output_path):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(vanilla_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(improved_path))

        # Tìm lát cắt có vùng segmentation lớn nhất
        z_slice = np.argmax(np.sum(label, axis=(1, 2)))
        
        # Xác định Bounding Box để Zoom
        coords = np.array(np.where(label[z_slice] > 0))
        if coords.size == 0: return
        y_min, y_max = coords[0].min() - 10, coords[0].max() + 10
        x_min, x_max = coords[1].min() - 10, coords[1].max() + 10

        # Cắt vùng ROI
        def crop(data): return data[z_slice, y_min:y_max, x_min:x_max]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        titles = ["Original MRI (Zoomed)", "Ground Truth", "Vanilla WNet", "Improved WNet"]
        data_list = [crop(img), crop(label), crop(vanilla), crop(improved)]
        cmaps = ['gray', 'Greens', 'Blues', 'Reds']
        alphas = [1.0, 0.5, 0.5, 0.5]

        for i in range(4):
            axes[i].imshow(data_list[0], cmap='gray') # Luôn vẽ ảnh gốc làm nền
            if i > 0:
                axes[i].imshow(data_list[i], alpha=alphas[i], cmap=cmaps[i])
            axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
            axes[i].axis('off')

        plt.suptitle(f"Detailed Comparison - ROI Zoom (Case {os.path.basename(img_path)[:15]})", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced view saved: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_res = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
    raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
    out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
    
    cases = ["017", "033"]
    for c in cases:
        visualize_enhanced(
            f"{raw_dir}/imagesTr/hippocampus_{c}_0000.nii.gz",
            f"{raw_dir}/labelsTr/hippocampus_{c}.nii.gz",
            f"{base_res}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation/hippocampus_{c}.nii.gz",
            f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{c}.nii.gz",
            f"{out_dir}/enhanced_comparison_{c}.png"
        )
