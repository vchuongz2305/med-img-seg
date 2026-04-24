
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def compare_models(img_path, label_path, vanilla_path, improved_path, output_path):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(vanilla_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(improved_path))

        z_slice = np.argmax(np.sum(label, axis=(1, 2)))
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        # 1. Original
        axes[0].imshow(img[z_slice], cmap='gray')
        axes[0].set_title("Original MRI", fontsize=16)
        axes[0].axis('off')

        # 2. Ground Truth
        axes[1].imshow(img[z_slice], cmap='gray')
        axes[1].imshow(label[z_slice], alpha=0.5, cmap='Greens')
        axes[1].set_title("Ground Truth", fontsize=16)
        axes[1].axis('off')

        # 3. Vanilla Result
        axes[2].imshow(img[z_slice], cmap='gray')
        axes[2].imshow(vanilla[z_slice], alpha=0.5, cmap='Blues')
        axes[2].set_title("Vanilla (84.2%)", fontsize=16)
        axes[2].axis('off')

        # 4. Improved Result
        axes[3].imshow(img[z_slice], cmap='gray')
        axes[3].imshow(improved[z_slice], alpha=0.5, cmap='Reds')
        axes[3].set_title("Improved (88.1%)", fontsize=16)
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    base_results = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
    vanilla_dir = f"{base_results}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation"
    improved_dir = f"{base_results}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation"
    raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
    output_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"

    case_id = "hippocampus_017" # Lấy mẫu điển hình
    compare_models(
        f"{raw_dir}/imagesTr/{case_id}_0000.nii.gz",
        f"{raw_dir}/labelsTr/{case_id}.nii.gz",
        f"{vanilla_dir}/{case_id}.nii.gz",
        f"{improved_dir}/{case_id}.nii.gz",
        f"{output_dir}/head_to_head_comparison.png"
    )
