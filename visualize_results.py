
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(image_path, label_path, pred_path, output_path):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))

        z_slice = np.argmax(np.sum(label, axis=(1, 2)))
        img_slice = img[z_slice]
        label_slice = label[z_slice]
        pred_slice = pred[z_slice]

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(img_slice, cmap='gray'); plt.title("Original MRI"); plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(img_slice, cmap='gray')
        plt.imshow(label_slice, alpha=0.5, cmap='Greens'); plt.title("Ground Truth"); plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(img_slice, cmap='gray')
        plt.imshow(pred_slice, alpha=0.5, cmap='Reds'); plt.title("WNet Improved Prediction"); plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(img_slice, cmap='gray')
        overlay = np.zeros((*label_slice.shape, 3))
        overlay[label_slice > 0] = [0, 1, 0]
        overlay[pred_slice > 0] = [1, 0, 0]
        plt.imshow(overlay, alpha=0.4); plt.title("Overlay (G:GT, R:Pred)"); plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Success: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    base_dir = "f:/Workspace/med-img-seg/nnUNet_data"
    raw_dir = f"{base_dir}/nnUNet_raw/Dataset004_Hippocampus"
    # Point to the validation folder where we found results
    pred_dir = f"{base_dir}/nnUNet_results/Dataset004_Hippocampus/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation"
    output_vis_dir = f"{base_dir}/visualizations"
    
    if not os.path.exists(output_vis_dir): os.makedirs(output_vis_dir)

    # Validation cases we found
    cases = ["hippocampus_017", "hippocampus_019", "hippocampus_033"]
    
    for case_id in cases:
        img_p = f"{raw_dir}/imagesTr/{case_id}_0000.nii.gz"
        lbl_p = f"{raw_dir}/labelsTr/{case_id}.nii.gz"
        prd_p = f"{pred_dir}/{case_id}.nii.gz"
        
        if os.path.exists(prd_p) and os.path.exists(img_p):
            visualize_prediction(img_p, lbl_p, prd_p, f"{output_vis_dir}/{case_id}_improved_result.png")
        else:
            print(f"File missing for {case_id}")
