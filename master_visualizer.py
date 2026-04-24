
import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WNet2D_Improved import WNet2D_Improved

def master_visualize(case_id):
    try:
        # 1. Paths
        base_res = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus"
        raw_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus"
        out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
        
        img_path = f"{raw_dir}/imagesTr/hippocampus_{case_id}_0000.nii.gz"
        v_path = f"{base_res}/nnUNetTrainer_WNet2D__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz"
        imp_path = f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz"
        checkpoint_path = f"{base_res}/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/checkpoint_best.pth"

        # 2. Load Data
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        vanilla = sitk.GetArrayFromImage(sitk.ReadImage(v_path))
        improved = sitk.GetArrayFromImage(sitk.ReadImage(imp_path))
        
        # 3. Đồng bộ hóa Logic Lát cắt & Zoom
        z_slice = np.argmax(np.sum(improved, axis=(1, 2)))
        coords = np.array(np.where(improved[z_slice] > 0))
        margin = 20
        y_min, y_max = max(0, coords[0].min() - margin), min(img.shape[1], coords[0].max() + margin)
        x_min, x_max = max(0, coords[1].min() - margin), min(img.shape[2], coords[1].max() + margin)

        def crop(data): return data[y_min:y_max, x_min:x_max]
        
        roi_img = crop(img[z_slice])
        roi_v = crop(vanilla[z_slice])
        roi_imp = crop(improved[z_slice])
        
        cmap_custom = ListedColormap(['none', '#ff7f0e', '#9467bd'])

        # --- A. Tạo ảnh So sánh (Consistent Comparison) ---
        fig1, axes1 = plt.subplots(1, 4, figsize=(24, 6))
        axes1[0].imshow(roi_img, cmap='gray'); axes1[0].set_title("1. Original MRI")
        axes1[1].imshow(roi_img, cmap='gray'); axes1[1].imshow(roi_v, cmap=cmap_custom, alpha=0.7); axes1[1].set_title("2. Vanilla")
        axes1[2].imshow(roi_img, cmap='gray'); axes1[2].imshow(roi_imp, cmap=cmap_custom, alpha=0.7); axes1[2].set_title("3. Improved")
        rescue = ((roi_imp > 0).astype(float) - (roi_v > 0).astype(float))
        rescue[rescue < 0] = 0
        mask_res = np.ma.masked_where(rescue == 0, rescue)
        axes1[3].imshow(roi_img, cmap='gray'); axes1[3].imshow(roi_imp, cmap=cmap_custom, alpha=0.3)
        axes1[3].imshow(mask_res, cmap='Wistia', alpha=1.0); axes1[3].set_title("4. Rescue Zone")
        for ax in axes1: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/consistent_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- B. Tạo ảnh Định hình (Formation Process) ---
        # Load model để lấy Attention Probs
        model = WNet2D_Improved(in_channel=1, num_classes=3, deep_supervised=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['network_weights'])
        model.eval()
        
        # Prepare input
        slice_data = img[z_slice]
        norm_slice = (slice_data - slice_data.mean()) / (slice_data.std() + 1e-8)
        input_t = torch.from_numpy(norm_slice).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(input_t)
            if isinstance(output, (list, tuple)): output = output[0]
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            roi_prob = crop(probs[1] + probs[2])

        fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))
        axes2[0].imshow(roi_img, cmap='gray'); axes2[0].set_title("1. Input MRI")
        # Feature map (dùng gradient của ROI thực tế)
        roi_feat = crop(np.abs(np.gradient(slice_data.astype(float))[0]) + np.abs(np.gradient(slice_data.astype(float))[1]))
        axes2[1].imshow(roi_feat, cmap='inferno'); axes2[1].set_title("2. Features")
        axes2[2].imshow(roi_img, cmap='gray'); axes2[2].imshow(roi_prob, cmap='jet', alpha=0.6); axes2[2].set_title("3. Attention")
        axes2[3].imshow(roi_img, cmap='gray'); axes2[3].imshow(roi_imp, cmap=cmap_custom, alpha=0.8); axes2[3].set_title("4. Result")
        for ax in axes2: ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/formation_process_final.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"DONE: Both images updated for Case {case_id}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    master_visualize("017")
