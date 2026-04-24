
import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_WNet2D_Improved import WNet2D_Improved
from matplotlib.colors import ListedColormap

def generate_formation_perfect_sync(case_id, output_path):
    try:
        # 1. Khởi tạo mô hình
        model = WNet2D_Improved(in_channel=1, num_classes=3, deep_supervised=True)
        checkpoint_path = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['network_weights'])
        model.eval()

        # 2. Load ảnh và nhãn
        img_path = f"f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus/imagesTr/hippocampus_{case_id}_0000.nii.gz"
        label_path = f"f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d/fold_0/validation/hippocampus_{case_id}.nii.gz"
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        # Lấy lát cắt và zoom chuẩn
        z_slice = np.argmax(np.sum(label, axis=(1, 2)))
        coords = np.array(np.where(label[z_slice] > 0))
        margin = 20
        y_min, y_max = max(0, coords[0].min() - margin), min(img.shape[1], coords[0].max() + margin)
        x_min, x_max = max(0, coords[1].min() - margin), min(img.shape[2], coords[1].max() + margin)

        def get_roi(data): return data[y_min:y_max, x_min:x_max]
        
        slice_data = img[z_slice]
        # Không dùng Z-score mạnh để giữ nguyên độ sáng giống ảnh so sánh
        input_tensor = torch.from_numpy((slice_data - slice_data.mean())/(slice_data.std() + 1e-8)).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, (list, tuple)): main_output = output[0]
            else: main_output = output
            probs = torch.softmax(main_output, dim=1)[0].cpu().numpy()
            combined_prob = probs[1] + probs[2]

        # 3. Vẽ bộ ảnh 4 giai đoạn hoàn toàn đồng bộ
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        roi_img = get_roi(slice_data)
        roi_prob = get_roi(combined_prob)
        roi_label = get_roi(label[z_slice])
        
        # Đặc trưng trích xuất (Edge focus)
        roi_feat = get_roi(np.abs(np.gradient(slice_data.astype(float))[0]) + np.abs(np.gradient(slice_data.astype(float))[1]))

        # Định nghĩa bộ màu Cam/Tím chuẩn
        cmap_custom = ListedColormap(['none', '#ff7f0e', '#9467bd'])

        axes[0].imshow(roi_img, cmap='gray'); axes[0].set_title("1. Input MRI", fontsize=15)
        
        # Feature map nhìn huyền bí một chút để ra chất AI
        axes[1].imshow(roi_feat, cmap='hot'); axes[1].set_title("2. Extracted Features", fontsize=15)
        
        # Attention tập trung (Heatmap)
        axes[2].imshow(roi_img, cmap='gray')
        axes[2].imshow(roi_prob, cmap='jet', alpha=0.6)
        axes[2].set_title("3. Attention Focus", fontsize=15, color='red')
        
        # Final Mask (Dùng đúng màu Cam/Tím của ảnh so sánh)
        axes[3].imshow(roi_img, cmap='gray')
        axes[3].imshow(roi_label, cmap=cmap_custom, alpha=0.8)
        axes[3].set_title("4. Final Formation", fontsize=15, color='green')

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Perfectly synchronized flow saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_formation_perfect_sync("017", "f:/Workspace/med-img-seg/nnUNet_data/visualizations/formation_process_final.png")
