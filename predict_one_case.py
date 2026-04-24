
import torch
import numpy as np
import os
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def predict_single_case():
    # Cấu hình đường dẫn
    model_training_output_dir = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_results/Dataset004_Hippocampus/nnUNetTrainer_WNet2D_Improved__nnUNetPlans__2d"
    input_file = "f:/Workspace/med-img-seg/nnUNet_data/nnUNet_raw/Dataset004_Hippocampus/imagesTs/hippocampus_002_0000.nii.gz"
    output_file = "f:/Workspace/med-img-seg/nnUNet_data/test_predictions_improved/hippocampus_002.nii.gz"
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Khởi tạo Predictor (Dùng CPU nếu GPU quá tải, nhưng ở đây ưu tiên GPU)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    print("--- Đang nạp mô hình... ---")
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir,
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    print(f"--- Đang dự đoán case: {os.path.basename(input_file)} ---")
    # Dự đoán trực tiếp từ file
    predictor.predict_from_files([[input_file]],
                                 [output_file],
                                 save_probabilities=False,
                                 overwrite=True,
                                 num_processes_preprocessing=1,
                                 num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None,
                                 num_parts=1,
                                 part_id=0)
    print(f"--- Hoàn thành! Kết quả tại: {output_file} ---")

if __name__ == "__main__":
    predict_single_case()
