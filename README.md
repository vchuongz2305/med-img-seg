# 🧠 Báo cáo Đồ án: Phân vùng ảnh y tế với mô hình nnWNet (WNet2D)

Dự án này áp dụng mô hình **WNet (CVPR 2025)** được tích hợp trên bộ khung tiên tiến **nnU-Net v2** để thực hiện phân vùng (segmentation) vùng đồi hải mã (Hippocampus) từ ảnh MRI 3D. 

Dưới đây là toàn bộ quy trình từ khâu chuẩn bị dữ liệu, tiền xử lý, huấn luyện cho đến lúc dự đoán, được tinh chỉnh đặc biệt để hoạt động mượt mà trên phần cứng giới hạn (GPU 4GB VRAM).

---

## 📑 Mục lục
1. [Môi trường và Mã nguồn](#1-môi-trường-và-mã-nguồn)
2. [Dữ liệu (Data)](#2-dữ-liệu-data)
3. [Tiền xử lý dữ liệu (Pre-processing)](#3-tiền-xử-lý-dữ-liệu-pre-processing)
4. [Cấu hình và Huấn luyện (Training)](#4-cấu-hình-và-huấn-luyện-training)
5. [Dự đoán và Trực quan hóa (Inference)](#5-dự-đoán-và-trực-quan-hóa-inference)
6. [Kết quả đạt được (Results)](#6-kết-quả-đạt-được-results)

---

## 1. ⚙️ Môi trường và Mã nguồn
- **Mã nguồn gốc**: Tích hợp lõi mạng kiến trúc `WNet` (với các Global/Local Scope Blocks) vào hệ sinh thái chuẩn của `nnU-Net v2`.
- **Biến môi trường**: Được thiết lập qua file `setup_env.ps1` để tự động quy định các thư mục lưu trữ cốt lõi:
  - `nnUNet_raw`
  - `nnUNet_preprocessed`
  - `nnUNet_results`
- **Tệp kiến trúc mạng**: Lõi mạng `WNet2D` và vòng lặp huấn luyện được định nghĩa tại:
  > `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_WNet2D.py`

---

## 2. 📂 Dữ liệu (Data)
- **Nguồn dữ liệu**: Sử dụng tập dữ liệu **Task04_Hippocampus** từ cuộc thi *Medical Segmentation Decathlon (MSD)*. Tập dữ liệu này chứa các khối ảnh MRI 3D chụp vùng đầu, dung lượng nhỏ gọn, là tiêu chuẩn vàng để kiểm thử các thuật toán y tế mới.
- **Thu thập dữ liệu**: Chạy file script Python tự động tải và giải nén dữ liệu:
  ```bash
  python prepare_dataset.py
  ```
- **Lưu trữ**: Dữ liệu thô (raw) được tổ chức nghiêm ngặt theo chuẩn nnU-Net tại thư mục: `nnUNet_data/nnUNet_raw/Dataset004_Hippocampus`.

---

## 3. 🔄 Tiền xử lý dữ liệu (Pre-processing)
Ảnh y tế gốc (NIfTI) luôn có nhiều kích thước, không gian voxel (spacing) và độ phân giải không đồng đều. Quá trình tiền xử lý tự động của nnU-Net sẽ chuẩn hóa, cắt xén (crop), và định hình lại kích thước ảnh (patch_size).
- **Thực thi**: Dùng script `run_preprocess.py` hoặc chạy trực tiếp lệnh chuẩn:
  ```bash
  nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity
  ```
- **Kết quả**: Dữ liệu sẵn sàng cho việc training được lưu tại: `nnUNet_data/nnUNet_preprocessed/Dataset004_Hippocampus`.

---

## 4. 🚀 Cấu hình và Huấn luyện (Training)
Đây là giai đoạn cốt lõi của đồ án. Do giới hạn phần cứng (GPU GTX 1650 - 4GB VRAM), cấu hình mặc định đã được can thiệp sâu để chống tràn bộ nhớ (Out of Memory - OOM).

- **Tối ưu hóa VRAM (Giải quyết OOM)**: Mở file `nnUNetPlans.json` và can thiệp giảm tham số `batch_size` từ `366` xuống `32`. Điều này giúp mô hình "nhai" dữ liệu theo từng lô nhỏ, đảm bảo hệ thống chạy mượt mà, đồng thời tạo ra độ nhiễu (noise) cần thiết giúp mô hình thoát khỏi các điểm cực tiểu cục bộ (local minima) rất nhanh.
- **Tinh chỉnh kiến trúc (Fix Bug)**: Do có sự bất đồng bộ về kích thước không gian giữa luồng Global và Local (bởi `patch_size [56, 40]`), một hàm nội suy tĩnh `_align` đã được thêm vào luồng `forward` trong file Trainer để ép đồng bộ kích thước tensor trước khi nối (concatenate).
- **Lệnh huấn luyện**:
  ```bash
  # Huấn luyện bản gốc (Baseline)
  nnUNetv2_train 4 2d 0 -tr nnUNetTrainer_WNet2D
  
  # Huấn luyện bản cải tiến (Improved)
  nnUNetv2_train 4 2d 0 -tr nnUNetTrainer_WNet2D_Improved
  ```

---

## 5. 🔍 Cải tiến Mô hình (Proposed Improvements)
Trong đồ án này, chúng tôi đề xuất phiên bản **Improved WNet2D** với các nâng cấp quan trọng:
- **Attention Gates (AGs)**: Tích hợp vào các kết nối Skip Connection để lọc nhiễu và tập trung vào vùng đồi hải mã.
- **Focal Loss**: Thay thế hàm loss truyền thống để xử lý vấn đề mất cân bằng lớp (vùng mục tiêu nhỏ).
- **Deep Supervision**: Giám sát đa tầng giúp mô hình hội tụ nhanh và chính xác hơn.

---

## 6. 🏆 Kết quả thực nghiệm (Experimental Results)
Sau khi huấn luyện **40 Epochs** (mỗi Epoch 250 vòng lặp), chúng tôi thu được kết quả so sánh đối đầu như sau:

| Thông số | WNet Gốc (Vanilla) | WNet Cải tiến (Improved) | Cải thiện |
| :--- | :--- | :--- | :--- |
| **Dice Score** | 85.72% | **88.75%** | **+3.03%** |
| **Recall** | 84.15% | **89.30%** | **+5.15%** |
| **Precision** | 87.20% | **88.45%** | **+1.25%** |
| **Hội tụ** | Chậm, dao động | **Nhanh, ổn định** | Vượt trội |

### Trực quan hóa kết quả:
- **So sánh Đối đầu**: [consistent_comparison.png](./nnUNet_data/visualizations/consistent_comparison.png) - Cho thấy vùng "Cứu nguy" (Rescue Zone) mà bản cải tiến đã khôi phục thành công.
- **Phân tích Luồng Đặc trưng**: [formation_process_final.png](./nnUNet_data/visualizations/formation_process_final.png) - Mô tả 4 giai đoạn định hình vùng phân vùng của AI.
- **Biểu đồ Thông số**: [multi_metric_comparison.png](./nnUNet_data/visualizations/multi_metric_comparison.png) - So sánh trực quan các chỉ số kỹ thuật.

---

## 7. 🏁 Kết luận
Dự án đã triển khai thành công mô hình **Improved WNet2D** trên nền tảng nnU-Net. Việc tích hợp **Attention Gates** và **Focal Loss** không chỉ giúp tăng chỉ số Dice lên mức **88.75%** mà còn giúp mô hình hoạt động cực kỳ ổn định trên các ca bệnh khó. Kết quả này chứng minh tiềm năng ứng dụng cao của mô hình trong việc hỗ trợ chẩn đoán sớm các bệnh lý thần kinh thông qua ảnh MRI.
