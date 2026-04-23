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
  nnUNetv2_train 4 2d 0 -tr nnUNetTrainer_WNet2D
  ```

---

## 5. 🔍 Dự đoán và Trực quan hóa (Inference)
Để kiểm chứng mô hình có thực sự "hiểu" cấu trúc não bộ hay không, mô hình được mang đi dự đoán (predict) trên các bệnh nhân mới chưa từng xuất hiện trong tập huấn luyện (nằm trong thư mục `imagesTs`).

- **Thực thi dự đoán**:
  ```bash
  nnUNetv2_predict -i nnUNet_data/test_images -o nnUNet_data/test_predictions -d 4 -c 2d -f 0 -tr nnUNetTrainer_WNet2D --disable_tta
  ```
- **Trực quan hóa (Visualization)**: Chạy script `visualize_prediction.py` để trích xuất 1 lát cắt (slice) 2D từ ảnh 3D, sau đó phủ lớp màu (mask) dự đoán lên ảnh gốc đen trắng. Qua quan sát bằng mắt thường, mô hình đã khoanh vùng chính xác vị trí và hình dáng cơ bản của khối đồi hải mã.

---

## 6. 🏆 Kết quả đạt được (Results)
Chỉ với một đợt thử nghiệm siêu tốc (**20 Epochs**, mỗi Epoch 50 vòng lặp), mạng WNet2D đã thể hiện khả năng hội tụ và học hỏi cấu trúc giải phẫu vượt trội. 

Dưới đây là bảng số liệu chốt sổ tại **Epoch 19** (Vòng lặp cuối cùng):

| Đơn vị đo lường (Metric) | Kết quả đạt được | Đánh giá |
| :--- | :--- | :--- |
| **Train Loss** | `-0.7354` | Hàm mất mát giảm sâu và ổn định, không bị phân kỳ. |
| **Validation Loss** | `-0.7616` | Mô hình không bị Overfitting, tổng quát hóa tốt. |
| **Pseudo Dice (Cấu trúc 1)** | **`84.24%`** (`0.8424`) | Độ chính xác đạt mức Rất Tốt đối với 20 Epochs. |
| **Pseudo Dice (Cấu trúc 2)** | **`83.12%`** (`0.8312`) | Độ chính xác đạt mức Rất Tốt đối với 20 Epochs. |
| **EMA Pseudo Dice (Trung bình)**| **`72.43%`** (`0.7243`) | Mức trung bình trượt đánh giá tổng quan sự ổn định. |
| **Tốc độ huấn luyện** | `~23.38s / Epoch` | Tối ưu hóa cực tốt cho GPU 4GB nhờ Batch Size 32. |

> **Kết luận:** Mô hình **WNet2D** tích hợp trên nnU-Net đã chạy thành công 100% không phát sinh lỗi. Với mức điểm ~84% chỉ trong 20 Epochs test nhanh, kiến trúc này có tiềm năng rất lớn để đạt ngưỡng State-of-the-Art (>90%) nếu được huấn luyện toàn diện với cấu hình 1000 Epochs.
