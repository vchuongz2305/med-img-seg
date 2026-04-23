# 📑 Kế hoạch thực hiện Dự án nnWNet (CVPR 2025)

Dưới đây là lộ trình chi tiết để bạn có thể clone, chạy và cải tiến mô hình nnWNet trên ảnh y tế.

## 🚀 Tiến độ hiện tại

- [x] Clone mã nguồn `nnUNet` và `nnWNet`.
- [x] Tích hợp Trainers của `nnWNet` vào `nnUNet`.
- [x] Tạo cấu trúc thư mục dữ liệu.
- [x] Cài đặt thư viện (Đã cài đặt PyTorch GPU + dependencies).
- [x] Chuẩn bị tập dữ liệu mẫu (Hippocampus).
- [/] Chạy tiền xử lý (Đang đạt 63%).
- [ ] Huấn luyện thử nghiệm.

## 🛠 Hướng dẫn chi tiết (Mục tiêu 1)

### 1. Cài đặt môi trường
Tôi đang thực hiện cài đặt tự động. Nếu bạn muốn tự chạy, hãy đảm bảo đã cài đặt:
- Python 3.10+
- PyTorch (Bản có hỗ trợ CUDA)
- `nnunetv2`, `timm`, `acvl_utils`

### 2. Thiết lập Biến môi trường
nnU-Net yêu cầu 3 đường dẫn quan trọng. Tôi đã tạo file `setup_env.ps1`. Bạn hãy chạy lệnh sau trong PowerShell mỗi khi bắt đầu phiên làm việc:
```powershell
.\setup_env.ps1
```

### 3. Chuẩn bị dữ liệu mẫu
Tôi đã chuẩn bị script `prepare_dataset.py` để tải tập dữ liệu **Task04_Hippocampus** (khoảng 20MB) để test nhanh.
Lệnh chạy:
```powershell
python prepare_dataset.py
```

### 4. Huấn luyện (Train)
Sau khi có dữ liệu, ta sẽ chạy lệnh huấn luyện với Trainer của nnWNet:
```powershell
# Chạy tiền xử lý
nnUNetv2_plan_and_preprocess -d 4 --verify_dataset_integrity

# Chạy huấn luyện (2D)
nnUNetv2_train 4 2d 0 -tr nnUNetTrainer_WNet2D
```

---

## 🧠 Hiểu về Kiến trúc (Mục tiêu 2)

**nnWNet** giải quyết xung đột giữa đặc trưng cục bộ (Local) và toàn cục (Global) bằng cách:
1.  Dùng **LSB (Local Scope Blocks)** để truyền tải chi tiết không gian.
2.  Dùng **GSB (Global Scope Bridges)** để kết nối các tầng Encoder-Decoder bằng Transformer.
3.  **W-Shape:** Cấu trúc mạng lồng nhau giúp thông tin luân chuyển liên tục hơn U-Net truyền thống.

## 💡 Hướng cải tiến đề xuất
- Thử nghiệm với **Swin Transformer** trong GSB.
- Áp dụng **Mixed Precision Training** để tiết kiệm VRAM trên GPU 4GB của bạn.
- Tinh chỉnh **Loss Function** (ví dụ: Tversky Loss cho dữ liệu mất cân bằng).
