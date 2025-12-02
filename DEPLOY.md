# Bước 5 & 6 (đơn giản hóa cho người mới)

Tài liệu này giải thích nhanh hai bước trong pipeline:

- Bước 5 — Đánh giá (Evaluation):
  - Mục tiêu: kiểm tra mô hình trên tập `valid`/`test` để biết chất lượng (ví dụ: mAP, precision, recall).
  - Cách đơn giản: sử dụng tính năng có sẵn của Ultralytics.

  Ví dụ (dùng `model_utils.py`):

```powershell
# Chạy đánh giá
python model_utils.py eval runs/dovat2/weights/best.pt data.yaml
```

Ultralytics sẽ in ra các metric cơ bản (mAP, precision, recall). Không cần tự cài metric từ đầu.

- Bước 6 — Triển khai (Deployment):
  - Mục tiêu: chuyển mô hình sang định dạng phù hợp để chạy nhanh trên thiết bị/đám mây (ONNX, TorchScript, TensorRT...).
  - Cách đơn giản: xuất mô hình bằng `.export()` rồi dùng runtime phù hợp.

Ví dụ (xuất ONNX):

```powershell
python model_utils.py export runs/dovat2/weights/best.pt --format onnx --outdir exported
```

Sau khi có file ONNX, bạn có thể dùng `onnxruntime` để chạy inference hoặc chuyển tiếp sang TensorRT.

Gợi ý cho người mới:
- Khi học: làm đúng 1 script `eval` và 1 script `export` như trên là đủ.
- Không cần lúc đầu tối ưu quá: focus vào hiểu dữ liệu, augmentation, và việc đo metric.
