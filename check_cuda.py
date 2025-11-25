import torch
import subprocess
import sys

print("=== Kiểm tra PyTorch và CUDA ===\n")

# 1️⃣ Kiểm tra phiên bản PyTorch
print("PyTorch version:", torch.__version__)

# 2️⃣ Kiểm tra CUDA có sẵn không
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# 3️⃣ Số GPU
num_gpus = torch.cuda.device_count()
print("Số GPU phát hiện:", num_gpus)

# 4️⃣ Tên GPU (nếu có)
if num_gpus > 0:
    for i in range(num_gpus):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
else:
    print("Không tìm thấy GPU CUDA")

# 5️⃣ Phiên bản CUDA runtime mà PyTorch sử dụng
print("CUDA version used by PyTorch:", torch.version.cuda)

# 6️⃣ Tùy chọn: kiểm tra driver NVIDIA
try:
    driver_info = subprocess.check_output("nvidia-smi", shell=True, text=True)
    print("\n=== NVIDIA Driver Info ===")
    print(driver_info)
except Exception as e:
    print("\nKhông thể truy cập nvidia-smi (chưa cài driver hoặc không có GPU)")
