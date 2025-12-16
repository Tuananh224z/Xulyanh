from ultralytics import YOLO
import torch, os
from datetime import datetime

def main():
    os.environ["WANDB_DISABLED"] = "true"

    data_path = "data.yaml"
    use_gpu = torch.cuda.is_available()
    device  = 0 if use_gpu else "cpu"

    # ===== SETUP FULL TRAIN =====
    epochs  = 55
    imgsz   = 640
    batch   = -1
    workers = 4
    cache   = True
    amp     = True

    if not use_gpu:
        epochs  = 40
        imgsz   = 512
        batch   = 8
        workers = 0
        cache   = False
        amp     = False

    print(f"Using device: {device} | epochs={epochs} imgsz={imgsz} batch={batch}")

    # train 10 lần
    for i in range(1, 11):
        run_name = f"dovat_acc_full_run{i:02d}"
        print(f"\n========== START RUN {i}/10: {run_name} ==========")

        # luôn khởi tạo model mới cho mỗi run
        model = YOLO("yolo11n.pt")

        # ===== TRAIN =====
        model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            cache=cache,
            amp=amp,

            patience=20,
            cos_lr=True,
            close_mosaic=10,
            deterministic=False,

            project="runs",
            name=run_name,
            exist_ok=False,

            val=True,    # có validation trong lúc train
            plots=True,  # vẽ loss, lr, metrics
            save=True,
            verbose=True,
        )

        # ===== VAL LẠI ĐỂ LẤY ĐẦY ĐỦ PLOTS =====
        # Bước này giúp sinh đủ:
        # - PR curve
        # - F1 curve
        # - Confusion matrix
        # - Labels.png, etc...
        model.val(
            data=data_path,
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=True,     # bắt buộc để vẽ toàn bộ biểu đồ
            verbose=True,
            project="runs",
            name=run_name,  # dùng chung tên run
            exist_ok=True,  # cho phép ghi thêm vào cùng folder
        )

        print(f"[DONE] Run {i}/10 saved under: runs/ (search {run_name})")

    print("\n✅ Hoàn tất 10 runs. Vào thư mục runs/ để lấy results.png và các biểu đồ của từng run.")

if __name__ == "__main__":
    main()
