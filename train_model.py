from ultralytics import YOLO
import torch, os

def main():
    os.environ["WANDB_DISABLED"] = "true"  # tránh lỗi wandb

    data_path = "data.yaml"                # đổi đúng
    model = YOLO("yolo11n.pt")             # model nhẹ nhất

    use_gpu = torch.cuda.is_available()
    device  = 0 if use_gpu else "cpu"

    # Preset cho 3050 (4GB VRAM)
    epochs   = 18          # nhanh vòng 1; khi ổn nâng 30–50
    imgsz    = 512         # nếu OOM: 448
    batch    = -1          # auto theo VRAM
    workers  = 2           # laptop để thấp cho ổn định
    cache    = True        # cache RAM giúp dataloader nhanh
    amp      = True        # mixed precision

    if not use_gpu:
        # fallback CPU (chậm): giảm cấu hình cho an toàn
        epochs  = 10
        imgsz   = 384
        batch   = 8
        workers = 0
        cache   = False
        amp     = False

    print(f"Using device: {device} | epochs={epochs} imgsz={imgsz}")

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        cache=cache,
        amp=amp,
        patience=5,          # early-stop nhanh
        cos_lr=True,
        close_mosaic=10,
        deterministic=False, # cho phép cudnn.benchmark -> nhanh hơn
        project="runs",
        name="dovat_fast",
        save_period=-1,      # không lưu mỗi epoch (giảm I/O)
        verbose=True,
    )

    print("Best weights:", "runs/dovat_fast/weights/best.pt")

if __name__ == "__main__":
    main()
