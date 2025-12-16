from ultralytics import YOLO
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data.yaml"
MODEL_PATH = r"runs/dovat_10ep_fast/weights/best.pt"
RUN_RESULTS_CSV = r"runs/dovat_10ep_fast/results.csv"

def plot_results_from_csv(csv_path: str, out_png: str):
    """Vẽ bảng 2x5 giống results.png từ results.csv (nếu có đủ cột)."""
    if not os.path.exists(csv_path):
        print(f"[SKIP] Không thấy results.csv: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    panels = [
        ("train/box_loss", "train/box_loss"),
        ("train/cls_loss", "train/cls_loss"),
        ("train/dfl_loss", "train/dfl_loss"),
        ("metrics/precision(B)", "metrics/precision(B)"),
        ("metrics/recall(B)", "metrics/recall(B)"),
        ("val/box_loss", "val/box_loss"),
        ("val/cls_loss", "val/cls_loss"),
        ("val/dfl_loss", "val/dfl_loss"),
        ("metrics/mAP50(B)", "metrics/mAP50(B)"),
        ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
    ]

    # chỉ vẽ cột nào tồn tại
    available = [(t, c) for t, c in panels if c in df.columns]

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]
        if i < len(available):
            title, col = available[i]
            y = df[col]

            # vẽ đường results
            ax.plot(y.values, marker='o', linewidth=1, markersize=2, label='results')

            # smooth rolling mean
            y_smooth = y.rolling(window=5, min_periods=1).mean()
            ax.plot(y_smooth.values, linestyle='--', linewidth=1, label='smooth')

            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 1:
                ax.legend(fontsize=7)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved training curves: {out_png}")

def main():
    os.environ["WANDB_DISABLED"] = "true"
    device = 0 if torch.cuda.is_available() else "cpu"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không thấy model: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    # 1) VAL + PLOTS (PR/F1/Confusion + val_batch*_pred/labels)
    # Lưu ra thư mục riêng cho gọn
    val_res = model.val(
        data=DATA_PATH,
        imgsz=448,
        device=device,
        plots=True,

        # thêm vài tham số để “đủ bài” hơn
        save_json=True,     # xuất COCO-style json (nếu dataset hỗ trợ)
        save_txt=True,      # xuất dự đoán dạng .txt
        save_conf=True,     # kèm confidence trong .txt
        verbose=True,

        project="runs",
        name="dovat_10ep_fast_val_full",
        exist_ok=True,
    )

    # Thư mục output thật sự
    save_dir = str(getattr(val_res, "save_dir", ""))
    if save_dir:
        print(f"\n[VAL OUTPUT] {save_dir}")
    else:
        print("\n[WARN] Không lấy được save_dir từ val_res (tuỳ phiên bản ultralytics).")

    # 2) Vẽ lại bảng 2x5 (training curves) từ results.csv của run train
    # và lưu chung vào thư mục val cho dễ nộp
    out_png = os.path.join(save_dir if save_dir else ".", "results_from_csv.png")
    plot_results_from_csv(RUN_RESULTS_CSV, out_png)

    # 3) (Tuỳ chọn) In nhanh metrics cuối để bạn chụp terminal nộp
    try:
        m = val_res.results_dict
        print("\n[FINAL METRICS]")
        for k, v in m.items():
            print(f"{k}: {v}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
