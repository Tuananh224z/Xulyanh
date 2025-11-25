from ultralytics import YOLO

def main():
    data_path = "data.yaml"      # data.yaml của bạn

    model = YOLO("yolo11n.pt")   # hoặc "yolov8n.pt"

    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs",
        name="dovat2",            # => runs/detect/dovat/...
    )

if __name__ == "__main__":
    main()
