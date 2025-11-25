import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from datetime import datetime
import os


class ObjectDetectionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ Thống Nhận Biết Đồ Vật")
        self.root.geometry("1180x680")
        self.root.configure(bg="#1f2933")

        # trạng thái
        self.is_running = False
        self.cap = None
        self.model = None

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        # grid: title / main / status
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)

        # ==== Thanh tiêu đề ====
        title_bar = tk.Frame(self.root, bg="#111827", height=50)
        title_bar.grid(row=0, column=0, sticky="ew")
        title_bar.grid_propagate(False)

        title_label = tk.Label(
            title_bar,
            text="HỆ THỐNG NHẬN BIẾT ĐỒ VẬT",
            bg="#111827",
            fg="#e5e7eb",
            font=("Segoe UI", 16, "bold"),
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=8)

        # ==== Nội dung chính ====
        main = tk.Frame(self.root, bg="#1f2933")
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)  # video
        main.columnconfigure(1, weight=1)  # history

        # ----- Khu video + nút -----
        video_area = tk.Frame(main, bg="#111827")
        video_area.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_area.rowconfigure(0, weight=1)  # video
        video_area.rowconfigure(1, weight=0)  # buttons
        video_area.columnconfigure(0, weight=1)

        # label hiển thị video
        self.video_label = tk.Label(video_area, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # hàng nút
        controls = tk.Frame(video_area, bg="#111827", height=60)
        controls.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls.grid_propagate(False)
        for i in range(4):
            controls.columnconfigure(i, weight=1)

        btn_style = dict(
            font=("Segoe UI", 11, "bold"),
            height=2,
            bd=0,
            activeforeground="#ffffff",
            cursor="hand2",
        )

        self.btn_webcam = tk.Button(
            controls,
            text="Webcam",
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            command=self.start_webcam,
            **btn_style,
        )
        self.btn_webcam.grid(row=0, column=0, sticky="ew", padx=4)

        self.btn_video = tk.Button(
            controls,
            text="Video",
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            command=self.start_video,
            **btn_style,
        )
        self.btn_video.grid(row=0, column=1, sticky="ew", padx=4)

        self.btn_image = tk.Button(
            controls,
            text="Hình ảnh",
            bg="#16a34a",
            fg="#ffffff",
            activebackground="#15803d",
            command=self.start_image,
            **btn_style,
        )
        self.btn_image.grid(row=0, column=2, sticky="ew", padx=4)

        self.btn_stop = tk.Button(
            controls,
            text="Dừng",
            bg="#6b7280",
            fg="#ffffff",
            activebackground="#4b5563",
            command=self.stop,
            **btn_style,
        )
        self.btn_stop.grid(row=0, column=3, sticky="ew", padx=4)

        # ----- Lịch sử -----
        side = tk.Frame(main, bg="#111827")
        side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        side.rowconfigure(1, weight=1)
        side.columnconfigure(0, weight=1)

        history_title = tk.Label(
            side,
            text="LỊCH SỬ NHẬN DIỆN",
            bg="#111827",
            fg="#e5e7eb",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        history_title.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))

        self.history_text = scrolledtext.ScrolledText(
            side,
            width=40,
            font=("Consolas", 9),
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            borderwidth=0,
            wrap=tk.WORD,
        )
        self.history_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # ----- Thanh trạng thái -----
        self.status_label = tk.Label(
            self.root,
            text="Chưa tải mô hình",
            bg="#020617",
            fg="#9ca3af",
            anchor="w",
            font=("Segoe UI", 10),
        )
        self.status_label.grid(row=2, column=0, sticky="ew")

    # ---------- Tiện ích ----------
    def log(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{t}] {msg}\n")
        self.history_text.see(tk.END)

    def set_status(self, text: str):
        self.status_label.config(text=text)

    def load_model_if_needed(self):
        """Chỉ load model 1 lần khi cần."""
        if self.model is not None:
            return

        # 🔥 ĐƯỜNG DẪN TỚI MODEL: runs/dovat2/weights/best.pt
        model_path = os.path.join("runs", "dovat2", "weights", "best.pt")

        if not os.path.exists(model_path):
            self.set_status(f"KHÔNG TÌM THẤY MODEL: {model_path}")
            self.log(f"Lỗi: không tìm thấy file mô hình: {model_path}")
            return

        self.set_status(f"Đang tải mô hình đồ vật ({model_path})...")
        self.log(f"Đang tải mô hình: {model_path}")

        self.model = YOLO(model_path)

        self.set_status("Đã sẵn sàng (model đồ vật)")
        self.log(f"Đã tải mô hình {model_path}")

    # ---------- Nút bấm ----------
    def start_webcam(self):
        if self.is_running:
            return
        self.load_model_if_needed()
        if self.model is None:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.set_status("Không mở được webcam")
            self.log("Lỗi: không mở được webcam")
            return
        self.is_running = True
        self.set_status("Đang nhận diện từ webcam...")
        self.log("Bắt đầu từ webcam")
        self.update_frame()

    def start_video(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả", "*.*")],
        )
        if not path:
            return
        self.load_model_if_needed()
        if self.model is None:
            return

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.set_status("Không mở được video")
            self.log(f"Lỗi: không mở được video: {path}")
            return
        self.is_running = True
        self.set_status(f"Đang nhận diện video: {os.path.basename(path)}")
        self.log(f"Bắt đầu video: {path}")
        self.update_frame()

    def start_image(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="Chọn hình ảnh",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")],
        )
        if not path:
            return
        self.load_model_if_needed()
        if self.model is None:
            return

        self.set_status(f"Nhận diện ảnh: {os.path.basename(path)}")
        self.log(f"Nhận diện ảnh: {path}")

        img = cv2.imread(path)
        if img is None:
            self.set_status("Không đọc được ảnh")
            self.log("Lỗi: không đọc được ảnh")
            return

        annotated, num = self.run_yolo(img)
        self.display_frame(annotated)
        self.log(f"Ảnh: phát hiện {num} đối tượng")

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.set_status("Đã dừng")
        self.log("Dừng nhận diện")

    # ---------- Vòng lặp video ----------
    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.log("Kết thúc video / webcam")
            self.stop()
            return

        annotated, num = self.run_yolo(frame)
        self.display_frame(annotated)

        self.set_status(f"Đang nhận diện... {num} đối tượng")
        self.root.after(30, self.update_frame)

    # ---------- YOLO ----------
    def run_yolo(self, frame):
        """
        Chạy YOLO trên frame, trả về:
        - annotated_frame: ảnh đã vẽ bbox + label
        - num_objects: số đối tượng phát hiện
        """
        results = self.model.predict(
            frame,
            conf=0.25,      # ↓ hạ ngưỡng cho nhạy hơn
            verbose=False,
            max_det=50,
        )
        r = results[0]
        annotated = r.plot()  # YOLO tự vẽ bbox + tên class
        num_objects = len(r.boxes)
        return annotated, num_objects

    # ---------- Hiển thị ----------
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        max_w, max_h = 900, 520
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


def main():
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
