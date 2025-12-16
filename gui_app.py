import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from datetime import datetime
import os

class ObjectDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Cài đặt cửa sổ
        self.title("Hệ Thống Nhận Biết Đồ Vật")
        self.geometry("1200x700")
        self.configure(bg="#1f2933")
        
        # Biến quan trọng
        self.model = None  # Model AI
        self.cap = None    # Camera
        self.running = False  # Đang chạy?
        self.file_path = ""   # Đường dẫn file
        self.show_confidence = True  # Hiển thị độ tin cậy
        
        # Tải model
        model_path = "runs/dovat_acc_full_run01/weights/best.pt"
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        
        # Tạo giao diện
        self.create_ui()

    def create_ui(self):
        """Tạo giao diện người dùng"""
        
        # === PHẦN 1: TIÊU ĐỀ ===
        title = tk.Label(self, 
                        text="HỆ THỐNG NHẬN BIẾT ĐỒ VẬT",
                        bg="#111827", 
                        fg="white",
                        font=("Segoe UI", 18, "bold"),
                        height=2)
        title.pack(fill=tk.X)
        
        # === PHẦN 2: KHUNG CHÍNH ===
        main_frame = tk.Frame(self, bg="#1f2933")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # KHUNG TRÁI - Hiển thị ảnh/video
        left_frame = tk.Frame(main_frame, bg="#111827")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Màn hình hiển thị
        self.display = tk.Label(left_frame, bg="black")
        self.display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Khung nút bấm
        button_frame = tk.Frame(left_frame, bg="#111827")
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Nút 1: Chọn File
        btn_file = tk.Button(button_frame, 
                            text="Chọn File",
                            bg="#2563eb", 
                            fg="white",
                            font=("Segoe UI", 11, "bold"),
                            command=self.select_file)
        btn_file.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # Nút 2: Nhận Diện
        btn_detect = tk.Button(button_frame,
                              text="Nhận Diện",
                              bg="#16a34a",
                              fg="white",
                              font=("Segoe UI", 11, "bold"),
                              command=self.detect)
        btn_detect.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # Nút 3: Webcam
        btn_webcam = tk.Button(button_frame,
                              text="Webcam",
                              bg="#ea580c",
                              fg="white",
                              font=("Segoe UI", 11, "bold"),
                              command=self.start_webcam)
        btn_webcam.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # Nút 4: Dừng
        btn_stop = tk.Button(button_frame,
                            text="Dừng",
                            bg="#6b7280",
                            fg="white",
                            font=("Segoe UI", 11, "bold"),
                            command=self.stop)
        btn_stop.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # KHUNG PHẢI - Lịch sử và Thống kê
        right_frame = tk.Frame(main_frame, bg="#111827", width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        # Khung thống kê
        stats_frame = tk.Frame(right_frame, bg="#0f172a")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Tiêu đề thống kê
        stats_title = tk.Label(stats_frame,
                              text="📊 THỐNG KÊ",
                              bg="#0f172a",
                              fg="#60a5fa",
                              font=("Segoe UI", 11, "bold"))
        stats_title.pack(pady=5)
        
        # Hiển thị số đối tượng
        self.total_label = tk.Label(stats_frame,
                                   text="Tổng đối tượng: 0",
                                   bg="#0f172a",
                                   fg="#e5e7eb",
                                   font=("Segoe UI", 10))
        self.total_label.pack(pady=2)
        
        # Hiển thị độ tin cậy trung bình
        self.avg_conf_label = tk.Label(stats_frame,
                                      text="Độ tin cậy TB: -",
                                      bg="#0f172a",
                                      fg="#e5e7eb",
                                      font=("Segoe UI", 10))
        self.avg_conf_label.pack(pady=2)
        
        # Tiêu đề lịch sử
        log_title = tk.Label(right_frame,
                            text="📋 LỊCH SỬ PHÁT HIỆN",
                            bg="#111827",
                            fg="white",
                            font=("Segoe UI", 12, "bold"))
        log_title.pack(pady=10)
        
        # Ô text hiển thị lịch sử
        self.log = scrolledtext.ScrolledText(right_frame,
                                            bg="#020617",
                                            fg="#e5e7eb",
                                            font=("Consolas", 9),
                                            borderwidth=0)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # === PHẦN 3: THANH TRẠNG THÁI ===
        self.status = tk.Label(self,
                              text="✅ Sẵn sàng",
                              bg="#020617",
                              fg="#9ca3af",
                              anchor="w",
                              font=("Segoe UI", 10))
        self.status.pack(fill=tk.X)
    
    def log_msg(self, message):
        """Ghi tin nhắn vào lịch sử"""
        time_now = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{time_now}] {message}\n")
        self.log.see(tk.END)  # Cuộn xuống cuối
    
    def update_stats(self, total, avg_conf):
        """Cập nhật thống kê"""
        self.total_label.config(text=f"Tổng đối tượng: {total}")
        if avg_conf > 0:
            self.avg_conf_label.config(text=f"Độ tin cậy TB: {avg_conf:.1%}")
        else:
            self.avg_conf_label.config(text="Độ tin cậy TB: -")
    
    def select_file(self):
        """Chọn file ảnh hoặc video"""
        # Mở hộp thoại chọn file
        path = filedialog.askopenfilename(
            title="Chọn file",
            filetypes=[("Hỗ trợ", "*.png *.jpg *.jpeg *.mp4 *.avi *.mkv")])
        
        if not path:  # Nếu không chọn gì
            return
        
        self.file_path = path  # Lưu đường dẫn
        
        # Nếu là ảnh → hiển thị preview
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(path)
            img.thumbnail((900, 550))  # Resize nhỏ lại
            photo = ImageTk.PhotoImage(img)
            self.display.configure(image=photo)
            self.display.image = photo  # Giữ reference
        
        # Ghi log
        file_name = os.path.basename(path)
        self.log_msg(f"📁 Đã chọn: {file_name}")
        self.status.config(text=f"File: {file_name}")
    
    def detect(self):
        """Nhận diện đồ vật trong file"""
        
        # Kiểm tra đã chọn file chưa
        if not self.file_path:
            messagebox.showwarning("Cảnh báo", "Chọn file trước!")
            return
        
        # Kiểm tra model đã load chưa
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return
        
        # === XỬ LÝ VIDEO ===
        if self.file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            self.stop()  # Dừng cái đang chạy
            self.cap = cv2.VideoCapture(self.file_path)
            self.running = True
            self.log_msg(f"▶️ Video: {os.path.basename(self.file_path)}")
            self.process_video()
            return
        
        # === XỬ LÝ ẢNH ===
        img = cv2.imread(self.file_path)  # Đọc ảnh
        results = self.model.predict(img, conf=0.25, verbose=False)  # Nhận diện
        
        self.log.delete(1.0, tk.END)  # Xóa log cũ
        
        count = 0  # Đếm số đồ vật
        total_conf = 0  # Tổng độ tin cậy
        
        for result in results:
            # Vẽ khung lên ảnh
            annotated_img = result.plot()
            self.show_frame(annotated_img)
            
            # Đếm và ghi log từng đồ vật
            for box in result.boxes:
                count += 1
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                total_conf += confidence
                name = self.model.names[class_id]
                
                # Chọn màu dựa trên độ tin cậy
                if confidence >= 0.8:
                    icon = "🟢"  # Xanh lá - Rất tốt
                elif confidence >= 0.6:
                    icon = "🟡"  # Vàng - Tốt
                else:
                    icon = "🟠"  # Cam - Trung bình
                
                self.log_msg(f"{icon} {name} - {confidence:.1%}")
        
        # Hiển thị tổng kết
        if count > 0:
            avg_conf = total_conf / count
            self.log_msg(f"\n{'='*40}")
            self.log_msg(f"📊 Tổng: {count} đối tượng")
            self.log_msg(f"📈 Độ tin cậy TB: {avg_conf:.1%}")
            self.status.config(text=f"✅ Phát hiện {count} đối tượng (TB: {avg_conf:.1%})")
            self.update_stats(count, avg_conf)
        else:
            self.log_msg("⚠️ Không phát hiện đối tượng")
            self.status.config(text="⚠️ Không phát hiện")
            self.update_stats(0, 0)
    
    def start_webcam(self):
        """Bật webcam để nhận diện real-time"""
        
        if self.running:  # Đang chạy rồi
            return
        
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return
        
        # Mở camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được webcam!")
            return
        
        self.running = True
        self.log_msg("📹 Webcam đã bật")
        self.status.config(text="🔴 Webcam đang chạy...")
        self.process_video()  # Bắt đầu xử lý
    
    def process_video(self):
        """Xử lý từng frame video/webcam"""
        
        if not self.running or not self.cap:
            return
        
        # Đọc 1 frame
        ret, frame = self.cap.read()
        
        if not ret:  # Hết video hoặc lỗi
            self.stop()
            return
        
        # Nhận diện trong frame
        results = self.model.predict(frame, conf=0.25, verbose=False, max_det=50)
        
        # Vẽ và hiển thị
        annotated = results[0].plot()
        count = len(results[0].boxes)
        
        # Tính độ tin cậy trung bình
        if count > 0:
            total_conf = sum([float(box.conf[0]) for box in results[0].boxes])
            avg_conf = total_conf / count
            self.update_stats(count, avg_conf)
            self.status.config(text=f"🔴 Đang chạy... {count} đối tượng (TB: {avg_conf:.1%})")
        else:
            self.update_stats(0, 0)
            self.status.config(text="🔴 Đang chạy... 0 đối tượng")
        
        self.show_frame(annotated)
        
        # Gọi lại sau 30ms (tạo hiệu ứng video)
        self.after(30, self.process_video)
    
    def stop(self):
        """Dừng camera/video"""
        self.running = False
        
        if self.cap:
            self.cap.release()  # Tắt camera
            self.cap = None
        
        self.log_msg("⏹️ Đã dừng")
        self.status.config(text="⏸️ Đã dừng")
        self.update_stats(0, 0)
    
    def show_frame(self, frame):
        """Hiển thị frame lên màn hình"""
        
        # Chuyển từ BGR (OpenCV) sang RGB (Tkinter)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tính toán kích thước phù hợp
        h, w = frame_rgb.shape[:2]
        scale = min(900/w, 550/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Chuyển sang ảnh Tkinter
        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(img)
        
        # Hiển thị
        self.display.configure(image=photo)
        self.display.image = photo  # Giữ reference

# === CHẠY CHƯƠNG TRÌNH ===
if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.mainloop()