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
        self.title("Hệ Thống Nhận Biết Đồ Vật")
        self.geometry("1200x700")
        self.configure(bg="#1f2933")
        
        # Biến chính
        self.model = None
        self.cap = None
        self.is_running = False
        self.file_path = ""
        
        self.load_model()
        self.create_ui()

    def load_model(self):
        """Tải YOLO model"""
        try:
            model_path = os.path.join("runs", "dovat_fast", "weights", "best.pt")
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print("✅ Model loaded")
            else:
                print(f"⚠️ Không tìm thấy model tại: {model_path}")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
    
    def create_ui(self):
        """Tạo giao diện"""
        # Header
        header = tk.Frame(self, bg="#111827", height=60)
        header.pack(fill=tk.X)
        tk.Label(header, text="HỆ THỐNG NHẬN BIẾT ĐỒ VẬT", 
                bg="#111827", fg="white", 
                font=("Segoe UI", 18, "bold")).pack(pady=15)
        
        # Main container
        main = tk.Frame(self, bg="#1f2933")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Display
        left = tk.Frame(main, bg="#111827")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.display = tk.Label(left, bg="black")
        self.display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(left, bg="#111827")
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Chọn File", "#2563eb", self.select_file),
            ("Nhận Diện", "#16a34a", self.detect),
            ("Webcam", "#ea580c", self.start_webcam),
            ("Dừng", "#6b7280", self.stop)
        ]
        
        for text, color, cmd in buttons:
            tk.Button(btn_frame, text=text, bg=color, fg="white",
                     font=("Segoe UI", 11, "bold"), command=cmd
                     ).pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # Right: Log
        right = tk.Frame(main, bg="#111827", width=350)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        tk.Label(right, text="LỊCH SỬ", bg="#111827", fg="white",
                font=("Segoe UI", 12, "bold")).pack(pady=10)
        
        self.log = scrolledtext.ScrolledText(right, bg="#020617", fg="#e5e7eb",
                                            font=("Consolas", 9), borderwidth=0)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Status bar
        self.status = tk.Label(self, text="Sẵn sàng", bg="#020617", 
                              fg="#9ca3af", anchor="w", font=("Segoe UI", 10))
        self.status.pack(fill=tk.X)
    
    def log_msg(self, msg):
        """Ghi log"""
        time = datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{time}] {msg}\n")
        self.log.see(tk.END)
    
    def select_file(self):
        """Chọn ảnh/video"""
        path = filedialog.askopenfilename(
            title="Chọn file",
            filetypes=[("Hỗ trợ", "*.png *.jpg *.jpeg *.mp4 *.avi *.mkv")]
        )
        if not path:
            return
            
        self.file_path = path
        
        # Preview nếu là ảnh
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(path)
            img.thumbnail((900, 550))
            photo = ImageTk.PhotoImage(img)
            self.display.configure(image=photo)
            self.display.image = photo
        
        self.log_msg(f"Đã chọn: {os.path.basename(path)}")
        self.status.config(text=f"File: {os.path.basename(path)}")
    
    def detect(self):
        """Nhận diện ảnh/video"""
        if not self.file_path:
            messagebox.showwarning("Cảnh báo", "Chọn file trước!")
            return
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return
        
        # Video
        if self.file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            if self.is_running:
                self.stop()
            
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không mở được video!")
                return
            
            self.is_running = True
            self.log_msg(f"▶ Video: {os.path.basename(self.file_path)}")
            self.process_video()
            return
        
        # Ảnh
        img = cv2.imread(self.file_path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh!")
            return
        
        results = self.model.predict(img, conf=0.25, verbose=False)
        self.log.delete(1.0, tk.END)
        
        count = 0
        for r in results:
            annotated = r.plot(line_width=2, font_size=12)
            
            for box in r.boxes:
                count += 1
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls]
                self.log_msg(f"✅ {name} - {conf:.0%}")
            
            self.show_frame(annotated)
        
        if count > 0:
            self.log_msg(f"📊 Tổng: {count}")
            self.status.config(text=f"Phát hiện {count} đối tượng")
        else:
            self.log_msg("⚠️ Không phát hiện")
            self.status.config(text="Không phát hiện")
    
    def start_webcam(self):
        """Bật webcam"""
        if self.is_running:
            return
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được webcam!")
            return
        
        self.is_running = True
        self.log_msg("📹 Webcam bật")
        self.status.config(text="Webcam chạy...")
        self.process_video()
    
    def process_video(self):
        """Xử lý video/webcam"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        
        results = self.model.predict(frame, conf=0.25, verbose=False, max_det=50)
        
        count = 0
        for r in results:
            annotated = r.plot(line_width=2, font_size=12)
            count = len(r.boxes)
            self.show_frame(annotated)
        
        self.status.config(text=f"Đang chạy... {count} đối tượng")
        self.after(30, self.process_video)
    
    def stop(self):
        """Dừng"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log_msg("⏹️ Dừng")
        self.status.config(text="Đã dừng")
    
    def show_frame(self, frame):
        """Hiển thị frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = frame_rgb.shape[:2]
        scale = min(900/w, 550/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(img)
        
        self.display.configure(image=photo)
        self.display.image = photo

if __name__ == "__main__":
    try:
        print("🚀 Khởi động ứng dụng...")
        app = ObjectDetectionApp()
        print("✅ Giao diện đã sẵn sàng!")
        app.mainloop()
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {e}")
        import traceback
        traceback.print_exc()