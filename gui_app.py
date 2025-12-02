import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from datetime import datetime
import os

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hệ Thống Nhận Biết Đồ Vật")
        self.geometry("1200x700")
        self.configure(bg="#1f2933")
        
        # Khởi tạo biến
        self.model = None
        self.cap = None
        self.is_running = False
        self.duongdan = ""
        
        # Tải model
        self.load_model()
        
        # Tạo giao diện
        self.create_ui()
        
    def load_model(self):
        """Tải model YOLO"""
        model_path = os.path.join("runs", "dovat2", "weights", "best.pt")
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                print("Model đã được tải thành công!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải model: {e}")
        else:
            messagebox.showwarning("Cảnh báo", f"Không tìm thấy file model tại: {model_path}")
    
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
        
        # Left: Video/Image display
        left = tk.Frame(main, bg="#111827")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.lbl_display = tk.Label(left, bg="black")
        self.lbl_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(left, bg="#111827", height=70)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame, text="Chọn Ảnh/Video", bg="#2563eb", fg="white",
                 font=("Segoe UI", 11, "bold"), command=self.chon_file,
                 cursor="hand2").pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        tk.Button(btn_frame, text="Kiểm Tra Ảnh", bg="#16a34a", fg="white",
                 font=("Segoe UI", 11, "bold"), command=self.kiem_tra_anh,
                 cursor="hand2").pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        tk.Button(btn_frame, text="Bật Camera", bg="#ea580c", fg="white",
                 font=("Segoe UI", 11, "bold"), command=self.bat_camera,
                 cursor="hand2").pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        tk.Button(btn_frame, text="Dừng", bg="#6b7280", fg="white",
                 font=("Segoe UI", 11, "bold"), command=self.dung,
                 cursor="hand2").pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        
        # Right: Log
        right = tk.Frame(main, bg="#111827", width=350)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        
        tk.Label(right, text="LỊCH SỬ NHẬN DIỆN", bg="#111827", fg="white",
                font=("Segoe UI", 12, "bold")).pack(pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            right, bg="#020617", fg="#e5e7eb",
            font=("Consolas", 9), borderwidth=0
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Status bar
        self.status = tk.Label(self, text="Sẵn sàng", bg="#020617", 
                              fg="#9ca3af", anchor="w", 
                              font=("Segoe UI", 10))
        self.status.pack(fill=tk.X)
    
    def log(self, message):
        """Ghi log"""
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{time_str}] {message}\n")
        self.log_text.see(tk.END)
    
    def chon_file(self):
        """Chọn file ảnh hoặc video"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=[("File hỗ trợ", "*.png *.jpg *.jpeg *.mp4 *.avi")]
        )
        
        if not file_path:
            return
        
        self.duongdan = file_path
        
        # Hiển thị ảnh nếu là file ảnh
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(file_path)
            img.thumbnail((900, 550))
            photo = ImageTk.PhotoImage(img)
            self.lbl_display.configure(image=photo)
            self.lbl_display.image = photo
            
        self.log(f"Đã chọn file: {os.path.basename(file_path)}")
        self.status.config(text=f"File: {os.path.basename(file_path)}")
    
    def kiem_tra_anh(self):
        """Kiểm tra và nhận diện ảnh"""
        if not self.duongdan:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa được tải!")
            return
        
        # Đọc ảnh
        img = cv2.imread(self.duongdan)
        if img is None:
            self.log("Không thể đọc ảnh")
            return
        
        # Dự đoán
        results = self.model.predict(img, verbose=False)
        
        # Đếm đối tượng
        dem = 0
        self.log_text.delete(1.0, tk.END)
        
        for r in results:
            # Vẽ kết quả
            img_result = r.plot(line_width=2, font_size=12)
            
            # Đếm và log
            for box in r.boxes:
                dem += 1
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                self.log(f"Phát hiện: {name} - Độ tin cậy: {conf:.2f}")
            
            # Hiển thị
            self.hien_thi_frame(img_result)
        
        if dem > 0:
            self.log(f"Tổng số đối tượng phát hiện: {dem}")
            self.status.config(text=f"Phát hiện {dem} đối tượng")
        else:
            self.log("Không phát hiện đối tượng nào")
            self.status.config(text="Không phát hiện đối tượng")
    
    def bat_camera(self):
        """Bật camera"""
        if self.is_running:
            return
        
        if not self.model:
            messagebox.showerror("Lỗi", "Model chưa được tải!")
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        self.is_running = True
        self.log("Đã bật camera")
        self.status.config(text="Camera đang chạy...")
        self.xu_ly_camera()
    
    def xu_ly_camera(self):
        """Xử lý từng frame từ camera"""
        if not self.is_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.dung()
            return
        
        # Dự đoán
        results = self.model.predict(frame, verbose=False)
        
        dem = 0
        for r in results:
            frame = r.plot(line_width=2, font_size=12)
            dem = len(r.boxes)
        
        # Hiển thị
        self.hien_thi_frame(frame)
        self.status.config(text=f"Đang chạy... Phát hiện: {dem} đối tượng")
        
        # Gọi lại sau 30ms
        self.after(30, self.xu_ly_camera)
    
    def dung(self):
        """Dừng camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.log("Đã dừng")
        self.status.config(text="Đã dừng")
    
    def hien_thi_frame(self, frame):
        """Hiển thị frame lên label"""
        # Chuyển BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize để fit
        h, w = frame_rgb.shape[:2]
        max_w, max_h = 900, 550
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Chuyển sang ImageTk
        img = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(img)
        
        self.lbl_display.configure(image=photo)
        self.lbl_display.image = photo

if __name__ == "__main__":
    app = App()
    app.mainloop()