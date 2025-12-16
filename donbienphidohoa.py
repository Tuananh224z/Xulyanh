import os
import pandas as pd
from PIL import Image
from scipy import stats

# Đường dẫn đến dataset YOLOv11 (thay bằng đường dẫn thực tế)
data_dir = './train'  # Ví dụ: 'dataset/train/images' và 'dataset/train/labels'
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

# Map class_id thành nhãn - DỰA TRÊN data.yaml CỦA BẠN
# nc: 20
# names: ['Cai - coc', 'Giay', 'ban-phim', 'book', 'but - viet', 'cai - ghe',
#         'cai - o', 'cai -but', 'cai-to', 'chia - khoa', 'con -chuot', 'cuc-sac',
#         'dong - ho', 'kinh - mat', 'lap - top', 'manhinh-maytinh',
#         'quyen-sach', 'sac - dien -thoai', 'tai - nghe', 'thesinhvien']

label_map = {
    0: 'Cai - coc',
    1: 'Giay',
    2: 'ban-phim',
    3: 'book',
    4: 'but - viet',
    5: 'cai - ghe',
    6: 'cai - o',
    7: 'cai -but',
    8: 'cai-to',
    9: 'chia - khoa',
    10: 'con -chuot',
    11: 'cuc-sac',
    12: 'dong - ho',
    13: 'kinh - mat',
    14: 'lap - top',
    15: 'manhinh-maytinh',
    16: 'quyen-sach',
    17: 'sac - dien -thoai',
    18: 'tai - nghe',
    19: 'thesinhvien'
}

# Load metadata
data = []
all_labels = []  # Để tính counts cho tất cả nhãn xuất hiện (xử lý multi-label)

for img_file in os.listdir(images_dir):
    if img_file.endswith(('.jpg', '.png')):
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')  # Giả sử extension jpg/png
        label_path = os.path.join(labels_dir, label_file)
        
        # Lấy kích thước image
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Lấy labels
        num_boxes = 0
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                num_boxes = len(lines)
                for line in lines:
                    class_id = int(line.split()[0])
                    label = label_map.get(class_id, 'unknown')
                    labels.append(label)
                    all_labels.append(label)  # Thêm vào list tổng để tính counts
        
        # Primary label cho mỗi image (lấy label đầu tiên nếu multi)
        primary_label = labels[0] if labels else 'unknown'
        
        data.append({
            'image_file': img_file,
            'width': width,
            'height': height,
            'num_boxes': num_boxes,
            'label': primary_label  # Để phân tích univariate, nhưng counts dùng all_labels
        })

df = pd.DataFrame(data)

# Hàm tính thống kê cho một biến định lượng
def univariate_stats(var):
    mean_val = df[var].mean()
    median_val = df[var].median()
    mode_val = df[var].mode()[0] if not df[var].mode().empty else 'No mode'
    range_val = df[var].max() - df[var].min()
    var_val = df[var].var()
    std_val = df[var].std()
    skew_val = df[var].skew()
    kurt_val = df[var].kurtosis()
    
    # Ngoại lệ IQR
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[var] < Q1 - 1.5 * IQR) | (df[var] > Q3 + 1.5 * IQR)]
    num_outliers = len(outliers)
    
    print(f"Univariate Non-Graphical for '{var}':")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Mode: {mode_val}")
    print(f"Range: {range_val:.2f}")
    print(f"Variance: {var_val:.2f}")
    print(f"Standard Deviation: {std_val:.2f}")
    print(f"Skewness: {skew_val:.2f}")
    print(f"Kurtosis: {kurt_val:.2f}")
    print(f"Number of Outliers: {num_outliers}")

# Chạy cho các biến định lượng
print()
univariate_stats('width')
print()
univariate_stats('height')
print()
univariate_stats('num_boxes')

# Cho biến categorical 'label' (dùng all_labels để counts đầy đủ, xử lý multi-label)
label_series = pd.Series(all_labels)
counts = label_series.value_counts()
mode_label = label_series.mode()[0] if not label_series.mode().empty else 'No mode'
print("\nUnivariate Non-Graphical for 'label':")
print(f"Counts: {counts.to_dict()}")
print(f"Mode: {mode_label}")
