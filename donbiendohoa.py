import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ===== PATHS =====
data_dir = r'C:\Code\Xulyanh\train'   # <-- sửa thành đúng đường dẫn của bạn
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

# ===== CLASS MAP (nc=20) =====
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

# ===== LOAD METADATA =====
data = []
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(img_exts):
        continue

    img_path = os.path.join(images_dir, img_file)
    stem, _ = os.path.splitext(img_file)
    label_path = os.path.join(labels_dir, stem + '.txt')

    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception:
        continue

    labels = []
    num_boxes = 0
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        num_boxes = len(lines)
        for line in lines:
            class_id = int(line.split()[0])
            labels.append(label_map.get(class_id, 'unknown'))

    primary_label = labels[0] if labels else 'unknown'

    data.append({
        'image_file': img_file,
        'width': width,
        'height': height,
        'num_boxes': num_boxes,
        'label': primary_label
    })

df = pd.DataFrame(data)

# ===== QUICK CHECK (để bạn thấy vì sao plot bị dài) =====
print("Total images:", len(df))
print("num_boxes max:", df['num_boxes'].max())
print("num_boxes value counts (top 10):")
print(df['num_boxes'].value_counts().head(10))

# ===== PLOT SETTINGS =====
sns.set_theme(style="whitegrid")

def plot_num_boxes(df_in: pd.DataFrame, title: str, zoom_max: int | None = None, save_name: str | None = None):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Histogram: bin theo số nguyên (đúng cho biến rời rạc)
    plt.subplot(2, 2, 1)
    maxv = int(df_in['num_boxes'].max())
    bins = np.arange(-0.5, maxv + 1.5, 1)
    sns.histplot(df_in['num_boxes'], bins=bins, kde=True)
    plt.title("Histogram of num_boxes")
    plt.xlabel("Number of Bounding Boxes")
    plt.ylabel("Frequency")
    if zoom_max is not None:
        plt.xlim(-0.5, zoom_max + 0.5)

    # Boxplot
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df_in['num_boxes'])
    plt.title("Boxplot of num_boxes")
    plt.xlabel("Number of Bounding Boxes")
    if zoom_max is not None:
        plt.xlim(-0.5, zoom_max + 0.5)

    # QQ-Plot
    plt.subplot(2, 2, 3)
    qq_data = df_in['num_boxes']
    stats.probplot(qq_data, dist="norm", plot=plt)
    plt.title("QQ Plot for num_boxes")

    # Violin plot
    plt.subplot(2, 2, 4)
    sns.violinplot(y=df_in['num_boxes'])
    plt.title("Violin Plot of num_boxes")
    plt.ylabel("Number of Bounding Boxes")
    if zoom_max is not None:
        plt.ylim(-0.5, zoom_max + 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")

    plt.show()

# 1) FULL: đúng 100% dataset (thấy cả outlier tới ~17)
plot_num_boxes(df, "Univariate Analysis for num_boxes (FULL - your dataset)",
               zoom_max=None, save_name="num_boxes_FULL.png")

# 2) ZOOM: nhìn giống hình mẫu (chỉ là cách nhìn, không đổi dữ liệu)
MAX_SHOW = 4
plot_num_boxes(df[df['num_boxes'] <= df['num_boxes'].max()],  # vẫn dataset gốc, chỉ zoom trục
               f"Univariate Analysis for num_boxes (ZOOM view 0-{MAX_SHOW})",
               zoom_max=MAX_SHOW, save_name="num_boxes_ZOOM.png")
