import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
all_labels = []
co_occ_list = []  # list các set label theo từng ảnh (để tính co-occurrence)
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(img_exts):
        continue

    img_path = os.path.join(images_dir, img_file)

    stem, _ = os.path.splitext(img_file)
    label_path = os.path.join(labels_dir, stem + '.txt')

    # size
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception:
        continue

    # read labels
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for line in lines:
            class_id = int(line.split()[0])
            label = label_map.get(class_id, 'unknown')
            labels.append(label)
            all_labels.append(label)

    num_boxes = len(labels)
    primary_label = labels[0] if labels else 'unknown'

    # co-occurrence (unique labels / ảnh)
    unique_labels = sorted(set([lb for lb in labels if lb != 'unknown']))
    co_occ_list.append(unique_labels)

    data.append({
        'image_file': img_file,
        'width': width,
        'height': height,
        'num_boxes': num_boxes,
        'label': primary_label
    })

df = pd.DataFrame(data)

# ===== CORRELATION MATRIX =====
corr = df[['width', 'height', 'num_boxes']].corr()

# ===== CO-OCCURRENCE MATRIX =====
labels_list = list(label_map.values())
co_occ_matrix = pd.DataFrame(
    np.zeros((len(labels_list), len(labels_list))),
    index=labels_list,
    columns=labels_list
)

for combo in co_occ_list:
    # combo là list label unique trong 1 ảnh
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            l1, l2 = combo[i], combo[j]
            if l1 in co_occ_matrix.index and l2 in co_occ_matrix.columns:
                co_occ_matrix.at[l1, l2] += 1
                co_occ_matrix.at[l2, l1] += 1

# ===== MULTIVARIATE GRAPHICAL (2x3) =====
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Multivariate Analysis - YOLO Dataset (20 classes)', fontsize=16, fontweight='bold')

# (1) Scatter width vs num_boxes (hue label)
plt.subplot(2, 3, 1)
sns.scatterplot(x=df['width'], y=df['num_boxes'], hue=df['label'], palette='viridis', legend=False)
plt.title('width vs num_boxes by label')
plt.xlabel('Width (px)')
plt.ylabel('Number of Boxes')

# (2) Scatter height vs num_boxes (hue label)
plt.subplot(2, 3, 2)
sns.scatterplot(x=df['height'], y=df['num_boxes'], hue=df['label'], palette='viridis', legend=False)
plt.title('height vs num_boxes by label')
plt.xlabel('Height (px)')
plt.ylabel('Number of Boxes')

# (3) Heatmap correlation
plt.subplot(2, 3, 3)
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix')

# (4) Heatmap co-occurrence
plt.subplot(2, 3, 4)
# annot=True với 20 class sẽ rất rối; bạn có thể để False cho đẹp
sns.heatmap(co_occ_matrix, annot=False, cmap='YlGnBu', cbar_kws={'shrink': 0.8})
plt.title('Co-occurrence of Labels')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# (5) Bubble chart width vs height, size=num_boxes, color=label_code
plt.subplot(2, 3, 5)
label_codes = df['label'].astype('category').cat.codes
scatter = plt.scatter(df['width'], df['height'], s=(df['num_boxes'] + 1) * 25,
                      c=label_codes, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Label Code')
plt.title('Bubble: width vs height (size=boxes)')
plt.xlabel('Width (px)')
plt.ylabel('Height (px)')

# (6) Avg num_boxes by label (bar)
plt.subplot(2, 3, 6)
sns.barplot(x='label', y='num_boxes', data=df, estimator='mean', errorbar=None, palette='viridis')
plt.title('Avg num_boxes by label')
plt.xlabel('Label')
plt.ylabel('Avg Number of Boxes')
plt.xticks(rotation=45, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Lưu ảnh nếu cần
plt.savefig("multivariate_plots.png", dpi=300, bbox_inches="tight")

plt.show()
