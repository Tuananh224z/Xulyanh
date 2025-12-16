import os
import pandas as pd
from PIL import Image
from scipy import stats
from collections import Counter

# Đường dẫn đến dataset YOLO
data_dir = r'C:\Code\Xulyanh\train'   # <-- sửa thành đúng đường dẫn của bạn
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')

# Map class_id thành nhãn (theo data.yaml: nc=20)
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
all_labels = []  # list tổng labels (multi-label)

img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(img_exts):
        continue

    img_path = os.path.join(images_dir, img_file)

    stem, _ = os.path.splitext(img_file)
    label_file = stem + '.txt'
    label_path = os.path.join(labels_dir, label_file)

    # Lấy kích thước image
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception:
        continue

    # Lấy labels
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

    # Primary label cho mỗi image (lấy label đầu tiên nếu multi)
    primary_label = labels[0] if labels else 'unknown'

    data.append({
        'image_file': img_file,
        'width': width,
        'height': height,
        'num_boxes': num_boxes,
        'label': primary_label  # primary label
    })

df = pd.DataFrame(data)

# =========================
# Multivariate Non-Graphical
# =========================
grouped = df.groupby('label')

# Stats cho num_boxes
num_boxes_stats = grouped['num_boxes'].agg(['mean', 'median', 'std', 'count'])
print("Stats for num_boxes by label:\n", num_boxes_stats)

# Stats cho width
width_stats = grouped['width'].agg(['mean', 'median', 'std', 'count'])
print("\nStats for width by label:\n", width_stats)

# Stats cho height
height_stats = grouped['height'].agg(['mean', 'median', 'std', 'count'])
print("\nStats for height by label:\n", height_stats)

# Ma trận tương quan giữa các biến định lượng
corr_matrix = df[['width', 'height', 'num_boxes']].corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# =========================
# Co-occurrence of labels per image (multi-label)
# =========================
co_occ = []
for img_file in df['image_file']:
    stem, _ = os.path.splitext(img_file)
    label_path = os.path.join(labels_dir, stem + '.txt')

    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        img_labels = [label_map.get(int(line.split()[0]), 'unknown') for line in lines]
        if img_labels:
            co_occ.append(tuple(sorted(set(img_labels))))
        else:
            co_occ.append(('unknown',))
    else:
        co_occ.append(('unknown',))

co_occ_counts = Counter(co_occ)

print("\n" + "="*80)
print("CO-OCCURRENCE OF LABELS PER IMAGE")
print("="*80)
for labels_tuple, count in co_occ_counts.most_common():
    labels_str = " + ".join(labels_tuple)
    print(f"{labels_str:60} : {count:5} images")
print("="*80)

# =========================
# t-test: so sánh num_boxes giữa 2 class phổ biến nhất (auto)
# =========================
label_counts = df['label'].value_counts()
if len(label_counts) >= 2:
    a, b = label_counts.index[0], label_counts.index[1]
    group1 = df[df['label'] == a]['num_boxes']
    group2 = df[df['label'] == b]['num_boxes']

    # Welch's t-test (ổn hơn nếu variance khác nhau)
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    print(f"\nt-test for num_boxes between '{a}' and '{b}':")
    print(f"t-stat: {t_stat:.2f}, p-value: {p_value:.4f}")
else:
    print("\nNot enough groups for t-test.")
