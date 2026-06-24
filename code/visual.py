import os
import random
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
 
# ===== 경로 설정 =====
IMAGE_DIR  = r'C:\Users\Administrator\pill\exclude_107\image'
YOLO_DIR   = r'C:\Users\Administrator\pill\exclude_107\yolo_labels'
OUTPUT_PATH = r'C:\Users\Administrator\pill\exclude_107\verify_5.png'
# ====================
 
COLORS = ['red', 'lime', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'white']
 
def find_image(image_dir, stem):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(image_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None
 
def draw_yolo_boxes(image_path, txt_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    W, H = img.size
    draw = ImageDraw.Draw(img)
 
    with open(txt_path, 'r') as f:
        lines = f.read().strip().split('\n')
 
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
 
        x1 = int((cx - nw / 2) * W)
        y1 = int((cy - nh / 2) * H)
        x2 = int((cx + nw / 2) * W)
        y2 = int((cy + nh / 2) * H)
 
        color = COLORS[i % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
        draw.text((x1 + 6, y1 + 6), f'cls:{cls}', fill=color)
 
    return img
 
def main():
    txt_files = [f for f in os.listdir(YOLO_DIR) if f.endswith('.txt') and not f.startswith('_')]
 
    # 박스가 있는 파일만 필터링
    valid = []
    for txt_file in txt_files:
        txt_path = os.path.join(YOLO_DIR, txt_file)
        with open(txt_path, 'r') as f:
            if f.read().strip():
                valid.append(txt_file)
 
    # 5개 랜덤 선택
    samples = random.sample(valid, min(5, len(valid)))
    print(f"선택된 파일: {samples}")
 
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
 
    for i, txt_file in enumerate(samples):
        stem = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(YOLO_DIR, txt_file)
        image_path = find_image(IMAGE_DIR, stem)
 
        if image_path is None:
            axes[i].set_title(f'이미지 없음\n{stem}', fontsize=8)
            axes[i].axis('off')
            continue
 
        img = draw_yolo_boxes(image_path, txt_path)
        axes[i].imshow(img)
        axes[i].set_title(stem, fontsize=8)
        axes[i].axis('off')
 
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"저장 완료: {OUTPUT_PATH}")
    plt.show()
 
if __name__ == '__main__':
    main()