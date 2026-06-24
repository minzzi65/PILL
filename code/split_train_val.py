import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm
 
# ===== 경로 설정 =====
SRC_IMAGE_DIR = r'C:\Users\Administrator\pill\base1\split\train\images'
SRC_LABEL_DIR = r'C:\Users\Administrator\pill\base1\split\train\labels'
 
TRAIN_IMAGE_DIR = r'C:\Users\Administrator\pill\base1\split1\train\images'
TRAIN_LABEL_DIR = r'C:\Users\Administrator\pill\base1\split1\train\labels'
VAL_IMAGE_DIR   = r'C:\Users\Administrator\pill\base1\split1\val\images'
VAL_LABEL_DIR   = r'C:\Users\Administrator\pill\base1\split1\val\labels'
# ====================
 
# 1장짜리 클래스 → train만, val 비워도 됨
TRAIN_ONLY_CLASSES = {25, 51}
 
random.seed(42)
 
def find_image(image_dir, stem):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(image_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None
 
def copy_pair(stem, src_image_dir, src_label_dir, dst_image_dir, dst_label_dir):
    """이미지 + 라벨 쌍을 목적지로 복사"""
    image_path = find_image(src_image_dir, stem)
    label_path = os.path.join(src_label_dir, stem + '.txt')
 
    if image_path:
        ext = os.path.splitext(image_path)[1]
        shutil.copy2(image_path, os.path.join(dst_image_dir, stem + ext))
    if os.path.exists(label_path):
        shutil.copy2(label_path, os.path.join(dst_label_dir, stem + '.txt'))
 
def main():
    # 출력 폴더 생성
    for d in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, VAL_IMAGE_DIR, VAL_LABEL_DIR]:
        os.makedirs(d, exist_ok=True)
 
    # 클래스별 파일 목록 수집
    txt_files = [f for f in os.listdir(SRC_LABEL_DIR) if f.endswith('.txt')]
    class_to_stems = defaultdict(list)
 
    for txt_file in txt_files:
        stem = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(SRC_LABEL_DIR, txt_file)
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            cls = int(lines[0].split()[0])
            class_to_stems[cls].append(stem)
 
    train_stems = []
    val_stems = []
 
    for cls, stems in class_to_stems.items():
        random.shuffle(stems)
        n = len(stems)
 
        if cls in TRAIN_ONLY_CLASSES:
            # 1장짜리 → train만
            train_stems.extend(stems)
 
        elif n < 5:
            # 5개 미만 → val에 1개, 나머지 train
            val_stems.append(stems[0])
            train_stems.extend(stems[1:])
 
        else:
            # 5개 이상 → 8:2 분할
            n_val = max(1, round(n * 0.2))
            val_stems.extend(stems[:n_val])
            train_stems.extend(stems[n_val:])
 
    # 파일 복사
    print(f"train 복사 중: {len(train_stems)}개")
    for stem in tqdm(train_stems, desc="train"):
        copy_pair(stem, SRC_IMAGE_DIR, SRC_LABEL_DIR,
                  TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
 
    print(f"val 복사 중: {len(val_stems)}개")
    for stem in tqdm(val_stems, desc="val"):
        copy_pair(stem, SRC_IMAGE_DIR, SRC_LABEL_DIR,
                  VAL_IMAGE_DIR, VAL_LABEL_DIR)
 
    # ===== 결과 통계 =====
    def count_stats(image_dir, label_dir):
        images = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
 
        classes = set()
        for lf in labels:
            with open(os.path.join(label_dir, lf), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        classes.add(int(line.split()[0]))
        return len(images), len(labels), sorted(classes)
 
    t_img, t_lbl, t_cls = count_stats(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
    v_img, v_lbl, v_cls = count_stats(VAL_IMAGE_DIR, VAL_LABEL_DIR)
 
    print("\n===== 분할 완료 =====")
    print(f"[train]")
    print(f"  이미지 수:  {t_img}개")
    print(f"  라벨 수:    {t_lbl}개")
    print(f"  클래스 수:  {len(t_cls)}개")
    print(f"  클래스 목록: {t_cls}")
    print()
    print(f"[val]")
    print(f"  이미지 수:  {v_img}개")
    print(f"  라벨 수:    {v_lbl}개")
    print(f"  클래스 수:  {len(v_cls)}개")
    print(f"  클래스 목록: {v_cls}")
    print()
 
    # val에 없는 클래스 확인
    missing_in_val = set(t_cls) - set(v_cls)
    if missing_in_val:
        print(f"val에 없는 클래스: {sorted(missing_in_val)}")
    else:
        print("모든 클래스가 val에 존재합니다.")
 
if __name__ == '__main__':
    main()
 