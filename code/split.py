import os
import shutil
from tqdm import tqdm
 
# ===== 경로 설정 =====
IMAGE_DIR  = r'C:\Users\Administrator\pill\base1\640_crop\image'
YOLO_DIR   = r'C:\Users\Administrator\pill\base1\640_crop\labels'
 
TRAIN_IMAGE_DIR = r'C:\Users\Administrator\pill\base1\split\train\images'
TRAIN_LABEL_DIR = r'C:\Users\Administrator\pill\base1\split\train\labels'
TEST_IMAGE_DIR  = r'C:\Users\Administrator\pill\base1\split\test\images'
TEST_LABEL_DIR  = r'C:\Users\Administrator\pill\base1\split\test\labels'
# ====================
 
def find_image(image_dir, stem):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(image_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None
 
def count_objects(txt_path):
    """txt 파일에서 객체 수 카운트"""
    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return len(lines)
 
def main():
    # 출력 폴더 생성
    for d in [TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR]:
        os.makedirs(d, exist_ok=True)
 
    txt_files = [f for f in os.listdir(YOLO_DIR) if f.endswith('.txt') and not f.startswith('_')]
    print(f"총 txt 파일 수: {len(txt_files)}")
 
    single = 0
    multi = 0
    skip_no_image = 0
    skip_empty = 0
 
    for txt_file in tqdm(txt_files, desc="분류 중"):
        stem = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(YOLO_DIR, txt_file)
 
        # 객체 수 확인
        obj_count = count_objects(txt_path)
 
        # 빈 파일 (객체 없음) 스킵
        if obj_count == 0:
            skip_empty += 1
            continue
 
        # 대응 이미지 탐색
        image_path = find_image(IMAGE_DIR, stem)
        if image_path is None:
            skip_no_image += 1
            continue
 
        img_ext = os.path.splitext(image_path)[1]
 
        # 단일(1개) → train / 다중(2개 이상) → test
        if obj_count == 1:
            dst_image = os.path.join(TRAIN_IMAGE_DIR, stem + img_ext)
            dst_label = os.path.join(TRAIN_LABEL_DIR, txt_file)
            single += 1
        else:
            dst_image = os.path.join(TEST_IMAGE_DIR, stem + img_ext)
            dst_label = os.path.join(TEST_LABEL_DIR, txt_file)
            multi += 1
 
        shutil.copy2(image_path, dst_image)
        shutil.copy2(txt_path, dst_label)
 
    # 결과 요약
    print("\n===== 분류 완료 =====")
    print(f"단일 객체 → train:  {single}개")
    print(f"다중 객체 → test:   {multi}개")
    print(f"빈 파일 스킵:       {skip_empty}개")
    print(f"이미지 없음 스킵:   {skip_no_image}개")
    print(f"전체 처리:          {single + multi}개")
 
if __name__ == '__main__':
    main()