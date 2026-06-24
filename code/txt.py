import os
import json
from PIL import Image, ImageOps
from tqdm import tqdm
 
 
# ===== 경로 설정 (여기만 수정하면 됩니다) =====
JSON_DIR   = r'C:\Users\Administrator\pill\exclude_107\label'
IMAGE_DIR  = r'C:\Users\Administrator\pill\exclude_107\image'
OUTPUT_DIR = r'C:\Users\Administrator\pill\exclude_107\yolo_labels'
# =============================================
 
 
def get_image_size(image_path):
    """EXIF 회전 보정 후 실제 이미지 크기 반환"""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.size  # (width, height)
 
 
def convert_box_to_yolo(x, y, w, h, img_w, img_h):
    """절대 좌표 (좌상단 x, y, 너비, 높이) → YOLO 정규화 좌표"""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh
 
 
def find_image(image_dir, stem):
    """json 파일명(확장자 제외)과 같은 이미지 파일 탐색"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        path = os.path.join(image_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    print(f"총 JSON 파일 수: {len(json_files)}")
 
    success = 0
    skip_no_image = 0
    skip_no_box = 0
    error_files = []
 
    for json_file in tqdm(json_files, desc="변환 중"):
        stem = os.path.splitext(json_file)[0]
        json_path = os.path.join(JSON_DIR, json_file)
 
        # 대응 이미지 탐색
        image_path = find_image(IMAGE_DIR, stem)
        if image_path is None:
            skip_no_image += 1
            error_files.append(f"[이미지 없음] {json_file}")
            continue
 
        # 이미지 크기 (EXIF 보정 포함)
        try:
            img_w, img_h = get_image_size(image_path)
        except Exception as e:
            error_files.append(f"[이미지 오류] {json_file}: {e}")
            continue
 
        # JSON 로드
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                boxes = json.load(f)
        except Exception as e:
            error_files.append(f"[JSON 파싱 오류] {json_file}: {e}")
            continue
 
        if not boxes:
            skip_no_box += 1
            open(os.path.join(OUTPUT_DIR, stem + '.txt'), 'w').close()
            continue
 
        # YOLO txt 작성
        lines = []
        for b in boxes:
            try:
                cx, cy, nw, nh = convert_box_to_yolo(
                    b['x'], b['y'], b['w'], b['h'], img_w, img_h
                )
                lines.append(f"{b['label']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            except KeyError as e:
                error_files.append(f"[키 오류] {json_file}: 키 {e} 없음")
                continue
 
        out_path = os.path.join(OUTPUT_DIR, stem + '.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
 
        success += 1
 
    # 결과 요약
    print("\n===== 변환 완료 =====")
    print(f"성공:           {success}개")
    print(f"이미지 없음:    {skip_no_image}개")
    print(f"박스 없음(빈):  {skip_no_box}개")
    print(f"오류:           {len(error_files) - skip_no_image}개")
    print(f"출력 경로:      {OUTPUT_DIR}")
 
    if error_files:
        log_path = os.path.join(OUTPUT_DIR, '_error_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(error_files))
        print(f"오류 로그:      {log_path}")
 
 
if __name__ == '__main__':
    main()