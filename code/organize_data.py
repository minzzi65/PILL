import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# 경로 설정
base_dir = r"C:\Users\Administrator\pill\public_train\pill"
label_dir = os.path.join(base_dir, "label")
image_dir = os.path.join(base_dir, "image")
output_base = os.path.join(base_dir, "base")

# 분류 결과 저장
single_object = []
multi_object = []

# 모든 라벨 파일 분석
print("라벨 파일 분석 중...")
for label_file in sorted(os.listdir(label_dir)):
    if label_file.endswith('.json'):
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
                # 객체 개수 확인
                num_objects = len(data)
                image_name = label_file.replace('.json', '.png')
                
                if num_objects == 1:
                    single_object.append(image_name)
                else:
                    multi_object.append(image_name)
        except Exception as e:
            print(f"Error reading {label_file}: {e}")

print(f"\n분석 완료:")
print(f"- 단일 객체 이미지: {len(single_object)}개")
print(f"- 다중 객체 이미지: {len(multi_object)}개")

# 폴더 구조 생성
print("\n폴더 구조 생성 중...")
os.makedirs(os.path.join(output_base, "train", "image"), exist_ok=True)
os.makedirs(os.path.join(output_base, "train", "label"), exist_ok=True)
os.makedirs(os.path.join(output_base, "val", "image"), exist_ok=True)
os.makedirs(os.path.join(output_base, "val", "label"), exist_ok=True)
os.makedirs(os.path.join(output_base, "test", "image"), exist_ok=True)
os.makedirs(os.path.join(output_base, "test", "label"), exist_ok=True)

# train에 단일 객체 이미지 복사
print(f"\ntrain 폴더에 단일 객체 이미지 복사 중... ({len(single_object)}개)")
for image_name in single_object:
    label_name = image_name.replace('.png', '.json')
    src_image = os.path.join(image_dir, image_name)
    src_label = os.path.join(label_dir, label_name)
    
    if os.path.exists(src_image) and os.path.exists(src_label):
        dst_image = os.path.join(output_base, "train", "image", image_name)
        dst_label = os.path.join(output_base, "train", "label", label_name)
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

# test에 다중 객체 이미지 복사
print(f"test 폴더에 다중 객체 이미지 복사 중... ({len(multi_object)}개)")
for image_name in multi_object:
    label_name = image_name.replace('.png', '.json')
    src_image = os.path.join(image_dir, image_name)
    src_label = os.path.join(label_dir, label_name)
    
    if os.path.exists(src_image) and os.path.exists(src_label):
        dst_image = os.path.join(output_base, "test", "image", image_name)
        dst_label = os.path.join(output_base, "test", "label", label_name)
        shutil.copy2(src_image, dst_image)
        shutil.copy2(src_label, dst_label)

print("\n완료!")
print(f"✓ base/train: {len(single_object)}개 이미지 (단일 객체)")
print(f"✓ base/test: {len(multi_object)}개 이미지 (다중 객체)")
print(f"✓ base/val: 0개 이미지 (향후 사용)")
