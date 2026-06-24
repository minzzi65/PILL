import os
from collections import defaultdict

# ===== 경로 설정 =====
LABEL_DIR = r'C:\Users\Administrator\pill\base\test\labels'
# ====================

def main():
    if not os.path.exists(LABEL_DIR):
        print(f"경로를 찾을 수 없습니다: {LABEL_DIR}")
        return

    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    class_count = defaultdict(int)  # 클래스별 이미지 수

    for txt_file in txt_files:
        txt_path = os.path.join(LABEL_DIR, txt_file)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            if lines:
                # 첫 번째 줄의 첫 번째 값을 클래스 ID로 가져옴
                cls = int(lines[0].split()[0])
                class_count[cls] += 1
        except Exception as e:
            print(f"파일 읽기 오류 ({txt_file}): {e}")

    # 1. 전체 데이터 범위 정의 (0번 ~ 106번)
    START_CLASS = 0
    END_CLASS = 106
    full_class_range = set(range(START_CLASS, END_CLASS + 1))
    existing_classes = set(class_count.keys())

    # 2. 결과 데이터 추출
    missing_classes = sorted(list(full_class_range - existing_classes))
    under10_classes = {k: v for k, v in sorted(class_count.items()) if v <= 10}

    # ===== 결과 출력 =====
    print(f"총 이미지(텍스트 파일) 수: {len(txt_files)}개")
    print(f"데이터가 존재하는 총 클래스 수: {len(class_count)}개")
    print(f"0~106번 중 데이터가 없는 총 클래스 수: {len(missing_classes)}개")
    print("-" * 40)

    # A. 존재하지 않는 클래스 목록 출력
    print(f"\n===== 존재하지 않는 클래스 목록 (총 {len(missing_classes)}개) =====")
    if missing_classes:
        print(", ".join(map(str, missing_classes)))
    else:
        print("모든 클래스의 데이터가 존재합니다.")

    # B. 10개 이하인 클래스 목록 출력 (5개 미만 별도 표시)
    print(f"\n===== 10개 이하인 클래스 목록 (총 {len(under10_classes)}개) =====")
    for cls, cnt in sorted(under10_classes.items()):
        marker = " (★ 5개 미만 위험)" if cnt < 5 else ""
        print(f"  클래스 {cls:3d}: {cnt:4d}개{marker}")

if __name__ == '__main__':
    main()