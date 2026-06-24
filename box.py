import os
import glob

def count_boxes_in_txt(filepath):
    """txt 파일에서 빈 줄을 제외한 박스(줄)의 개수를 셉니다."""
    if not os.path.exists(filepath):
        return 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # 공백이나 줄바꿈만 있는 빈 줄은 제외하고 카운트
    valid_boxes = [line for line in lines if line.strip()]
    return len(valid_boxes)

def evaluate_box_counts(gt_dir, pred_dir):
    """정답(GT)과 예측(Pred) 디렉토리 내의 파일들을 비교하여 Box 개수를 검증합니다."""
    
    # 정답 폴더에 있는 모든 txt 파일 목록 가져오기
    gt_files = glob.glob(os.path.join(gt_dir, '*.txt'))
    
    if not gt_files:
        print("정답 폴더(gt_dir)에 txt 파일이 없습니다. 경로를 다시 확인해주세요.")
        return

    total_images = len(gt_files)
    exact_match_count = 0
    over_predict_count = 0  # 정답보다 많이 찾은 경우 (False Positive 의심)
    under_predict_count = 0 # 정답보다 적게 찾은 경우 (False Negative 의심)
    
    mismatch_list = [] # 불일치한 파일들의 상세 내역

    print(f"총 {total_images}개의 이미지에 대해 Box 개수 검증을 시작합니다...\n")
    print("-" * 50)

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        
        # 각각의 파일에서 Box 개수 추출
        gt_count = count_boxes_in_txt(gt_path)
        pred_count = count_boxes_in_txt(pred_path)
        
        # 개수 비교
        if gt_count == pred_count:
            exact_match_count += 1
        else:
            if pred_count > gt_count:
                over_predict_count += 1
            else:
                under_predict_count += 1
                
            mismatch_list.append({
                'filename': filename,
                'gt': gt_count,
                'pred': pred_count,
                'diff': pred_count - gt_count
            })

    # === 결과 출력 ===
    match_rate = (exact_match_count / total_images) * 100
    
    print(f"📊 [Box 개수 검증 결과 요약]")
    print(f"  - 전체 테스트 이미지: {total_images}장")
    print(f"  - 개수 완벽 일치 (Exact Match): {exact_match_count}장 ({match_rate:.2f}%)")
    print(f"  - 초과 예측 (Over-predict, 과탐지): {over_predict_count}장")
    print(f"  - 미달 예측 (Under-predict, 미탐지): {under_predict_count}장")
    print("-" * 50)
    
    # 개수가 틀린 경우, 분석을 위해 상위 10개만 예시로 출력
    if mismatch_list:
        print("\n🔍 [불일치 상세 내역 (일부)]")
        # 차이가 많이 나는 순으로 정렬
        mismatch_list.sort(key=lambda x: abs(x['diff']), reverse=True)
        
        for item in mismatch_list[:10]:
            status = "초과 예측 🔺" if item['diff'] > 0 else "미달 예측 🔻"
            print(f"  - {item['filename']}: 정답 {item['gt']}개 / 예측 {item['pred']}개 ({status})")

# ==========================================
# 실행 부분 (경로를 민지님의 환경에 맞게 수정하세요)
# ==========================================
if __name__ == "__main__":
    # TODO: 실제 정답 라벨 txt 파일들이 있는 폴더 경로
    GROUND_TRUTH_DIR = "./datasets/test/labels" 
    
    # TODO: 모델이 예측하여 생성한 txt 파일들이 있는 폴더 경로 (예: runs/detect/predict/labels)
    PREDICTION_DIR = "./runs/detect/exp/labels" 
    
    evaluate_box_counts(GROUND_TRUTH_DIR, PREDICTION_DIR)