from pathlib import Path
import torch
from ultralytics import YOLO

def evaluate_only() -> None:
    # 1. 경로 설정 (기존 학습 완료된 최고 가중치 경로)
    weights_path = Path(r"C:\Users\Administrator\pill\base\exp_results\sgd_exp04_batch8\weights\best.pt")
    data_path = Path(r"C:\Users\Administrator\pill\base\data.yaml")
    runs_dir = Path(r"C:\Users\Administrator\pill\base\exp_results")
    device = "0" if torch.cuda.is_available() else "cpu"

    if not weights_path.exists():
        print(f"\n[오류] 가중치 파일을 찾을 수 없습니다: {weights_path}")
        print("경로가 정확한지 다시 한번 확인해 주세요.")
        return

    # 2. 평가할 타겟 클래스 ID 리스트 (정확히 51개)
    target_classes = [
        1, 4, 8, 10, 12, 13, 14, 15, 16, 23, 25, 31, 33, 36, 37, 38, 40, 41, 42, 43, 
        45, 46, 47, 48, 50, 51, 54, 55, 56, 57, 58, 64, 68, 70, 72, 75, 78, 79, 85, 
        86, 87, 89, 90, 91, 92, 95, 96, 97, 99, 103, 104
    ]

    print("\n" + "=" * 60)
    print("학습된 모델을 불러옵니다...")
    print("=" * 60)
    
    # 모델 로드
    model = YOLO(str(weights_path))
    
    print("\n지정된 51개 클래스에 대해서만 Test 데이터셋 평가를 시작합니다...\n")
    
    # 3. 모델 평가 (test split 지정 및 타겟 클래스 제한)
    metrics = model.val(
        data=str(data_path),
        split="test",           
        batch=16,               
        device=device,
        classes=target_classes, # 핵심: 이 51개 클래스만 채점하도록 제한
        workers=0,
        project=str(runs_dir),
        name="test_evaluation_exp04", # 새로운 이름으로 결과 저장
        exist_ok=True
    )
    
    # 4. 최종 결과 요약 출력
    print("\n" + "-" * 60)
    print("🎯 [최종 평가 결과] (필터링 적용 완료)")
    print("-" * 60)
    print(f"Precision (P)  : {metrics.box.mp:.4f}")
    print(f"Recall (R)     : {metrics.box.mr:.4f}")
    print(f"mAP@50         : {metrics.box.map50:.4f}")
    print(f"mAP@50-95      : {metrics.box.map:.4f}")
    print("-" * 60)
    print(f"\n상세 지표 및 시각화 파일 저장 경로:")
    print(f"-> {runs_dir / 'test_evaluation_exp04'}")

if __name__ == "__main__":
    evaluate_only()